import pytest
import jax
import jax.numpy as jnp
import numpy as np
import subprocess
import tempfile
import pathlib
import shutil

from gayum.terms import s
import gayum


def test_gam_class_init(mtcars_pl):
    with pytest.raises(TypeError):
        gayum.GAM()
    
    f = gayum.Formula(mtcars_pl, 'mpg', s('hp') + s('wt'))
    _ = gayum.GAM(dist=gayum.dists.Normal(), formula=f)


def test_df_to_jnp(mtcarsGAM, mtcars_pl):
    X, y = mtcars_pl[['hp', 'wt']], mtcars_pl['mpg']
    result1, result2 = mtcarsGAM._to_jnp(X, y)
    assert isinstance(result1, jax.Array)
    assert isinstance(result2, jax.Array)


def test_fit_predict_with_reml(mtcars_pl):
    f = gayum.Formula(mtcars_pl, 'mpg', s('hp') + s('wt'))
    gam = gayum.GAM(dist=gayum.dists.Normal(), formula=f)

    X = mtcars_pl.select(['hp', 'wt'])
    y = mtcars_pl.select('mpg')
    gam.fit(X, y)
    preds = gam.predict(X)

    assert gam.coef_ is not None
    assert gam.lam_ is not None
    assert gam.lam_.shape[0] == 2
    assert jnp.all(gam.lam_ > 0)
    assert preds.shape[0] == mtcars_pl.height
    assert jnp.all(jnp.isfinite(preds))


def test_gam_learns_nonlinear_signal_better_than_linear_baseline():
    rng = np.random.default_rng(42)
    n = 250
    x = np.linspace(0, 1, n)
    noise = rng.normal(0, 0.1, size=n)
    y = np.sin(6 * np.pi * x) + noise

    import polars as pl

    train_df = pl.DataFrame({'x': x, 'y': y})
    f = gayum.Formula(train_df, 'y', s('x'))
    gam = gayum.GAM(formula=f, dist=gayum.dists.Normal())
    gam.fit(train_df.select('x'), train_df.select('y'))

    pred = np.asarray(gam.predict(train_df.select('x')))
    mse_gam = np.mean((y - pred) ** 2)

    X_lin = np.column_stack([np.ones(n), x])
    beta_lin, *_ = np.linalg.lstsq(X_lin, y, rcond=None)
    mse_lin = np.mean((y - X_lin @ beta_lin) ** 2)

    # Require a clear margin to show the smooth captures nonlinear structure.
    assert mse_gam < 0.8 * mse_lin


def test_predictions_align_with_mgcv_when_available():
    rscript = shutil.which('Rscript')
    if rscript is None:
        pytest.skip('Rscript not found; skipping mgcv comparison test')

    rng = np.random.default_rng(7)
    n = 300
    x = np.linspace(0.0, 1.0, n)
    y = np.sin(2 * np.pi * x) + 0.4 * np.cos(7 * np.pi * x) + rng.normal(0, 0.12, n)
    x_pred = np.linspace(0.0, 1.0, 200)

    import polars as pl

    train_df = pl.DataFrame({'x': x, 'y': y})
    pred_df = pl.DataFrame({'x': x_pred})

    # Fit gayum on the same synthetic data and predict on a common grid.
    f = gayum.Formula(train_df, 'y', s('x'))
    gayum_gam = gayum.GAM(formula=f, dist=gayum.dists.Normal())
    gayum_gam.fit(train_df.select('x'), train_df.select('y'))
    pred_gayum = np.asarray(gayum_gam.predict(pred_df.select('x')))

    # Export data for mgcv and run an R script that returns fit + SE on x_pred.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = pathlib.Path(tmp_dir)
        train_csv = tmp / 'train.csv'
        pred_csv = tmp / 'pred.csv'
        out_csv = tmp / 'mgcv_pred.csv'
        script_path = tmp / 'run_mgcv.R'

        np.savetxt(train_csv, np.column_stack([x, y]), delimiter=',', header='x,y', comments='')
        np.savetxt(pred_csv, x_pred[:, None], delimiter=',', header='x', comments='')

        script_path.write_text(
            """
            args <- commandArgs(trailingOnly = TRUE)
            train_path <- args[1]
            pred_path <- args[2]
            out_path <- args[3]

            suppressPackageStartupMessages(library(mgcv))
            train <- read.csv(train_path)
            pred <- read.csv(pred_path)

            fit <- gam(y ~ s(x, bs = 'tp', k = 10), data = train, method = 'REML')
            p <- predict(fit, newdata = pred, se.fit = TRUE, type = 'response')

            out <- data.frame(mu = as.numeric(p$fit), se = as.numeric(p$se.fit))
            write.csv(out, out_path, row.names = FALSE)
            """.strip(),
            encoding='utf-8',
        )

        proc = subprocess.run(
            [rscript, str(script_path), str(train_csv), str(pred_csv), str(out_csv)],
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            if 'there is no package called' in (proc.stderr or '').lower() and 'mgcv' in (proc.stderr or '').lower():
                pytest.skip('R found but mgcv package is unavailable; skipping mgcv comparison test')
            pytest.fail(f'mgcv comparison script failed. stderr: {proc.stderr}')

        mgcv_pred = np.genfromtxt(out_csv, delimiter=',', names=True)

    pred_mgcv = np.asarray(mgcv_pred['mu'])
    se_mgcv = np.asarray(mgcv_pred['se'])

    # Mean-prediction agreement on the same grid.
    corr = np.corrcoef(pred_gayum, pred_mgcv)[0, 1]
    rmse = np.sqrt(np.mean((pred_gayum - pred_mgcv) ** 2))
    assert corr > 0.999
    assert rmse < 0.005

    # Distribution-shape agreement of predicted means (quantile comparison).
    q = np.array([0.1, 0.5, 0.9])
    q_g = np.quantile(pred_gayum, q)
    q_m = np.quantile(pred_mgcv, q)
    assert np.max(np.abs(q_g - q_m)) < 0.01

    # mgcv should return finite uncertainty estimates for mean predictions.
    assert np.all(np.isfinite(se_mgcv))
    assert np.all(se_mgcv >= 0)


def test_multivariate_predictions_align_with_mgcv_when_available():
    rscript = shutil.which('Rscript')
    if rscript is None:
        pytest.skip('Rscript not found; skipping mgcv comparison test')

    rng = np.random.default_rng(11)
    n = 700
    n_pred = 350
    x1 = rng.uniform(0.0, 1.0, n)
    x2 = rng.uniform(0.0, 1.0, n)
    y = np.sin(2 * np.pi * x1) + 0.7 * np.cos(3 * np.pi * x2) + rng.normal(0, 0.1, n)

    x1_pred = rng.uniform(0.0, 1.0, n_pred)
    x2_pred = rng.uniform(0.0, 1.0, n_pred)

    import polars as pl

    train_df = pl.DataFrame({'x1': x1, 'x2': x2, 'y': y})
    pred_df = pl.DataFrame({'x1': x1_pred, 'x2': x2_pred})

    f = gayum.Formula(train_df, 'y', s('x1') + s('x2'))
    gayum_gam = gayum.GAM(formula=f, dist=gayum.dists.Normal())
    gayum_gam.fit(train_df.select(['x1', 'x2']), train_df.select('y'))
    pred_gayum = np.asarray(gayum_gam.predict(pred_df.select(['x1', 'x2'])))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = pathlib.Path(tmp_dir)
        train_csv = tmp / 'train.csv'
        pred_csv = tmp / 'pred.csv'
        out_csv = tmp / 'mgcv_pred.csv'
        script_path = tmp / 'run_mgcv_mv.R'

        np.savetxt(train_csv, np.column_stack([x1, x2, y]), delimiter=',', header='x1,x2,y', comments='')
        np.savetxt(pred_csv, np.column_stack([x1_pred, x2_pred]), delimiter=',', header='x1,x2', comments='')

        script_path.write_text(
            """
            args <- commandArgs(trailingOnly = TRUE)
            train_path <- args[1]
            pred_path <- args[2]
            out_path <- args[3]

            suppressPackageStartupMessages(library(mgcv))
            train <- read.csv(train_path)
            pred <- read.csv(pred_path)

            fit <- gam(y ~ s(x1, bs = 'tp', k = 10) + s(x2, bs = 'tp', k = 10), data = train, method = 'REML')
            p <- predict(fit, newdata = pred, se.fit = TRUE, type = 'response')

            out <- data.frame(mu = as.numeric(p$fit), se = as.numeric(p$se.fit))
            write.csv(out, out_path, row.names = FALSE)
            """.strip(),
            encoding='utf-8',
        )

        proc = subprocess.run(
            [rscript, str(script_path), str(train_csv), str(pred_csv), str(out_csv)],
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            if 'there is no package called' in (proc.stderr or '').lower() and 'mgcv' in (proc.stderr or '').lower():
                pytest.skip('R found but mgcv package is unavailable; skipping mgcv comparison test')
            pytest.fail(f'mgcv comparison script failed. stderr: {proc.stderr}')

        mgcv_pred = np.genfromtxt(out_csv, delimiter=',', names=True)

    pred_mgcv = np.asarray(mgcv_pred['mu'])
    se_mgcv = np.asarray(mgcv_pred['se'])

    diff = pred_gayum - pred_mgcv
    corr = np.corrcoef(pred_gayum, pred_mgcv)[0, 1]
    rmse = np.sqrt(np.mean(diff ** 2))

    assert corr > 0.999
    assert rmse < 0.01
    assert np.max(np.abs(diff)) < 0.02

    q = np.array([0.1, 0.5, 0.9])
    q_g = np.quantile(pred_gayum, q)
    q_m = np.quantile(pred_mgcv, q)
    assert np.max(np.abs(q_g - q_m)) < 0.01

    assert np.all(np.isfinite(se_mgcv))
    assert np.all(se_mgcv >= 0)


def test_mtcars_univariate_aligns_with_mgcv_when_available(mtcars_pd):
    rscript = shutil.which('Rscript')
    if rscript is None:
        pytest.skip('Rscript not found; skipping mgcv comparison test')

    import polars as pl

    train_pd = mtcars_pd[['mpg', 'hp']].copy()
    hp_grid = np.linspace(train_pd['hp'].min(), train_pd['hp'].max(), 200)
    pred_pd = train_pd.iloc[:0].copy()
    pred_pd = pred_pd.assign(hp=hp_grid, mpg=train_pd['mpg'].mean())

    train_pl = pl.from_pandas(train_pd)
    pred_pl = pl.from_pandas(pred_pd)

    f = gayum.Formula(train_pl, 'mpg', s('hp'))
    gayum_gam = gayum.GAM(formula=f, dist=gayum.dists.Normal())
    gayum_gam.fit(train_pl.select('hp'), train_pl.select('mpg'))
    pred_gayum = np.asarray(gayum_gam.predict(pred_pl.select('hp')))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = pathlib.Path(tmp_dir)
        train_csv = tmp / 'train.csv'
        pred_csv = tmp / 'pred.csv'
        out_csv = tmp / 'mgcv_pred.csv'
        script_path = tmp / 'run_mgcv_mtcars_uni.R'

        train_pd.to_csv(train_csv, index=False)
        pred_pd.to_csv(pred_csv, index=False)

        script_path.write_text(
            """
            args <- commandArgs(trailingOnly = TRUE)
            train_path <- args[1]
            pred_path <- args[2]
            out_path <- args[3]

            suppressPackageStartupMessages(library(mgcv))
            train <- read.csv(train_path)
            pred <- read.csv(pred_path)

            fit <- gam(mpg ~ s(hp), data = train, method = 'REML')
            p <- predict(fit, newdata = pred, se.fit = TRUE, type = 'response')

            out <- data.frame(mu = as.numeric(p$fit), se = as.numeric(p$se.fit))
            write.csv(out, out_path, row.names = FALSE)
            """.strip(),
            encoding='utf-8',
        )

        proc = subprocess.run(
            [rscript, str(script_path), str(train_csv), str(pred_csv), str(out_csv)],
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            if 'there is no package called' in (proc.stderr or '').lower() and 'mgcv' in (proc.stderr or '').lower():
                pytest.skip('R found but mgcv package is unavailable; skipping mgcv comparison test')
            pytest.fail(f'mgcv comparison script failed. stderr: {proc.stderr}')

        mgcv_pred = np.genfromtxt(out_csv, delimiter=',', names=True)

    pred_mgcv = np.asarray(mgcv_pred['mu'])

    diff = pred_gayum - pred_mgcv
    corr = np.corrcoef(pred_gayum, pred_mgcv)[0, 1]
    rmse = np.sqrt(np.mean(diff ** 2))
    q90 = np.quantile(np.abs(diff), 0.9)

    assert corr > 0.9999
    assert rmse < 0.03
    assert np.max(np.abs(diff)) < 0.08
    assert q90 < 0.06


def test_mtcars_multivariate_aligns_with_mgcv_when_available(mtcars_pd):
    rscript = shutil.which('Rscript')
    if rscript is None:
        pytest.skip('Rscript not found; skipping mgcv comparison test')

    import polars as pl

    train_pd = mtcars_pd[['mpg', 'hp', 'wt']].copy()
    train_pl = pl.from_pandas(train_pd)

    f = gayum.Formula(train_pl, 'mpg', s('hp') + s('wt'))
    gayum_gam = gayum.GAM(formula=f, dist=gayum.dists.Normal())
    gayum_gam.fit(train_pl.select(['hp', 'wt']), train_pl.select('mpg'))
    pred_gayum = np.asarray(gayum_gam.predict(train_pl.select(['hp', 'wt'])))

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = pathlib.Path(tmp_dir)
        train_csv = tmp / 'train.csv'
        pred_csv = tmp / 'pred.csv'
        out_csv = tmp / 'mgcv_pred.csv'
        script_path = tmp / 'run_mgcv_mtcars_multi.R'

        train_pd.to_csv(train_csv, index=False)
        train_pd.to_csv(pred_csv, index=False)

        script_path.write_text(
            """
            args <- commandArgs(trailingOnly = TRUE)
            train_path <- args[1]
            pred_path <- args[2]
            out_path <- args[3]

            suppressPackageStartupMessages(library(mgcv))
            train <- read.csv(train_path)
            pred <- read.csv(pred_path)

            form <- mpg ~ s(hp) + s(wt)
            fit <- gam(form, data = train, method = 'REML')
            p <- predict(fit, newdata = pred, se.fit = TRUE, type = 'response')

            out <- data.frame(mu = as.numeric(p$fit), se = as.numeric(p$se.fit))
            write.csv(out, out_path, row.names = FALSE)
            """.strip(),
            encoding='utf-8',
        )

        proc = subprocess.run(
            [rscript, str(script_path), str(train_csv), str(pred_csv), str(out_csv)],
            capture_output=True,
            text=True,
        )

        if proc.returncode != 0:
            if 'there is no package called' in (proc.stderr or '').lower() and 'mgcv' in (proc.stderr or '').lower():
                pytest.skip('R found but mgcv package is unavailable; skipping mgcv comparison test')
            pytest.fail(f'mgcv comparison script failed. stderr: {proc.stderr}')

        mgcv_pred = np.genfromtxt(out_csv, delimiter=',', names=True)

    pred_mgcv = np.asarray(mgcv_pred['mu'])

    diff = pred_gayum - pred_mgcv
    corr = np.corrcoef(pred_gayum, pred_mgcv)[0, 1]
    rmse = np.sqrt(np.mean(diff ** 2))
    q90 = np.quantile(np.abs(diff), 0.9)

    assert corr > 0.999
    assert rmse < 0.06
    assert np.max(np.abs(diff)) < 0.15
    assert q90 < 0.08
