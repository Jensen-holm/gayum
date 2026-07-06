from scipy.optimize import minimize as scipy_minimize
from narwhals.typing import Frame
import jax.numpy as jnp
import narwhals as nw
import numpy as np
import jax

from .dists import Dist, Normal
from .formula import Formula


__all__ = ['GAM']


class GAM:
    __slots__ = ['dist', 'formula', 'coef_', 'lam_']

    def __init__(self, formula: Formula, dist: Dist = Normal()):
        assert isinstance(dist, Dist)
        assert isinstance(formula, Formula)

        self.dist: Dist = dist
        self.formula: Formula = formula
        self.coef_ = None
        self.lam_ = None

    @nw.narwhalify
    def _to_jnp(self, *args) -> jax.Array | tuple[jax.Array, ...]:
        """convert dataframes from whatever backend into jax arrays"""

        def __convert(x: Frame) -> jax.Array:
            return jnp.array(x.to_numpy()).squeeze()

        converted = [__convert(x) for x in args]
        if len(converted) == 1:
            return converted[0]
        return tuple(converted)

    @nw.narwhalify
    def _fit_init(self, X: Frame, y: Frame) -> tuple[jax.Array, jax.Array]:
        assert len(y.columns) == 1, f'too many columns in y dataframe |{len(y.columns)}| max is 1'

        self._resolve_term_basis_dimensions(X)

        for t in self.formula.terms:
            t.build(self._to_jnp(X.select(t.col)))

        return self._to_jnp(X, y)

    @nw.narwhalify
    def _resolve_term_basis_dimensions(self, X: Frame) -> None:
        """Resolve per-term basis dimensions before basis construction.

        For thin-plate terms, this follows an mgcv-like default of k=10 for
        1D smooths, while adding a small-sample guard for additive models with
        multiple smooth terms. This keeps the API simple (users can omit k)
        and avoids unstable over-parameterized bases on tiny datasets.
        """
        n_obs = len(X)
        n_terms = len(self.formula.terms)
        min_k = 3

        for t in self.formula.terms:
            if not hasattr(t, 'k'):
                continue

            x = np.asarray(self._to_jnp(X.select(t.col)))
            n_unique = int(np.unique(x).shape[0])
            unique_cap = max(min_k, n_unique - 1)

            if t.k is None:
                default_k = 10
                # Keep enough data support per smooth in small multivariate fits.
                sample_cap = max(min_k, n_obs // max(1, 3 * n_terms))
                t.k = int(max(min_k, min(default_k, sample_cap, unique_cap)))
            else:
                # Respect explicit user k, but cap by unique x support.
                t.k = int(max(min_k, min(int(t.k), unique_cap)))

    def _build_design_matrix(self) -> jax.Array:
        n = self.formula.terms[0].basis_mat.shape[0]
        parts = [jnp.ones((n, 1))] + [t.basis_mat for t in self.formula.terms]
        return jnp.hstack(parts)

    def _build_term_penalties(self) -> list[jax.Array]:
        """Per-term penalty matrices in full coefficient space, unscaled."""
        total = 1 + sum(t.basis_mat.shape[1] for t in self.formula.terms)
        S_terms = []
        offset = 1
        for t in self.formula.terms:
            k = t.basis_mat.shape[1]
            S_j = jnp.zeros((total, total))
            S_j = S_j.at[offset:offset + k, offset:offset + k].set(t.penalty())
            S_terms.append(S_j)
            offset += k
        return S_terms

    def _build_penalty(self, lams: jax.Array, S_terms: list[jax.Array]) -> jax.Array:
        """Combine per-term penalties: S = Σ λⱼ Sⱼ"""
        return sum(lam_j * S_t for lam_j, S_t in zip(lams, S_terms))

    def _irls(
        self,
        X: jax.Array,
        y: jax.Array,
        S: jax.Array,
        max_iter: int = 25,
        tol: float = 1e-7,
    ) -> jax.Array:
        mu = jnp.full_like(y, jnp.maximum(y.mean(), 1e-4))
        beta = jnp.zeros(X.shape[1])

        for _ in range(max_iter):
            eta = self.dist.link(mu)
            d_link = jax.vmap(jax.grad(self.dist.link))(mu)
            V = self.dist.variance(mu)

            z = eta + (y - mu) * d_link
            w = jnp.clip(1.0 / (V * d_link**2), 0, 1e6)

            WX = w[:, None] * X
            beta_new = jnp.linalg.solve(X.T @ WX + S, X.T @ (w * z))
            mu = self.dist.inverse_link(X @ beta_new)

            if jnp.max(jnp.abs(beta_new - beta)) < tol:
                return beta_new
            beta = beta_new

        return beta_new

    def _optimize_reml(
        self,
        X: jax.Array,
        y: jax.Array,
        S_terms: list[jax.Array],
    ) -> jax.Array:
        """Select smoothing parameters by REML.

        For Normal (unknown dispersion): uses the profiled criterion that
        analytically integrates out σ²:

            REML(ρ) = n/2·log(pen_dev/n) + ½log|X'X+S| - ½log|S|₊

        where pen_dev = RSS + β̂'Sβ̂. This has a genuine minimum because
        log(pen_dev)→-∞ as λ→0 is balanced by -½log|S|₊→+∞.

        For Binomial/Poisson (fixed dispersion φ=1): uses the Laplace
        approximation, which is correct when φ is known.
        """
        dist = self.dist
        S_stack = jnp.stack(S_terms)  # (n_terms, p, p)
        n = y.shape[0]

        if getattr(dist, 'profile_dispersion', False):
            # Profiled Gaussian REML — β doesn't depend on σ², so solve directly.
            def reml_score(rho: jax.Array) -> jax.Array:
                S = jnp.einsum('j,jkl->kl', jnp.exp(rho), S_stack)
                beta = jnp.linalg.solve(X.T @ X + S, X.T @ y)
                mu = X @ beta
                pen_dev = jnp.sum((y - mu) ** 2) + beta @ S @ beta
                _, log_det_H = jnp.linalg.slogdet(X.T @ X + S)
                eigvals = jnp.linalg.eigvalsh(S)
                log_det_S_plus = jnp.sum(
                    jnp.where(eigvals > 1e-10, jnp.log(jnp.maximum(eigvals, 1e-10)), 0.0)
                )
                return n / 2 * jnp.log(pen_dev / n) + 0.5 * log_det_H - 0.5 * log_det_S_plus
        else:
            # Laplace approximate REML for non-Gaussian (φ = 1 known).
            mu0 = jnp.full(y.shape, jnp.maximum(y.mean(), 1e-4))

            def reml_score(rho: jax.Array) -> jax.Array:
                S = jnp.einsum('j,jkl->kl', jnp.exp(rho), S_stack)

                def irls_step(mu, _):
                    eta = dist.link(mu)
                    d_link = jax.vmap(jax.grad(dist.link))(mu)
                    V = dist.variance(mu)
                    z = eta + (y - mu) * d_link
                    w = jnp.clip(1.0 / (V * d_link**2), 0, 1e6)
                    WX = w[:, None] * X
                    beta_new = jnp.linalg.solve(X.T @ WX + S, X.T @ (w * z))
                    mu_new = dist.inverse_link(X @ beta_new)
                    return mu_new, (beta_new, w)

                mu_hat, (betas, ws) = jax.lax.scan(irls_step, mu0, None, length=25)
                beta, w = betas[-1], ws[-1]
                ll = dist.log_likelihood(y, mu_hat)
                pen = 0.5 * beta @ S @ beta
                _, log_det_H = jnp.linalg.slogdet(X.T @ (w[:, None] * X) + S)
                eigvals = jnp.linalg.eigvalsh(S)
                log_det_S_plus = jnp.sum(
                    jnp.where(eigvals > 1e-10, jnp.log(jnp.maximum(eigvals, 1e-10)), 0.0)
                )
                return -ll + pen + 0.5 * log_det_H - 0.5 * log_det_S_plus

        reml_vg = jax.jit(jax.value_and_grad(reml_score))

        def reml_np(rho_np):
            val, grad = reml_vg(jnp.array(rho_np))
            return float(val), np.array(grad, dtype=np.float64)

        rho0 = np.zeros(len(S_terms))
        result = scipy_minimize(reml_np, rho0, method='L-BFGS-B', jac=True)
        return jnp.exp(jnp.array(result.x))

    @nw.narwhalify
    def fit(self, X: Frame, y: Frame, lam: float = None) -> "GAM":
        """Fit a generalized additive model"""
        _, y_jnp = self._fit_init(X, y)
        design = self._build_design_matrix()
        S_terms = self._build_term_penalties()

        if lam is None:
            self.lam_ = self._optimize_reml(design, y_jnp, S_terms)
        else:
            self.lam_ = jnp.full(len(S_terms), float(lam))

        S = self._build_penalty(self.lam_, S_terms)
        self.coef_ = self._irls(design, y_jnp, S)
        return self

    @nw.narwhalify
    def predict(self, X: Frame) -> jax.Array:
        """Predict on new data using fitted coefficients."""
        for t in self.formula.terms:
            t.basis_mat = t.transform(self._to_jnp(X.select(t.col)))
        design = self._build_design_matrix()
        return self.dist.inverse_link(design @ self.coef_)
    
    def partial_effects(self) -> list[tuple]:
        """Per-term partial effects on the linear predictor scale.

        Returns a list of (col_name, x_vals, effect_vals) tuples — one per term.
        x_vals and effect_vals are jax arrays sorted by x.
        """
        offset = 1  # skip intercept
        results = []
        for t in self.formula.terms:
            k = t.basis_mat.shape[1]
            beta_j = self.coef_[offset:offset + k]
            effect = t.basis_mat @ beta_j
            order = jnp.argsort(t.x)
            results.append((t.col, t.x[order], effect[order]))
            offset += k
        return results

    def plot(self):
        """Plot each smooth term's partial effect."""
        from plotnine import ggplot, aes, geom_line, geom_hline, facet_wrap, theme_bw, labs
        import polars as pl

        effects = self.partial_effects()
        df = pl.concat([
            pl.DataFrame({'x': np.array(x).tolist(), 'effect': np.array(effect).tolist(), 'term': f's({col})'})
            for col, x, effect in effects
        ])

        return (
            ggplot(df, aes('x', 'effect'))
            + geom_line()
            + geom_hline(yintercept=0, linetype='dashed', color='gray', size=0.5)
            + facet_wrap('~term', scales='free')
            + labs(x=None, y='partial effect')
            + theme_bw()
        )
