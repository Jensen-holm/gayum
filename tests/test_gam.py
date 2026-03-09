import pytest
import jax

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
