import pytest
import gayum
import jax



def test_gam_class_init():
    with pytest.raises(TypeError):
        gayum.GAM()
    
    _ = gayum.GAM(dist=gayum.dists.Normal(), terms=[gayum.terms.s()])


def test_df_to_jnp(generic_GAM, random_pd_df, random_pl_df):
    result1, result2 = generic_GAM._to_jnp(random_pd_df, random_pl_df)
    assert isinstance(result1, jax.Array)
    assert isinstance(result2, jax.Array)
