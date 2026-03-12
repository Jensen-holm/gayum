def test_s_term_build(mtcars_pl, mtcarsGAM):
    for t in mtcarsGAM.formula.terms:
        t = t.build(x=mtcarsGAM._to_jnp(mtcars_pl.select(t.col)))
        assert t.basis_mat is not None
