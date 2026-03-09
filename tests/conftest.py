import pytest
import pandas as pd
import polars as pl
from plotnine.data import mtcars

import gayum
from gayum.terms import s


@pytest.fixture
def mtcars_pd() -> pd.DataFrame:
    return mtcars

@pytest.fixture
def mtcars_pl() -> pl.DataFrame:
    return pl.from_pandas(mtcars)

@pytest.fixture
def mtcarsGAM() -> gayum.GAM:
    f = gayum.Formula(mtcars, 'mpg', s('hp') + s('wt'))
    return gayum.GAM(dist=gayum.dists.Normal(), formula=f)
