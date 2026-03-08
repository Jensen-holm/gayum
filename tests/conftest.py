import pytest
import gayum
import pandas as pd
import polars as pl
import random

@pytest.fixture
def generic_GAM() -> gayum.GAM:
    return gayum.GAM(dist=gayum.dists.Normal(), terms=[gayum.terms.s()])


@pytest.fixture
def random_pd_df():
    return pd.DataFrame({'col': [random.random() * 1000 for _ in range(100)]})


@pytest.fixture
def random_pl_df():
    return pl.DataFrame({'col': [random.random() * 1000 for _ in range(100)]})

