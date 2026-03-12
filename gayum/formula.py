import narwhals as nw
from narwhals.typing import Frame

from .terms import Term


class Formula:
    __slots__ = ['response', 'features', 'terms']

    @nw.narwhalify
    def __init__(self, df: Frame, response: str, terms: list[Term]):
        assert response in df.columns
        assert all([x.col in df.columns for x in terms])

        self.terms = terms
        self.response = response
        self.features: list[str] = [t.col for t in terms]
