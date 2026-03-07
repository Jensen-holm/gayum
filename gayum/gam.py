import narwhals as nw
import jax.numpy as jnp

from .dists import Dist
from .terms import Term
from .exceptions import XTrainTypeError


__all__ = ['GAM']


class GAM:
    __slots__ = ['dist', 'terms']

    def __init__(self, dist: Dist, terms: list[Term]) -> None:
        assert isinstance(dist, Dist)
        assert isinstance(terms, list)
        assert all([isinstance(t, Term) for t in terms])

        self.dist = dist
        self.terms = terms
    
    @staticmethod
    def __df_to_jnp(df) -> jnp.ndarray:
        """convert narwhals dataframe into jnp.ndarray"""
        df = nw.from_native(df, eager_only=True)
        return jnp.array(df.to_numpy())
    
    def __to_jnp(self, *args) -> tuple[jnp.ndarray]:
        """convert data from whatever data type it is into jnp.ndarray"""

        def __convert(x) -> jnp.ndarray:
            if isinstance(x, jnp.ndarray):
                return x
            if nw.is_native_dataframe(x):
                return self.__df_to_jnp(x)
            raise XTrainTypeError(invalid_type=str(type(x)))
        
        return tuple([__convert(x) for x in args])

    def fit(self, X, y) -> "GAM":
        """fit a generalized additive model"""
        X, y = self.__to_jnp(X, y)
        return self
    