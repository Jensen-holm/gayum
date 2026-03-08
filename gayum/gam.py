from narwhals.typing import Frame
from typing import Optional
import narwhals as nw
import jax.numpy as jnp
import jax

from .dists import Dist
from .terms import Term
from .exceptions import DataTypeError


__all__ = ['GAM']


class GAM:
    __slots__ = ['dist', 'terms', '_features', '_target']

    def __init__(self, dist: Dist, terms: list[Term]):
        assert isinstance(dist, Dist)
        assert isinstance(terms, list)
        assert all([isinstance(t, Term) for t in terms])

        self.dist: Dist = dist
        self.terms: list[Term] = terms
        self._features: list[str] = []
        self._target: Optional[str] = None
    
    def _to_jnp(self, *args) -> jax.Array | tuple[jax.Array, ...]:
        """convert dataframes from whatever backend into jax arrays"""

        def __convert(x) -> jax.Array:
            try:
                nw_df = nw.from_native(x, eager_only=True)
                return jnp.array(nw_df.to_numpy())
            except TypeError:
                raise DataTypeError(invalid_type=str(type(x)))

        converted = [__convert(x) for x in args]
        
        if len(converted) == 1:
            return converted[0]
        return tuple(converted)
    
    @nw.narwhalify
    def _fit_init(self, X: Frame, y: Frame) -> tuple[jax.Array, jax.Array]:
        assert len(y.columns) == 1, f'too many columns in y dataframe |{len(y.columns)}| max is 1'

        self._features = X.columns
        self._target = y.columns
        return self._to_jnp(X, y)
    
    @nw.narwhalify
    def fit(self, X: Frame, y: Frame) -> "GAM":
        """fit a generalized additive model"""
        X, y = self._fit_init(X, y)
        return self
    