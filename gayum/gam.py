from narwhals.typing import Frame
import jax.numpy as jnp
import narwhals as nw
import jax

from .dists import Dist
from .formula import Formula


__all__ = ['GAM']


class GAM:
    __slots__ = ['dist', 'formula']

    def __init__(self, dist: Dist, formula: Formula):
        assert isinstance(dist, Dist)
        assert isinstance(formula, Formula)

        self.dist: Dist = dist
        self.formula: Formula = formula
    
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

        self._features = X.columns
        self._target = y.columns
        return self._to_jnp(X, y)
    
    @nw.narwhalify
    def fit(self, X: Frame, y: Frame) -> "GAM":
        """fit a generalized additive model"""
        X, y = self._fit_init(X, y)
        return self
    