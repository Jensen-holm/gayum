from abc import ABC, abstractmethod
import jax.numpy as jnp
import jax


class Term(ABC):
    @abstractmethod
    def build(self, x: jax.Array) -> "Term":
        pass

    def __add__(self, other: "Term") -> list["Term"]:
        return [self, other]

class s(Term):
    __slots__ = ['col', 'n_splines']

    def __init__(self, col: str, degree: int = 3, n_splines: int = 10):
        self.col = col
        self.n_splines = n_splines
    
    def _cox_de_boor_recursion(self, idx: int, xi: int):
        if not idx:
            return jnp.where()
    
    def build(self, x: jax.Array) -> jax.Array:
        knots = jnp.concatenate(
            jnp.repeat(x.min(), self.degree),
            jnp.linspace(x.min(), x.max(), self.n_splines),
            jnp.repeat(x.max(), self.degree),
        )
