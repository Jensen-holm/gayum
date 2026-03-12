from abc import ABC, abstractmethod
from typing import Optional
import jax.numpy as jnp
import jax


class Term(ABC):
    @abstractmethod
    def build(self, x: jax.Array) -> "Term":
        pass

    def __add__(self, other: "Term") -> list["Term"]:
        return [self, other]

class s(Term):
    __slots__ = ['col', 'n_splines', 'degree', 'basis_mat']

    def __init__(self, col: str, degree: int = 3, n_splines: int = 10):
        self.col: str = col
        self.degree: int = degree
        self.n_splines: int = n_splines
        self.basis_mat: Optional[jax.Array] = None
    
    def _compute_basis(self, x: float, knots: jax.Array) -> jax.Array:
        n_basis = len(knots) - self.degree - 1
        
        basis = jnp.array([
            jnp.where((knots[i] <= x) & (x < knots[i + 1]), 1.0, 0.0)
            for i in range(len(knots) - 1)
        ])
        
        for d in range(1, self.degree + 1):
            n = len(knots) - d - 1
            lw = (x - knots[:n]) / (knots[d:d+n] - knots[:n] + 1e-10)
            rw = (knots[d+1:d+1+n] - x) / (knots[d+1:d+1+n] - knots[1:n+1] + 1e-10)
            basis = lw * basis[:n] + rw * basis[1:n+1]
        
        return basis[:n_basis]

    def build(self, x: jax.Array) -> "Term":
        knots = jnp.concatenate([
            jnp.repeat(jnp.array([x.min()]), self.degree),
            jnp.linspace(x.min(), x.max(), self.n_splines),
            jnp.repeat(jnp.array([x.max()]), self.degree),
        ])
        self.basis_mat = jax.vmap(lambda xi: self._compute_basis(xi, knots))(x)
        return self
