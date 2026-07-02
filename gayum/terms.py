from abc import ABC, abstractmethod
from typing import Optional
import jax.numpy as jnp
import numpy as np
import jax


class Term(ABC):
    @abstractmethod
    def build(self, x: jax.Array) -> "Term":
        pass

    @abstractmethod
    def penalty(self) -> jax.Array:
        pass

    def __add__(self, other: "Term") -> list["Term"]:
        return [self, other]


class ps(Term):
    """P-spline: B-spline basis with second-difference penalty."""
    __slots__ = ['col', 'n_splines', 'degree', 'basis_mat', 'knots', 'x', '_basis_mean']

    def __init__(self, col: str, degree: int = 3, n_splines: int = 10):
        self.col: str = col
        self.degree: int = degree
        self.n_splines: int = n_splines
        self.basis_mat: Optional[jax.Array] = None
        self.knots: Optional[jax.Array] = None
        self.x: Optional[jax.Array] = None
        self._basis_mean: Optional[jax.Array] = None

    def _compute_basis(self, x: float, knots: jax.Array) -> jax.Array:
        n_basis = len(knots) - self.degree - 1
        n_intervals = len(knots) - 1

        # Use <= for the last interval so x = x.max() evaluates correctly.
        basis = jnp.array([
            jnp.where(
                (knots[i] <= x) & (x <= knots[i + 1]) if i == n_intervals - 1
                else (knots[i] <= x) & (x < knots[i + 1]),
                1.0, 0.0
            )
            for i in range(n_intervals)
        ])

        for d in range(1, self.degree + 1):
            n = len(knots) - d - 1
            lw = (x - knots[:n]) / (knots[d:d+n] - knots[:n] + 1e-10)
            rw = (knots[d+1:d+1+n] - x) / (knots[d+1:d+1+n] - knots[1:n+1] + 1e-10)
            basis = lw * basis[:n] + rw * basis[1:n+1]

        return basis[:n_basis]

    def build(self, x: jax.Array) -> "ps":
        self.x = x
        self.knots = jnp.concatenate([
            jnp.repeat(jnp.array([x.min()]), self.degree),
            jnp.linspace(x.min(), x.max(), self.n_splines),
            jnp.repeat(jnp.array([x.max()]), self.degree),
        ])
        raw = jax.vmap(lambda xi: self._compute_basis(xi, self.knots))(x)
        self._basis_mean = raw.mean(axis=0)
        self.basis_mat = raw - self._basis_mean
        return self

    def transform(self, x: jax.Array) -> jax.Array:
        raw = jax.vmap(lambda xi: self._compute_basis(xi, self.knots))(x)
        return raw - self._basis_mean

    def penalty(self) -> jax.Array:
        k = self.basis_mat.shape[1]
        D = jnp.diff(jnp.eye(k), n=2, axis=0)
        return D.T @ D


class tp(Term):
    """Thin plate regression spline (mgcv's default s() basis)."""
    __slots__ = ['col', 'k', 'basis_mat', 'x', '_knots', '_Q2', '_U', '_d', '_Z', '_x_min', '_x_range']

    def __init__(self, col: str, k: int = 10):
        self.col: str = col
        self.k: int = k
        self.basis_mat: Optional[jax.Array] = None
        self.x: Optional[jax.Array] = None
        self._knots: Optional[jax.Array] = None
        self._Q2: Optional[jax.Array] = None
        self._U: Optional[jax.Array] = None
        self._d: Optional[jax.Array] = None
        self._Z: Optional[jax.Array] = None
        self._x_min: Optional[float] = None
        self._x_range: Optional[float] = None

    def _eval_basis(self, x: float, knots: jax.Array, Q2: jax.Array, U: jax.Array, d: jax.Array) -> jax.Array:
        """Raw basis at scalar x (in [0,1] scaled space): [1, x, smooth_1, ..., smooth_{K-2}]."""
        t = jnp.array([1.0, x])
        e = jnp.abs(x - knots) ** 3          # TPS kernel, shape (K,)
        e_proj = Q2.T @ e                     # project out null space, shape (K-2,)
        b_smooth = (e_proj @ U) * (d ** -0.5) # normalize, shape (K-2,)
        return jnp.concatenate([t, b_smooth])

    def build(self, x: jax.Array) -> "tp":
        self.x = x  # kept in original scale for partial_effects/plot

        # Scale x to [0, 1] — mgcv normalises to unit range internally.
        # This makes kernel values O(1) so basis functions and coefficients
        # end up on the same scale as mgcv's output.
        x_min = float(x.min())
        x_range = float(x.max() - x.min())
        self._x_min = x_min
        self._x_range = x_range
        x_sc = (x - x_min) / x_range

        # Use all unique scaled values as knots — full TPRS construction.
        # Truncation to k basis functions happens in eigenspace, not by knot subsetting.
        unique_xs = np.unique(np.asarray(x_sc))
        K = len(unique_xs)
        knots = jnp.array(unique_xs)
        self._knots = knots

        # Null space at all unique points: T_k is K×2
        T_k = jnp.stack([jnp.ones(K), knots], axis=1)

        # TPS kernel matrix in scaled space: η(xi, xj) = |xi - xj|^3
        E_k = jax.vmap(lambda xi: jnp.abs(xi - knots) ** 3)(knots)

        # Q2: K×(K-2) basis for the null space complement of T_k
        U_T, _, _ = jnp.linalg.svd(T_k, full_matrices=True)
        Q2 = U_T[:, 2:]
        self._Q2 = Q2

        # Project E_k and eigendecompose; take the top r = k-2 eigenvectors
        E_proj = Q2.T @ E_k @ Q2             # (K-2, K-2)
        d_all, U_all = jnp.linalg.eigh(E_proj)  # ascending order
        r = min(self.k - 2, K - 2)
        d_r = jnp.maximum(d_all[-r:], 1e-8) # largest r eigenvalues, clipped
        U_r = U_all[:, -r:]
        self._d = d_r
        self._U = U_r

        # Raw basis at all training data: shape (n, 2+r)
        raw = jax.vmap(lambda xi: self._eval_basis(xi, knots, Q2, U_r, d_r))(x_sc)

        # Sum-to-zero reparametrisation matching mgcv's identifiability constraint.
        # C = colMeans(raw); Z spans the right null space of C via QR.
        # The new k-1 basis columns raw @ Z automatically have zero column means.
        C = raw.mean(axis=0)               # (k,) where k = 2 + r
        Q_c, _ = jnp.linalg.qr(C[:, None], mode='complete')
        Z = Q_c[:, 1:]                     # (k, k-1): null space of C
        self._Z = Z

        self.basis_mat = raw @ Z           # (n, k-1)
        return self

    def transform(self, x: jax.Array) -> jax.Array:
        x_sc = (x - self._x_min) / self._x_range
        raw = jax.vmap(lambda xi: self._eval_basis(xi, self._knots, self._Q2, self._U, self._d))(x_sc)
        return raw @ self._Z

    def penalty(self) -> jax.Array:
        """Penalty in the Z-reparametrised basis: Z^T S_old Z.

        S_old = diag(0, 0, 1, ..., 1): zero for the two TPS null-space columns
        ([1] and [x]), identity for the k-2 smooth eigenvector columns.
        After reparametrisation this becomes a (k-1)×(k-1) matrix with rank k-2.
        """
        k_raw = self._Z.shape[0]           # = 2 + r
        S_old = jnp.zeros((k_raw, k_raw))
        S_old = S_old.at[2:, 2:].set(jnp.eye(k_raw - 2))
        return self._Z.T @ S_old @ self._Z


# s() uses thin plate regression splines by default, matching mgcv
s = tp
