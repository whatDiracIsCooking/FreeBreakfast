"""Real solid harmonic decomposition of a degree-L Cartesian monomial space.

The construction (``src/basis/harmonics.cppm`` reaches the same space by a
different route, the closed-form Schlegel-Frisch expansion): a real solid
harmonic of degree ell is a degree-ell homogeneous polynomial annihilated by
the Laplacian, so ``graded_transform(L)`` finds it as the subspace of the
degree-L monomial space orthogonal, under the Gaussian-moment inner product
``<x^a1 y^b1 z^c1, x^a2 y^b2 z^c2> = (a1+a2-1)!!(b1+b2-1)!!(c1+c2-1)!!``, to
the image of "multiply a degree-(L-2) polynomial by r^2". Recursing down to
L=0,1 and running ordinary Gram-Schmidt gives every graded block L, L-2, ...
with no Clebsch-Gordan/Gaunt coefficients needed.

This is not capped at L=4: nothing in the construction assumes a maximum
combined angular momentum (the ``src/basis`` production path caps there only
because its own shells never exceed L=2, so a shell pair never combines past
L=4). ``graded_transform`` is memoized per combined_l, built lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache

import numpy as np

from .angular import cartesian_powers, double_factorial_odd, ncart


@dataclass(frozen=True)
class HarmonicBlock:
    """Columns [col_offset, col_offset + (2*ell+1)) of a GradedTransform
    realize pure angular momentum ell (embedded via r^(combined_l - ell) if
    ell < combined_l)."""

    ell: int
    col_offset: int


@dataclass(frozen=True)
class GradedTransform:
    """The change of basis between raw Cartesian monomial coefficients of
    degree combined_l and graded (harmonic_L, r^2 * harmonic_{L-2}, ...)
    coordinates. ``to_monomial`` is dim x dim (dim = ncart(combined_l));
    column j is that graded basis vector's coefficients over
    ``cartesian_powers(combined_l)``. ``blocks`` partitions the columns by
    which target angular momentum they realize, highest ell first."""

    combined_l: int
    dim: int
    to_monomial: np.ndarray
    blocks: list[HarmonicBlock]

    def block_columns(self, ell: int) -> np.ndarray:
        """The (dim, 2*ell+1) slice of to_monomial realizing target momentum
        ell."""
        for block in self.blocks:
            if block.ell == ell:
                lo = block.col_offset
                return self.to_monomial[:, lo : lo + 2 * ell + 1]
        raise ValueError(f"combined_l={self.combined_l} has no ell={ell} block")


def _monomial_gram(p: tuple[int, int, int], q: tuple[int, int, int]) -> float:
    sx, sy, sz = p[0] + q[0], p[1] + q[1], p[2] + q[2]
    if sx % 2 or sy % 2 or sz % 2:
        return 0.0
    return (
        double_factorial_odd(sx - 1)
        * double_factorial_odd(sy - 1)
        * double_factorial_odd(sz - 1)
    )


def _gram_matrix(ell: int) -> np.ndarray:
    powers = cartesian_powers(ell)
    n = len(powers)
    g = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            g[i, j] = _monomial_gram(powers[i], powers[j])
    return g


def _r_squared_map(ell: int) -> np.ndarray:
    """dim(ell) x dim(ell-2): column k has three unit entries, at the
    degree-ell monomials reached by adding 2 to one axis of the k-th
    degree-(ell-2) monomial (always three distinct targets)."""
    hi = cartesian_powers(ell)
    lo = cartesian_powers(ell - 2)
    index_of = {p: i for i, p in enumerate(hi)}
    r = np.zeros((len(hi), len(lo)))
    for k, base in enumerate(lo):
        for axis in range(3):
            target = list(base)
            target[axis] += 2
            r[index_of[tuple(target)], k] += 1.0
    return r


def _orthonormalize(raw: np.ndarray, g: np.ndarray, zero_tol: float) -> np.ndarray:
    """Pivoted Gram-Schmidt under <u,v> = u^T g v: repeatedly take the
    remaining column of largest residual G-norm, normalize it, and project it
    out of the rest -- atomic_cholesky's own pivot-and-project structure,
    applied here to build an orthonormal basis instead of a Cholesky factor.
    Stops early (fewer than `raw.shape[1]` columns returned) once every
    residual norm is below zero_tol."""
    dim, count = raw.shape
    remaining = raw.copy()
    used = [False] * count
    accepted: list[np.ndarray] = []
    for _ in range(count):
        best, best_norm2 = -1, zero_tol
        for j in range(count):
            if used[j]:
                continue
            norm2 = remaining[:, j] @ g @ remaining[:, j]
            if norm2 > best_norm2:
                best, best_norm2 = j, norm2
        if best < 0:
            break
        used[best] = True
        unit = remaining[:, best] / np.sqrt(best_norm2)
        for j in range(count):
            if used[j]:
                continue
            coeff = remaining[:, j] @ g @ unit
            remaining[:, j] -= coeff * unit
        accepted.append(unit)
    return np.stack(accepted, axis=1) if accepted else np.zeros((dim, 0))


@cache
def graded_transform(combined_l: int) -> GradedTransform:
    """The graded transform for combined angular momentum combined_l >= 0,
    built once and cached (see module docstring: no upper cap)."""
    dim = ncart(combined_l)
    if combined_l <= 1:
        # No degree-(L-2) space exists: the raw monomials ARE already the
        # harmonic basis.
        eye = np.eye(dim)
        return GradedTransform(combined_l, dim, eye, [HarmonicBlock(combined_l, 0)])

    lower = graded_transform(combined_l - 2)
    g = _gram_matrix(combined_l)
    r_map = _r_squared_map(combined_l)
    dim_lo = lower.dim

    # Reducible columns in THIS degree's graded coordinates: r^2 applied to
    # each of lower's own graded basis vectors, preserving lower's own graded
    # basis and block layout one level up.
    reducible_embedded = r_map @ lower.to_monomial
    reducible_orthonormal = _orthonormalize(reducible_embedded, g, 1e-12)
    if reducible_orthonormal.shape[1] != dim_lo:
        raise RuntimeError("reducible_embedded columns must be linearly independent")

    # Project the reducible subspace out of the canonical monomial basis, then
    # Gram-Schmidt-orthogonalize the residuals to extract the
    # (2*combined_l+1)-dimensional harmonic complement.
    residual = np.eye(dim)
    for k in range(dim_lo):
        uk = reducible_orthonormal[:, k]
        for i in range(dim):
            coeff = residual[:, i] @ g @ uk
            residual[:, i] -= coeff * uk

    harmonic = _orthonormalize(residual, g, 1e-9)
    n_harmonic = 2 * combined_l + 1
    if harmonic.shape[1] != n_harmonic:
        got = harmonic.shape[1]
        raise RuntimeError(
            f"expected {n_harmonic} harmonic columns at L={combined_l}, got {got}"
        )

    to_monomial = np.zeros((dim, dim))
    to_monomial[:, :n_harmonic] = harmonic
    to_monomial[:, n_harmonic:] = reducible_embedded
    blocks = [HarmonicBlock(combined_l, 0)]
    blocks += [HarmonicBlock(b.ell, b.col_offset + n_harmonic) for b in lower.blocks]
    return GradedTransform(combined_l, dim, to_monomial, blocks)
