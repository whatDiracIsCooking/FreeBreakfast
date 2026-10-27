"""Pivoted (incomplete) Cholesky decomposition of a positive semidefinite
matrix (Beebe & Linderberg), shared by both stages here: aCD runs it on one
atom's two-electron integrals over its uncontracted primitives, packed by full
eight-fold symmetry (``eri_quartet.screen_pairs``, an
``eri_quartet.PackedSymmetric``; the atomic form is Aquilante/Pedersen/
Lindh's aCD), acCD on each angular momentum's dense unit-diagonal candidate
metric (``compact.coulomb_metric``). Either is read only through
``m.diagonal()``, ``m[p]`` (one row) and, for the check, lower-triangle row
blocks. Cauchy-Schwarz gives the accuracy bound: once
every diagonal residual is below `threshold`, every off-diagonal residual is
bounded by `threshold` too.

The matrix is held whole (packed, for aCD) rather than computed one pivot
row at a time: a GPU implementation would compute each row on demand to
avoid materializing it, but at atomic problem sizes numpy can just slice
it.
"""

from __future__ import annotations

import math

import numpy as np

# rows per block of reconstruction_error's V^T V, bounding its temporaries
_CHECK_BLOCK = 1024


def pivoted_cholesky(
    m, threshold: float, max_vectors: int | None = None
) -> tuple[list[int], np.ndarray]:
    """Decompose the symmetric PSD `m` (n, n) -- a dense array or an
    ``eri_quartet.PackedSymmetric`` -- as V^T V. Each iteration picks
    the largest remaining diagonal residual as the pivot, takes that row of
    `m`, projects out every accepted vector in one product, and normalizes
    by the residual. Stops once every residual is below `threshold`.

    Returns (pivots, vectors): `vectors` is (len(pivots), n), row J being
    Cholesky vector J -- (phi_k|psi_J) at each k, NOT expansion
    coefficients. The pivot's own residual is set to exactly zero after its
    update (it is zero in exact arithmetic), so roundoff can never select
    it twice. `max_vectors` caps the count (default n, the rank bound) -- a
    safety net against noise stalling convergence just above `threshold`."""
    n = m.shape[0]
    cap = min(max_vectors or n, n)
    d = m.diagonal().copy()
    vectors = np.empty((min(cap, 64), n))
    pivots: list[int] = []
    for j in range(cap):
        pivot = int(np.argmax(d))
        if d[pivot] < threshold:
            break
        if j == vectors.shape[0]:
            vectors = np.concatenate([vectors, np.empty_like(vectors)])[:cap]
        col = m[pivot] - vectors[:j, pivot] @ vectors[:j]
        col *= 1.0 / math.sqrt(d[pivot])
        d -= col * col
        d[pivot] = 0.0
        vectors[j] = col
        pivots.append(pivot)
    return pivots, vectors[: len(pivots)]


def reconstruction_error(m, vectors: np.ndarray) -> float:
    """max |m - V^T V| for symmetric `m` (dense or ``PackedSymmetric``) --
    should stay below the decomposition's threshold everywhere (the
    Cauchy-Schwarz bound above). Symmetry makes the lower triangle the whole
    answer, taken in row blocks so the check costs no more memory than `m`
    itself."""
    n = m.shape[0]
    worst = 0.0
    for r0 in range(0, n, _CHECK_BLOCK):
        r1 = min(r0 + _CHECK_BLOCK, n)
        exact = m[r0:r1, :r1] if isinstance(m, np.ndarray) else m.lower_rows(r0, r1)
        diff = exact - vectors[:, r0:r1].T @ vectors[:, :r1]
        lower = np.arange(r1)[None, :] <= np.arange(r0, r1)[:, None]
        worst = max(worst, float(np.abs(diff[lower]).max()))
    return worst
