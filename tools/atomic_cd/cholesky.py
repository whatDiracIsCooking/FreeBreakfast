"""aCD: pivoted (incomplete) Cholesky decomposition of one atom's raw
two-electron tensor (Beebe & Linderberg; the atomic form is
Aquilante/Pedersen/Lindh's aCD). Cauchy-Schwarz gives the accuracy bound: once
every diagonal residual is below `threshold`, every off-diagonal residual is
bounded by `threshold` too. AO pairs are dense (redundant, not triangular):
all nao*nao ordered pairs, p*nao+q.

This operates on the WHOLE dense ``(nao,nao,nao,nao)`` tensor
``eri_quartet.eri_ao`` already built, rather than gathering one pivot column
at a time: a GPU implementation would gather to avoid materializing the
tensor, but at atomic problem sizes numpy can just slice it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class AtomicCholeskyResult:
    nao: int
    threshold: float
    # each length nao*nao, AO-pair-major p*nao+q
    vectors: list[np.ndarray] = field(default_factory=list)


def atomic_cholesky(
    eri: np.ndarray, threshold: float, max_vectors: int | None = None
) -> AtomicCholeskyResult:
    """Pivoted (incomplete) Cholesky decomposition of `eri` (nao,nao,nao,nao),
    flattened to (nao*nao, nao*nao) over ALL ordered AO pairs. Each iteration:
    pick the AO pair with the largest remaining diagonal residual as the
    pivot, take that column, project out every previously accepted vector,
    normalize by the residual diagonal, update every remaining residual.
    Stops once every remaining diagonal residual is below `threshold`.

    `max_vectors` caps the vector count (default: nao*nao, the dimension of
    the redundant AO-pair space a full-rank decomposition can never exceed) --
    a safety net against numerical noise stalling convergence just above
    `threshold`.
    """
    nao = eri.shape[0]
    n = nao * nao
    m = eri.reshape(n, n)
    d = np.diagonal(m).copy()
    cap = max_vectors if max_vectors else n

    vectors: list[np.ndarray] = []
    for _ in range(cap):
        pivot = int(np.argmax(d))
        if d[pivot] < threshold:
            break
        col = m[:, pivot].copy()
        for prev in vectors:
            coeff = prev[pivot]
            if coeff != 0.0:
                col -= coeff * prev
        col *= 1.0 / np.sqrt(d[pivot])
        d -= col * col
        vectors.append(col)
    return AtomicCholeskyResult(nao, threshold, vectors)


def reconstruct(result: AtomicCholeskyResult) -> np.ndarray:
    """sum_J L_J[pq] L_J[rs], as a flat (nao*nao, nao*nao) matrix -- directly
    comparable to ``eri.reshape(nao*nao, nao*nao)``."""
    n = result.nao * result.nao
    out = np.zeros((n, n))
    for v in result.vectors:
        out += np.outer(v, v)
    return out


def reconstruction_error(eri: np.ndarray, result: AtomicCholeskyResult) -> float:
    """max |reconstruct(result) - eri|, flattened the same way -- should stay
    below `result.threshold` everywhere (the Cauchy-Schwarz bound above)."""
    n = result.nao * result.nao
    return float(np.abs(reconstruct(result) - eri.reshape(n, n)).max())
