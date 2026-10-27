"""SVD contraction of a primitive auxiliary basis: Lehtola, J. Chem. Theory
Comput. 19, 6242 (2023), arXiv:2306.11039, after Kallay's natural auxiliary
functions. acCD's pivots fit every candidate product to the threshold, but
most of that span is never needed by the atom's ACTUAL orbital products;
this keeps only the combinations those products use.

THE MATRIX. For one atom in its own (contracted, spherical) orbital basis,
take the three-index integrals I[mu nu, P] = (mu nu|P) over every AO pair
and the primitive aux functions P, Coulomb-orthonormalize the aux side,
J = I V^-1/2 with V = (P|Q), and form W = J^T J. Dropping a direction of W
with eigenvalue lambda changes the fitted (mu nu|kappa lambda) by at most
lambda, so keeping lambda >= `threshold` reproduces the three-index
integrals to that precision. Here it is the singular values s of J that
are computed (lambda = s^2), never W itself -- forming J^T J would square
J's condition number for nothing.

THE BLOCKS. On one centre, V and W couple only aux functions of the same
(l, m), and every m of one l gives the same block (the sum over all AO pairs
is rotationally invariant), so one component per shell -- the first -- is
the whole problem for that l. The kept eigenvectors map back to primitive
coefficients as C = V_l^-1/2 U: one contracted function per kept
eigenvector, all over the same primitives, so each l becomes ONE general
contraction (the shape cc-pVXZ shells already have). C is over normalized
primitives -- what PySCF and Basis Set Exchange coefficients both mean --
so the paper's sqrt(alpha) Coulomb-to-overlap conversion is not needed; an
overall scale per function is irrelevant to fitting, and readers
renormalize each contraction anyway.
"""

from __future__ import annotations

import numpy as np
from pyscf import df

from . import eri_quartet
from .compact import AuxShell, to_pyscf_basis

# V_l eigenvalues below this (relative to the largest) are dropped from V^-1/2
_METRIC_RCOND = 1e-14


def _inverse_sqrt(v: np.ndarray) -> np.ndarray:
    w, u = np.linalg.eigh(v)
    keep = w > _METRIC_RCOND * w.max()
    return (u[:, keep] / np.sqrt(w[keep])) @ u[:, keep].T


def contract(
    element: str, basis, aux_shells: dict[int, list[AuxShell]], threshold: float
) -> dict[int, list[AuxShell]]:
    """`aux_shells` -- uncontracted, as ``compact.to_aux_shells`` makes them
    -- contracted against `element`'s own orbital `basis` (any form
    ``mol.basis`` takes): per l, one general contraction over that l's
    primitives, one function per singular value s of J with s^2 >=
    `threshold`, largest first. An l whose every s^2 falls below
    `threshold` is dropped entirely (module docstring)."""
    mol = eri_quartet.build_mol([(element, (0.0, 0.0, 0.0))], basis)
    auxmol = df.make_auxmol(mol, to_pyscf_basis(aux_shells))
    if any(auxmol.bas_nprim(ib) != 1 for ib in range(auxmol.nbas)):
        raise ValueError("contract() takes uncontracted aux shells only")
    # unique pairs (ij|P), i >= j; W sums over ALL pairs, so weight ij and ji
    three = df.incore.aux_e2(mol, auxmol, "int3c2e", aosym="s2ij")
    i, j = np.tril_indices(mol.nao)
    three[i != j] *= np.sqrt(2.0)
    metric = auxmol.intor("int2c2e")

    contracted: dict[int, list[AuxShell]] = {}
    for ell in sorted(aux_shells):
        shells = [ib for ib in range(auxmol.nbas) if auxmol.bas_angular(ib) == ell]
        first = auxmol.ao_loc[shells]
        exponents = tuple(float(auxmol.bas_exp(ib)[0]) for ib in shells)
        v_isqrt = _inverse_sqrt(metric[np.ix_(first, first)])
        _u, s, vt = np.linalg.svd(three[:, first] @ v_isqrt, full_matrices=False)
        keep = s * s >= threshold
        if not keep.any():
            continue
        coefficients = (v_isqrt @ vt[keep].T).T
        contracted[ell] = [
            AuxShell(ell, exponents, tuple(tuple(map(float, c)) for c in coefficients))
        ]
    return contracted
