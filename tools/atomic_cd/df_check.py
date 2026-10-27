"""How good is an auxiliary basis? Two density-fitting error measures, both
over PySCF and both spherical (cart=False, what every basis here means):

- ``one_centre_error``: max |(ij|kl) - (ij|kl)_DF| over one atom's own
  orbital basis, Coulomb-metric fitting (ij|kl)_DF = sum_PQ (ij|P)
  [(P|Q)^-1]_PQ (Q|kl) -- the quantity acCD's candidate pool targets.
- ``rhf_energy_error``: E(DF-RHF) - E(RHF) for a small molecule, the aux
  basis given per element. This also exposes what a one-centre pool cannot
  see: two-centre orbital products, which no candidate was built from.

The aux basis is anything PySCF's ``auxbasis`` accepts: a name such as
``"cc-pvdz-jkfit"``, a shell list (``compact.to_pyscf_basis``), or a
per-element dict of either.
"""

from __future__ import annotations

import numpy as np
from pyscf import df, scf

from . import cholesky, eri_quartet

# Angstrom; CO at its experimental r_e, CH4 tetrahedral at r(CH) = 1.087
_CH = 1.087 / np.sqrt(3.0)
MOLECULES: dict[str, list] = {
    "CO": [["C", (0.0, 0.0, 0.0)], ["O", (0.0, 0.0, 1.128)]],
    "CH4": [
        ["C", (0.0, 0.0, 0.0)],
        ["H", (_CH, _CH, _CH)],
        ["H", (-_CH, -_CH, _CH)],
        ["H", (-_CH, _CH, -_CH)],
        ["H", (_CH, -_CH, -_CH)],
    ],
}

# metric eigenvalues below this (relative to the largest) are dropped
_METRIC_RCOND = 1e-14


def aux_function_count(element: str, auxbasis) -> int:
    """Spherical auxiliary functions `auxbasis` puts on one `element` atom."""
    mol = eri_quartet.build_mol([(element, (0.0, 0.0, 0.0))], "sto-3g")
    return df.make_auxmol(mol, auxbasis).nao


def one_centre_error(element: str, basis, auxbasis) -> float:
    """max |(ij|kl) - (ij|kl)_DF| over one `element` atom in `basis`. The
    metric is inverted through its eigendecomposition with relative cutoff
    ``_METRIC_RCOND``, so a near-singular aux basis degrades instead of
    failing."""
    mol = eri_quartet.build_mol([(element, (0.0, 0.0, 0.0))], basis)
    auxmol = df.make_auxmol(mol, auxbasis)
    # unique pairs only: (ji|..) repeats (ij|..), and (ij|kl) = (kl|ij), so
    # the eight-fold packed lower triangle holds every distinct value
    npair = mol.nao * (mol.nao + 1) // 2
    eri = eri_quartet.PackedSymmetric(mol.intor("int2e", aosym="s8"), npair)
    three = df.incore.aux_e2(mol, auxmol, "int3c2e", aosym="s2ij")
    w, v = np.linalg.eigh(auxmol.intor("int2c2e"))
    keep = w > _METRIC_RCOND * w.max()
    b = three @ (v[:, keep] / np.sqrt(w[keep]))
    # (ij|kl)_DF = B B^T, the same low-rank shape a Cholesky check compares
    return cholesky.reconstruction_error(eri, b.T)


def rhf_energy_error(atoms: list, basis, auxbasis) -> tuple[float, float]:
    """(E(RHF), E(DF-RHF) - E(RHF)) in hartree for `atoms` (Angstrom) in
    `basis`, J and K both fit over `auxbasis`. DF-RHF starts from the
    conventional density, so both land in the same SCF solution."""
    mol = eri_quartet.build_mol(atoms, basis, unit="Angstrom")
    conventional = scf.RHF(mol)
    conventional.conv_tol = 1e-10
    e_exact = conventional.kernel()
    fitted = scf.RHF(mol).density_fit(auxbasis=auxbasis)
    fitted.conv_tol = 1e-10
    e_fitted = fitted.kernel(conventional.make_rdm1())
    if not (conventional.converged and fitted.converged):
        raise RuntimeError("RHF or DF-RHF did not converge")
    return e_exact, e_fitted - e_exact
