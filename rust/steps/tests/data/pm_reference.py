"""The fixtures of fb-steps' localization tests (`src/steps/localization.rs`,
its `tests` module), first written for the old localization/pipek_mezey
crate's `tests/pipek_mezey.rs`: PySCF's closed-shell RHF of two molecules,
localized by PySCF's Pipek-Mezey with Mulliken populations at both
exponents.

Usage (from the checkout root; PySCF is in the `tools` dependency group):

    OMP_NUM_THREADS=8 uv run --group tools python \\
        rust/steps/tests/data/pm_reference.py

The cases are water (`h2o.xyz`, five occupied orbitals, an odd count, so
the schedule's padding runs) and borane (`bh3.xyz`, four, an even count),
both in cc-pVDZ, read from the BSE JSON document the binary compiles in and
the coordinates converted to bohr by CODATA 2018, through `molecule.py`
beside this script, as every reference of the rebuild is. For each it
writes beside this script, every array float64 in Fortran order:

- `pm_<case>_s.npy`: S, n x n.
- `pm_<case>_c.npy`: the canonical occupied orbitals, n x n_occ, ascending.
- `pm_<case>_functional.npy`: [L_2 of the canonical orbitals, L_2 localized,
  L_4 of the canonical orbitals, L_4 localized].

S and C are PySCF's, in its AO normalization and its order within a shell:
the functional and the populations are invariant under any transformation
within an atom's AOs, so they need no conversion to the tree's, and the
test hands both to the localization as they are. Only the atoms' AO ranges
come from the tree's own basis, and a set's function count per atom is the
same in both codes. The functional is compared, not the orbitals:
Pipek-Mezey over Mulliken populations leaves rotations within one atom's
orbitals (the lone pairs of water) nearly flat.

PySCF's optimizer stops at any stationary point, and from a symmetric
molecule's canonical orbitals that can be a saddle, so the kernel is
alternated with its Jacobi stability check until the check finds no uphill
rotation, as tools/emit/localization_reference.py does for the C++ tree's
tests. It prints, per case, L and the number of rounds.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pyscf
from pyscf import lo, scf

HERE = Path(__file__).resolve().parent
MOLECULE = HERE / "molecule.py"
CASES = ["h2o", "bh3"]
EXPONENTS = (2, 4)
# Kernel-then-stability rounds before giving up.
MAX_ROUNDS = 20


def molecule_reference():
    """`molecule.py`, beside this script, as a module."""
    spec = importlib.util.spec_from_file_location("molecule_reference", MOLECULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def populations(mol, s: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Q^A_ii, orbital by atom: the sum over atom A's AOs of C (S C)."""
    w = s @ c
    return np.array(
        [
            np.einsum("mi,mi->i", c[p0:p1], w[p0:p1])
            for _, _, p0, p1 in mol.offset_nr_by_atom()
        ]
    ).T


def localize(mol, orbocc: np.ndarray, exponent: int) -> tuple[np.ndarray, int]:
    """PySCF's Pipek-Mezey of `orbocc` over Mulliken populations, driven to
    a maximum, and the kernel rounds it took."""
    pm = lo.PM(mol, orbocc)
    pm.pop_method = "mulliken"
    pm.exponent = exponent
    pm.conv_tol = 1e-12
    pm.verbose = 0
    c = pm.kernel()
    for rounds in range(1, MAX_ROUNDS + 1):
        # (orbitals, stable), in whichever order this PySCF returns them.
        status = pm.stability_jacobi(return_status=True)
        uphill = next(x for x in status if getattr(x, "ndim", 0) == 2)
        if next(x for x in status if getattr(x, "ndim", 0) != 2):
            return c, rounds
        c = pm.kernel(uphill)
    raise SystemExit(f"no stable maximum after {MAX_ROUNDS} rounds")


def save(name: str, a: np.ndarray) -> None:
    np.save(HERE / f"pm_{name}.npy", np.asfortranarray(a))


def main() -> None:
    ref = molecule_reference()
    print(f"PySCF {pyscf.__version__}")
    for case in CASES:
        mol, _, _, _ = ref.named(HERE / f"{case}.xyz", "cc-pvdz", "cc-pvdz-jkfit")
        mf = scf.RHF(mol)
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-9
        mf.kernel()
        if not mf.converged:
            raise SystemExit(f"{case}: PySCF did not converge")
        s = mol.intor_symmetric("int1e_ovlp")
        orbocc = mf.mo_coeff[:, mf.mo_occ > 0]
        save(f"{case}_s", s)
        save(f"{case}_c", orbocc)
        functional = []
        q0 = populations(mol, s, orbocc)
        for p in EXPONENTS:
            c, rounds = localize(mol, orbocc, p)
            q = populations(mol, s, c)
            functional += [float((q0**p).sum()), float((q**p).sum())]
            print(f"{case} p={p}: L = {functional[-1]:.15f} in {rounds} round(s)")
        save(f"{case}_functional", np.array(functional))


if __name__ == "__main__":
    main()
