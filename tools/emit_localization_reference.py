#!/usr/bin/env python3
"""Write the reference data the Localization step's tests read.

For each case below, PySCF's closed-shell RHF (as emit_scf_reference.py runs
it, over this tree's own basis files) is localized by PySCF's Pipek-Mezey with
Mulliken populations, for both exponents. PySCF's optimizer stops at any
stationary point, and from a symmetric molecule's canonical orbitals that is
often a saddle (water's bonds smeared over both hydrogens), so the kernel is
alternated with PySCF's Jacobi stability check until the check finds no
uphill rotation: a maximum, reached by a code that shares nothing with ours.

Recorded: the functional L_p, and every localized orbital's Mulliken
population on every atom, Q^A_ii. Both are invariant under any transformation
within an atom's AOs (normalisation, order, sign), which differ between the
two codes; the orbitals themselves are not recorded for that reason. The
populations are the convention-free projection of one code's orbitals onto
the other's: a test matches them up to orbital order.

    uv run --group tools tools/emit_localization_reference.py

writes test/gtest/data/localization/pipek_mezey.json. Re-run it to regenerate;
the file is never hand-edited.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import platform

import numpy
import pyscf
from emit_scf_reference import AMMONIUM, BASES, WATER, pyscf_basis
from pyscf import gto, lo, scf

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "test" / "gtest" / "data" / "localization" / "pipek_mezey.json"

CASES = [
    {"name": "h2o_sto3g", "atoms": WATER, "basis": "sto-3g", "charge": 0},
    {"name": "h2o_ccpvdz", "atoms": WATER, "basis": "cc-pvdz", "charge": 0},
    {"name": "nh4_ccpvdz", "atoms": AMMONIUM, "basis": "cc-pvdz", "charge": 1},
]

EXPONENTS = (2, 4)

# Kernel-then-stability rounds before giving up.
MAX_ROUNDS = 20


def populations(mol: gto.Mole, s: numpy.ndarray, c: numpy.ndarray) -> numpy.ndarray:
    """Q^A_ii, orbital by atom: sum over atom A's AOs of C (S C)."""
    w = s @ c
    return numpy.array(
        [
            numpy.einsum("mi,mi->i", c[p0:p1], w[p0:p1])
            for _, _, p0, p1 in mol.offset_nr_by_atom()
        ]
    ).T


def localize(mol: gto.Mole, mf: scf.hf.RHF, exponent: int) -> numpy.ndarray:
    """PySCF's Mulliken Pipek-Mezey of the occupied orbitals, to a maximum."""
    pm = lo.PM(mol, mf.mo_coeff[:, mf.mo_occ > 0], mf)
    pm.pop_method = "mulliken"
    pm.exponent = exponent
    pm.conv_tol = 1e-12
    pm.verbose = 0
    c = pm.kernel()
    for _ in range(MAX_ROUNDS):
        # (orbitals, stable), in whichever order this PySCF returns them.
        status = pm.stability_jacobi(return_status=True)
        uphill = next(x for x in status if getattr(x, "ndim", 0) == 2)
        if next(x for x in status if getattr(x, "ndim", 0) != 2):
            return c
        c = pm.kernel(uphill)
    raise RuntimeError(f"no stable maximum after {MAX_ROUNDS} rounds")


def case_reference(case: dict) -> dict:
    """The localized references of one molecule, for every exponent."""
    symbols = {symbol for symbol, _ in case["atoms"]}
    mol = gto.M(
        atom=[[symbol, coord] for symbol, coord in case["atoms"]],
        unit="Bohr",
        basis=pyscf_basis(BASES[case["basis"]], symbols),
        charge=case["charge"],
        spin=0,
        cart=False,
        verbose=0,
    )
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged, f"{case['name']} did not converge"
    s = mol.intor_symmetric("int1e_ovlp")
    localized = []
    for exponent in EXPONENTS:
        q = populations(mol, s, localize(mol, mf, exponent))
        localized.append(
            {
                "exponent": exponent,
                "functional": float((q**exponent).sum()),
                "populations": [[float(v) for v in row] for row in q],
            }
        )
    return {
        "name": case["name"],
        "basis": case["basis"],
        "charge": case["charge"],
        "atoms": [
            {"symbol": symbol, "z": gto.charge(symbol), "coord": list(coord)}
            for symbol, coord in case["atoms"]
        ],
        "electrons": int(mol.nelectron),
        "localized": localized,
    }


def main() -> None:
    doc = {
        "description": "PySCF closed-shell RHF over this tree's data/basis "
        "files, its occupied orbitals localized by PySCF's Pipek-Mezey "
        "(Mulliken populations, conv_tol 1e-12, kernel alternated with "
        "stability_jacobi until stable): the functional sum_i sum_A "
        "(Q^A_ii)^p, and Q^A_ii per localized orbital (rows) and atom "
        "(columns, in the atoms' order); coordinates in bohr",
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": numpy.__version__,
            "date": datetime.date.today().isoformat(),
            "host": platform.node(),
            "command": "uv run --group tools tools/emit_localization_reference.py",
        },
        "cases": [case_reference(case) for case in CASES],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
