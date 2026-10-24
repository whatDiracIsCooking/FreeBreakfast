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

The same orbitals are also localized into intrinsic bond orbitals: PySCF's IAOs
over its MINAO table (which data/basis/minao.json is, bit for bit),
Loewdin-orthonormalized, and its Pipek-Mezey over their populations
(`lo.ibo.PipekMezey`), driven to a maximum the same way. Recorded: the
functional over the IAO populations, every IBO's Mulliken population on every
atom (as above, the convention-free match), and the IAO partial charges of
eq 3, Z_A - 2 sum_i sum_{r in A} (A^T S C)_ri^2, which check the IAOs alone.

    uv run --group tools tools/emit_localization_reference.py

writes test/gtest/data/localization/pipek_mezey.json and ibo.json. Re-run it
to regenerate; the files are never hand-edited.
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
OUT_IBO = OUT.with_name("ibo.json")

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


def orthonormal_iaos(
    mol: gto.Mole, s: numpy.ndarray, orbocc: numpy.ndarray
) -> numpy.ndarray:
    """PySCF's IAOs over MINAO, Loewdin-orthonormalized under S."""
    return lo.orth.vec_lowdin(lo.iao.iao(mol, orbocc), s)


def iao_populations(
    mol: gto.Mole, s: numpy.ndarray, iaos: numpy.ndarray, c: numpy.ndarray
) -> numpy.ndarray:
    """Each orbital's IAO population on each atom, orbital by atom."""
    c_iao = iaos.T @ s @ c
    return numpy.array(
        [
            numpy.einsum("ri,ri->i", c_iao[p0:p1], c_iao[p0:p1])
            for _, _, p0, p1 in lo.iao.reference_mol(mol).offset_nr_by_atom()
        ]
    ).T


def localize(
    mol: gto.Mole, mf: scf.hf.RHF, exponent: int, iaos: numpy.ndarray | None = None
) -> numpy.ndarray:
    """PySCF's Pipek-Mezey of the occupied orbitals, to a maximum: over
    Mulliken populations, or over those of the orthonormal `iaos`."""
    orbocc = mf.mo_coeff[:, mf.mo_occ > 0]
    if iaos is None:
        pm = lo.PM(mol, orbocc, mf)
        pm.pop_method = "mulliken"
    else:
        s = mol.intor_symmetric("int1e_ovlp")
        pm = lo.ibo.PipekMezey(mol, orbocc, iaos, s, exponent)
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


def case_reference(case: dict) -> tuple[dict, dict]:
    """The Pipek-Mezey and IBO references of one molecule, every exponent."""
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
    orbocc = mf.mo_coeff[:, mf.mo_occ > 0]
    iaos = orthonormal_iaos(mol, s, orbocc)
    occupied = iao_populations(mol, s, iaos, orbocc).sum(axis=0)
    charges = mol.atom_charges() - 2.0 * occupied
    ibos = []
    for exponent in EXPONENTS:
        c = localize(mol, mf, exponent, iaos)
        q_iao = iao_populations(mol, s, iaos, c)
        q = populations(mol, s, c)
        ibos.append(
            {
                "exponent": exponent,
                "functional": float((q_iao**exponent).sum()),
                "populations": [[float(v) for v in row] for row in q],
            }
        )
    common = {
        "name": case["name"],
        "basis": case["basis"],
        "charge": case["charge"],
        "atoms": [
            {"symbol": symbol, "z": gto.charge(symbol), "coord": list(coord)}
            for symbol, coord in case["atoms"]
        ],
        "electrons": int(mol.nelectron),
    }
    return (
        {**common, "localized": localized},
        {**common, "iao_charges": [float(q) for q in charges], "localized": ibos},
    )


def main() -> None:
    provenance = {
        "pyscf": pyscf.__version__,
        "numpy": numpy.__version__,
        "date": datetime.date.today().isoformat(),
        "host": platform.node(),
        "command": "uv run --group tools tools/emit_localization_reference.py",
    }
    pm_cases, ibo_cases = zip(*(case_reference(case) for case in CASES), strict=True)
    doc = {
        "description": "PySCF closed-shell RHF over this tree's data/basis "
        "files, its occupied orbitals localized by PySCF's Pipek-Mezey "
        "(Mulliken populations, conv_tol 1e-12, kernel alternated with "
        "stability_jacobi until stable): the functional sum_i sum_A "
        "(Q^A_ii)^p, and Q^A_ii per localized orbital (rows) and atom "
        "(columns, in the atoms' order); coordinates in bohr",
        "provenance": provenance,
        "cases": list(pm_cases),
    }
    ibo_doc = {
        "description": "PySCF closed-shell RHF over this tree's data/basis "
        "files, its occupied orbitals localized into intrinsic bond orbitals: "
        "PySCF's IAOs over MINAO, Loewdin-orthonormalized, and "
        "lo.ibo.PipekMezey over their populations (conv_tol 1e-12, kernel "
        "alternated with stability_jacobi until stable). iao_charges: Z_A "
        "minus twice the occupied orbitals' IAO population on atom A, per "
        "atom; per exponent, the functional sum_i sum_A (IAO population)^p, "
        "and each IBO's Mulliken population per orbital (rows) and atom "
        "(columns, in the atoms' order); coordinates in bohr",
        "provenance": provenance,
        "cases": list(ibo_cases),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for path, content in ((OUT, doc), (OUT_IBO, ibo_doc)):
        path.write_text(json.dumps(content, indent=1) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
