#!/usr/bin/env python3
"""Write the reference data the Localization step's tests read.

For each case below, PySCF's closed-shell RHF (as scf_reference.py runs
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

Open shells, in rohf.json: PySCF's DF-ROHF (as df_reference.py runs it, over
the same orbital and fitting sets) of a doublet and a triplet, whose doubly
and singly occupied orbitals are localized each within its own space, by both
methods at both exponents. The IAOs are those of all the occupied orbitals,
doubly and singly together, so that they span both spaces; each space is then
localized over them. Recorded per space: the functional and the populations,
as above; and the IAO charges, Z_A minus twice the doubly occupied orbitals'
IAO population on A and once the singly occupied ones'.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/localization_reference.py

writes test/gtest/data/localization/pipek_mezey.json, ibo.json and rohf.json.
Re-run it to regenerate; the files are never hand-edited.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import platform

import numpy
import pyscf
from df_reference import ROHF_SCF_CASES, df_scf, molecules
from pyscf import gto, lo, scf
from scf_reference import AMMONIUM, BASES, WATER, pyscf_basis

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "localization" / "pipek_mezey.json"
OUT_IBO = OUT.with_name("ibo.json")
OUT_ROHF = OUT.with_name("rohf.json")

CASES = [
    {"name": "h2o_sto3g", "atoms": WATER, "basis": "sto-3g", "charge": 0},
    {"name": "h2o_ccpvdz", "atoms": WATER, "basis": "cc-pvdz", "charge": 0},
    {"name": "nh4_ccpvdz", "atoms": AMMONIUM, "basis": "cc-pvdz", "charge": 1},
]

# The open shells, by df_rohf.json's names: a doublet, whose singly occupied
# space is one orbital, and a triplet, whose two are localized as a pair.
ROHF_NAMES = ("h2o_cation_ccpvdz", "ch2_ccpvdz")

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
    mol: gto.Mole,
    mf: scf.hf.SCF,
    exponent: int,
    iaos: numpy.ndarray | None = None,
    orbocc: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """PySCF's Pipek-Mezey of the occupied orbitals (or of `orbocc`), to a
    maximum: over Mulliken populations, or over those of the orthonormal
    `iaos`."""
    if orbocc is None:
        orbocc = mf.mo_coeff[:, mf.mo_occ > 0]
    if orbocc.shape[1] < 2:
        # No pair to rotate: the orbital is its own maximum.
        return orbocc
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


def rohf_case(case: dict) -> dict:
    """The per-space Pipek-Mezey and IBO references of one open shell."""
    mol, auxmol, aux_name = molecules(case)
    mf = df_scf(mol, auxmol, scf.ROHF)
    _, _, stable, _ = mf.stability(return_status=True)
    assert stable, f"{case['name']} is not internally stable"
    n_alpha, n_beta = mol.nelec
    s = mol.intor_symmetric("int1e_ovlp")
    spaces = {
        "doubly": mf.mo_coeff[:, mf.mo_occ == 2],
        "singly": mf.mo_coeff[:, mf.mo_occ == 1],
    }
    assert spaces["doubly"].shape[1] == n_beta
    assert spaces["singly"].shape[1] == n_alpha - n_beta
    iaos = orthonormal_iaos(mol, s, mf.mo_coeff[:, mf.mo_occ > 0])
    charges = mol.atom_charges() - sum(
        occupation * iao_populations(mol, s, iaos, spaces[space]).sum(axis=0)
        for space, occupation in (("doubly", 2.0), ("singly", 1.0))
    )

    def record(exponent: int, over_iaos: bool) -> dict:
        entry: dict = {"exponent": exponent}
        for space, orbocc in spaces.items():
            c = localize(mol, mf, exponent, iaos if over_iaos else None, orbocc)
            q = populations(mol, s, c)
            q_l = iao_populations(mol, s, iaos, c) if over_iaos else q
            entry[space] = {
                "functional": float((q_l**exponent).sum()),
                "populations": [[float(v) for v in row] for row in q],
            }
        return entry

    return {
        "name": case["name"],
        "basis": case["basis"],
        "aux_basis": aux_name,
        "charge": case["charge"],
        "multiplicity": case["spin"] + 1,
        "atoms": [
            {"symbol": symbol, "z": gto.charge(symbol), "coord": list(coord)}
            for symbol, coord in case["atoms"]
        ],
        "electrons": int(mol.nelectron),
        "n_alpha": int(n_alpha),
        "n_beta": int(n_beta),
        "pipek_mezey": [record(exponent, False) for exponent in EXPONENTS],
        "ibo": [record(exponent, True) for exponent in EXPONENTS],
        "iao_charges": [float(q) for q in charges],
    }


def main() -> None:
    provenance = {
        "pyscf": pyscf.__version__,
        "numpy": numpy.__version__,
        "date": datetime.date.today().isoformat(),
        "host": platform.node(),
        "command": "uv run --group tools tools/emit/localization_reference.py",
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
    rohf_doc = {
        "description": "PySCF DF-ROHF over this tree's data/basis and "
        "data/aux_basis files, its doubly (mo_occ 2) and singly (mo_occ 1) "
        "occupied orbitals localized each within its own space, as in "
        "pipek_mezey.json (pipek_mezey) and ibo.json (ibo), the IAOs being "
        "those of every occupied orbital. Per exponent and space: the "
        "functional and each localized orbital's Mulliken population per "
        "orbital (rows) and atom (columns, in the atoms' order). iao_charges: "
        "Z_A minus twice the doubly and once the singly occupied orbitals' "
        "IAO population on atom A; coordinates in bohr",
        "provenance": provenance,
        "cases": [
            rohf_case(case) for case in ROHF_SCF_CASES if case["name"] in ROHF_NAMES
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for path, content in ((OUT, doc), (OUT_IBO, ibo_doc), (OUT_ROHF, rohf_doc)):
        path.write_text(json.dumps(content, indent=1) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
