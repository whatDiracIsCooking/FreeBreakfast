#!/usr/bin/env python3
"""Write the reference data the Scf step's tests read.

For each case below, PySCF's closed-shell RHF, or for an open-shell case its
ROHF, is run to a tight tolerance in the spherical basis, over this tree's own
copy of the basis (`data/basis/*.json`, the file `resolve_basis` compiles in),
so a difference between PySCF's basis library and the Basis Set Exchange's
cannot pass for an SCF error. The geometry is recorded in bohr, the unit both
sides then read it in. Recorded: the total energy, the nuclear repulsion, and
the orbital energies, all invariant under the AO normalisation, which differs
between the two codes. An ROHF case also records its spin (2S, PySCF's
convention), its multiplicity and n_alpha / n_beta; its orbital energies are
the eigenvalues of PySCF's default Roothaan effective Fock, one
canonicalization of several, and it is checked internally stable before it
is written.

BLAS and OpenMP are pinned to one thread, and nothing run-dependent (date,
host) is written, so a rerun regenerates both files byte for byte.

    uv run --group tools tools/emit/scf_reference.py

writes test/gtest/data/scf/rhf.json and rohf.json. Re-run it to regenerate;
the files are never hand-edited.
"""

from __future__ import annotations

import os

# Run as a script, before numpy loads its BLAS: one thread, so every reduction
# runs in one order and the output is reproducible bit for bit. Not on import:
# the other emitters that take this module's geometries pin their own.
if __name__ == "__main__":
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"

import json  # noqa: E402
import pathlib  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
from pyscf import gto, lib, scf  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "scf" / "rhf.json"
ROHF_OUT = OUT.with_name("rohf.json")

# The builtin basis name the C++ side resolves, and its file under data/basis.
BASES = {
    "sto-3g": "sto-3g.json",
    "cc-pvdz": "cc-pVDZ.json",
    "cc-pvtz": "cc-pVTZ.json",
}

# Water near its equilibrium geometry, no atom on an axis past O; and NH4+,
# which carries a charge, so the SAD guess (which places sum Z electrons)
# must be rescaled. Bohr.
WATER = [
    ("O", (0.0, 0.0, 0.2216)),
    ("H", (0.0, 1.4309, -0.8867)),
    ("H", (0.0, -1.4309, -0.8867)),
]
AMMONIUM = [
    ("N", (0.0, 0.0, 0.0)),
    ("H", (1.1, 1.1, 1.1)),
    ("H", (-1.1, -1.1, 1.1)),
    ("H", (-1.1, 1.1, -1.1)),
    ("H", (1.1, -1.1, -1.1)),
]
# CO, 2.10 bohr apart along no axis, so every f component of cc-pVTZ's f
# shells is reached; the geometry of test/gtest/data/coulomb/co_ccpvtz.json.
CARBON_MONOXIDE = [
    ("C", (0.13, -0.21, 0.17)),
    ("O", (0.77, 0.86, 1.86)),
]

# The open-shell molecules. OH (0.97 A) and NH2 (1.024 A, 103 deg) are
# doublets; CH2 (1.078 A, 134 deg) is the 3B1 triplet, tilted out of every
# coordinate plane so every f component of cc-pVTZ is reached. (O2's triplet
# is not here: its default-guess ROHF is internally unstable, and the stable
# one stalls short of conv_tol_grad along the soft rotation in its pi* pair.)
# Bohr.
HYDROXYL = [
    ("O", (0.0, 0.0, 0.0)),
    ("H", (0.0, 1.05, 1.5)),
]
AMIDOGEN = [
    ("N", (0.0, 0.0, 0.0)),
    ("H", (0.0, 1.514, -1.205)),
    ("H", (0.0, -1.514, -1.205)),
]
METHYLENE = [
    ("C", (0.13, -0.21, 0.17)),
    ("H", (-1.4343, 1.0313, 0.5717)),
    ("H", (0.9901, -0.8926, -1.5457)),
]

CASES = [
    {"name": "h2o_sto3g", "atoms": WATER, "basis": "sto-3g", "charge": 0},
    {"name": "h2o_ccpvdz", "atoms": WATER, "basis": "cc-pvdz", "charge": 0},
    {"name": "nh4_ccpvdz", "atoms": AMMONIUM, "basis": "cc-pvdz", "charge": 1},
    {"name": "co_ccpvtz", "atoms": CARBON_MONOXIDE, "basis": "cc-pvtz", "charge": 0},
]

# "spin" is 2S = n_alpha - n_beta. H2O+ is the cation, at the neutral's
# geometry.
ROHF_CASES = [
    {"name": "oh_sto3g", "atoms": HYDROXYL, "basis": "sto-3g", "charge": 0, "spin": 1},
    {
        "name": "oh_ccpvdz",
        "atoms": HYDROXYL,
        "basis": "cc-pvdz",
        "charge": 0,
        "spin": 1,
    },
    {
        "name": "nh2_ccpvdz",
        "atoms": AMIDOGEN,
        "basis": "cc-pvdz",
        "charge": 0,
        "spin": 1,
    },
    {
        "name": "ch2_ccpvdz",
        "atoms": METHYLENE,
        "basis": "cc-pvdz",
        "charge": 0,
        "spin": 2,
    },
    {
        "name": "h2o_cation_ccpvdz",
        "atoms": WATER,
        "basis": "cc-pvdz",
        "charge": 1,
        "spin": 1,
    },
    {
        "name": "ch2_ccpvtz",
        "atoms": METHYLENE,
        "basis": "cc-pvtz",
        "charge": 0,
        "spin": 2,
    },
]


def pyscf_basis(file: str, symbols: set[str]) -> dict[str, list]:
    """This tree's BSE JSON basis, in PySCF's form, for the given elements.

    A BSE shell of several momenta (sto-3g's sp) carries one coefficient
    column per momentum; PySCF wants one shell per momentum.
    """
    doc = json.loads((ROOT / "data" / "basis" / file).read_text())
    out: dict[str, list] = {}
    for symbol in symbols:
        shells = []
        for shell in doc["elements"][str(gto.charge(symbol))]["electron_shells"]:
            exponents = [float(e) for e in shell["exponents"]]
            momenta = shell["angular_momentum"]
            columns = [[float(c) for c in col] for col in shell["coefficients"]]
            if len(momenta) == 1:
                groups = [(momenta[0], columns)]
            else:
                assert len(momenta) == len(columns)
                groups = [
                    (ang, [col]) for ang, col in zip(momenta, columns, strict=True)
                ]
            for ang, cols in groups:
                rows = [[e, *(col[k] for col in cols)] for k, e in enumerate(exponents)]
                shells.append([ang, *rows])
        out[symbol] = shells
    return out


def molecule(case: dict) -> gto.Mole:
    """One case's molecule, over this tree's basis file."""
    symbols = {symbol for symbol, _ in case["atoms"]}
    return gto.M(
        atom=[[symbol, coord] for symbol, coord in case["atoms"]],
        unit="Bohr",
        basis=pyscf_basis(BASES[case["basis"]], symbols),
        charge=case["charge"],
        spin=case.get("spin", 0),
        cart=False,
        verbose=0,
    )


def converge(mf: scf.hf.SCF, name: str) -> None:
    """Run mf to this file's tolerances; it must converge."""
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.max_cycle = 200
    mf.kernel()
    assert mf.converged, f"{name} did not converge"


def rhf(case: dict) -> dict:
    """PySCF's converged RHF for one case."""
    mol = molecule(case)
    mf = scf.RHF(mol)
    converge(mf, case["name"])
    return {
        "name": case["name"],
        "basis": case["basis"],
        "charge": case["charge"],
        "atoms": [
            {"symbol": symbol, "z": gto.charge(symbol), "coord": list(coord)}
            for symbol, coord in case["atoms"]
        ],
        "nao": int(mol.nao),
        "electrons": int(mol.nelectron),
        "energy": float(mf.e_tot),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "mo_energies": [float(e) for e in mf.mo_energy],
    }


def rohf(case: dict) -> dict:
    """PySCF's converged, internally stable ROHF for one case."""
    mol = molecule(case)
    mf = scf.ROHF(mol)
    converge(mf, case["name"])
    # The default guess can land on a saddle; this file holds minima only.
    _, _, stable, _ = mf.stability(return_status=True)
    assert stable, f"{case['name']} is not internally stable"
    n_alpha, n_beta = mol.nelec
    assert n_alpha - n_beta == case["spin"]
    return {
        "name": case["name"],
        "basis": case["basis"],
        "charge": case["charge"],
        "spin": case["spin"],
        "multiplicity": case["spin"] + 1,
        "atoms": [
            {"symbol": symbol, "z": gto.charge(symbol), "coord": list(coord)}
            for symbol, coord in case["atoms"]
        ],
        "nao": int(mol.nao),
        "electrons": int(mol.nelectron),
        "n_alpha": int(n_alpha),
        "n_beta": int(n_beta),
        "energy": float(mf.e_tot),
        "nuclear_repulsion": float(mol.energy_nuc()),
        "mo_energies": [float(e) for e in mf.mo_energy],
    }


def main() -> None:
    lib.num_threads(1)
    provenance = {
        "pyscf": pyscf.__version__,
        "numpy": numpy.__version__,
        "threads": 1,
        "command": "uv run --group tools tools/emit/scf_reference.py",
    }
    docs = {
        OUT: {
            "description": "PySCF closed-shell RHF (spherical, conv_tol 1e-12, "
            "conv_tol_grad 1e-9) over this tree's data/basis files: total "
            "energy, nuclear repulsion and orbital energies (hartree); "
            "coordinates in bohr",
            "provenance": provenance,
            "cases": [rhf(case) for case in CASES],
        },
        ROHF_OUT: {
            "description": "PySCF ROHF (spherical, conv_tol 1e-12, conv_tol_grad "
            "1e-9, internally stable) over this tree's data/basis files: total "
            "energy, nuclear repulsion and orbital energies, the eigenvalues "
            "of PySCF's default Roothaan effective Fock (hartree); spin is 2S "
            "= n_alpha - n_beta; coordinates in bohr",
            "provenance": provenance,
            "cases": [rohf(case) for case in ROHF_CASES],
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for path, doc in docs.items():
        path.write_text(json.dumps(doc, indent=1) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
