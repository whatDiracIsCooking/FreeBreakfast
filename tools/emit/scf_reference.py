#!/usr/bin/env python3
"""Write the reference data the Scf step's tests read.

For each case below, PySCF's closed-shell RHF is run to a tight tolerance in
the spherical basis, over this tree's own copy of the basis
(`data/basis/*.json`, the file `resolve_basis` compiles in), so a difference
between PySCF's basis library and the Basis Set Exchange's cannot pass for an
SCF error. The geometry is recorded in bohr, the unit both sides then read it
in. Recorded: the total energy, the nuclear repulsion, and the orbital
energies, all invariant under the AO normalisation, which differs between the
two codes.

    uv run --group tools tools/emit/scf_reference.py

writes test/gtest/data/scf/rhf.json. Re-run it to regenerate; the file is
never hand-edited.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import platform

import numpy
import pyscf
from pyscf import gto, scf

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "scf" / "rhf.json"

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

CASES = [
    {"name": "h2o_sto3g", "atoms": WATER, "basis": "sto-3g", "charge": 0},
    {"name": "h2o_ccpvdz", "atoms": WATER, "basis": "cc-pvdz", "charge": 0},
    {"name": "nh4_ccpvdz", "atoms": AMMONIUM, "basis": "cc-pvdz", "charge": 1},
    {"name": "co_ccpvtz", "atoms": CARBON_MONOXIDE, "basis": "cc-pvtz", "charge": 0},
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


def rhf(case: dict) -> dict:
    """PySCF's converged RHF for one case."""
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


def main() -> None:
    doc = {
        "description": "PySCF closed-shell RHF (spherical, conv_tol 1e-12, "
        "conv_tol_grad 1e-9) over this tree's data/basis files: total energy, "
        "nuclear repulsion and orbital energies (hartree); coordinates in bohr",
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": numpy.__version__,
            "date": datetime.date.today().isoformat(),
            "host": platform.node(),
            "command": "uv run --group tools tools/emit/scf_reference.py",
        },
        "cases": [rhf(case) for case in CASES],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
