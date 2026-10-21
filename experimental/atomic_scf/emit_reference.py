#!/usr/bin/env python3
"""Write the reference data experimental/atomic_scf's tests read.

For each (basis, element) below, PySCF's spherically averaged atomic RHF
(`pyscf.scf.atom_hf.AtomSphAverageRHF`, the atomic SCF behind PySCF's own SAD
guess) is run to a tight tolerance, and its total energy and per-l orbital
energies are recorded beside the basis it ran in. The basis is written in Basis
Set Exchange JSON form, raw coefficients over primitive-normalised Gaussians,
so the tests read it with free_breakfast.io.basis exactly as a production run
would.

Every element is one whose per-l electron counts in PySCF's
NRSRHF_CONFIGURATION equal the plain Madelung fill free_breakfast.sad uses; the
two differ from Sc onwards for some transition metals and lanthanides, where a
comparison would test the configuration table rather than the SCF.

    uv run --group tools experimental/atomic_scf/emit_reference.py

writes test/experimental/data/atomic_scf.json. Re-run it to regenerate; the
file is never hand-edited.
"""

from __future__ import annotations

import datetime
import json
import pathlib
import platform

import numpy
import pyscf
from pyscf import gto
from pyscf.scf import atom_hf

CASES: dict[str, list[str]] = {
    "sto-3g": ["H", "He", "Li", "Be", "C", "N", "O", "Ne", "Na", "Ar", "Zn"],
    "cc-pvdz": ["H", "He", "C", "O", "Ne"],
}

OUT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "test"
    / "experimental"
    / "data"
    / "atomic_scf.json"
)


def bse_shells(name: str, symbol: str) -> list[dict]:
    """PySCF's raw basis for one element, as BSE `electron_shells`."""
    shells = []
    for entry in gto.basis.load(name, symbol):
        ang, rows = entry[0], entry[1:]
        if not isinstance(rows[0], list):  # [l, kappa, rows...]
            rows = rows[1:]
        exponents = [repr(float(row[0])) for row in rows]
        columns = [
            [repr(float(row[k])) for row in rows] for k in range(1, len(rows[0]))
        ]
        shells.append(
            {
                "function_type": "gto_spherical",
                "angular_momentum": [ang],
                "exponents": exponents,
                "coefficients": columns,
            }
        )
    return shells


def atomic_scf(name: str, symbol: str) -> dict:
    """PySCF's spherically averaged atomic RHF for one element."""
    z = gto.charge(symbol)
    mol = gto.M(atom=f"{symbol} 0 0 0", basis=name, spin=z % 2, verbose=0)
    atm = atom_hf.AtomSphAverageRHF(mol)
    atm.conv_tol = 1e-12
    atm.max_cycle = 200
    atm.kernel()
    assert atm.converged, f"{symbol}/{name} did not converge"
    ao_l = atom_hf._angular_momentum_for_each_ao(mol)
    # eig() returns each l's eigenvalues repeated over its 2l+1 m, l by l, in
    # the order mo_energy is laid out: take one m's worth per l.
    orbital_energies = []
    start = 0
    for ang in range(int(ao_l.max()) + 1):
        count = int(numpy.count_nonzero(ao_l == ang))
        degen = 2 * ang + 1
        block = atm.mo_energy[start : start + count]
        orbital_energies.append([float(e) for e in block[::degen]])
        start += count
    return {
        "z": z,
        "symbol": symbol,
        "energy": float(atm.e_tot),
        "orbital_energies": orbital_energies,
    }


def main() -> None:
    bases = {}
    atoms = []
    for name, symbols in CASES.items():
        elements = {}
        for symbol in symbols:
            elements[str(gto.charge(symbol))] = {
                "electron_shells": bse_shells(name, symbol)
            }
            atoms.append({"basis": name, **atomic_scf(name, symbol)})
        bases[name] = {"elements": elements}
    doc = {
        "description": "PySCF AtomSphAverageRHF energies and per-l orbital energies "
        "(conv_tol 1e-12), and the BSE-form bases they were computed in",
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": numpy.__version__,
            "date": datetime.date.today().isoformat(),
            "host": platform.node(),
            "command": "uv run --group tools experimental/atomic_scf/emit_reference.py",
        },
        "bases": bases,
        "atoms": atoms,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
