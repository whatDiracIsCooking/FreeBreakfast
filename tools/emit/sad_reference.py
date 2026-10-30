#!/usr/bin/env python3
"""Write the reference data free_breakfast.sad's tests read.

For each (basis, element) below, PySCF's spherically averaged atomic RHF
(`pyscf.scf.atom_hf.AtomSphAverageRHF`, the atomic SCF behind PySCF's own SAD
guess) is run to a tight tolerance, and its density is recorded beside the
basis it ran in: per l, the block over that momentum's shells for one m (the
density is the same for every m and zero between different m), over PySCF's
unit-normalised spherical AOs. The energy is recorded too, for a reader; no
test compares it. The basis is written in Basis Set Exchange JSON form, raw
coefficients over primitive-normalised Gaussians, so the tests read it with
free_breakfast.io.basis exactly as a production run would.

Every element is one whose per-l electron counts in PySCF's
NRSRHF_CONFIGURATION equal the plain Madelung fill free_breakfast.sad uses; the
two differ from Sc onwards for some transition metals and lanthanides, where a
comparison would test the configuration table rather than the SCF.

    uv run --group tools tools/emit/sad_reference.py

writes test/sad/data/atomic_scf.json. Re-run it to regenerate; the file
is never hand-edited.
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
    "cc-pvtz": ["C", "O"],
}

OUT = (
    pathlib.Path(__file__).resolve().parents[2]
    / "test"
    / "sad"
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
    # The density blocks are over unit-normalised AOs; the test rescales them to
    # free_breakfast's normalisation from S's diagonal.
    assert numpy.allclose(numpy.diag(mol.intor("int1e_ovlp")), 1.0, atol=1e-12)
    atm = atom_hf.AtomSphAverageRHF(mol)
    atm.conv_tol = 1e-14
    atm.conv_tol_grad = 1e-10
    atm.max_cycle = 200
    atm.kernel()
    assert atm.converged, f"{symbol}/{name} did not converge"
    density = atm.make_rdm1()
    # Per l, the first AO (one m) of each shell of that momentum, in shell order.
    firsts: list[list[int]] = []
    for shell in range(mol.nbas):
        ang = mol.bas_angular(shell)
        firsts.extend([] for _ in range(ang + 1 - len(firsts)))
        start = int(mol.ao_loc_nr()[shell])
        for k in range(mol.bas_nctr(shell)):
            firsts[ang].append(start + k * (2 * ang + 1))
    density_blocks = [
        [[float(density[i, j]) for j in idx] for i in idx] for idx in firsts
    ]
    return {
        "z": z,
        "symbol": symbol,
        "energy": float(atm.e_tot),
        "density_blocks": density_blocks,
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
        "description": "PySCF AtomSphAverageRHF densities, per l over one m and "
        "unit-normalised spherical AOs (conv_tol 1e-14, conv_tol_grad 1e-10), their "
        "energies, and the BSE-form bases they were computed in",
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": numpy.__version__,
            "date": datetime.date.today().isoformat(),
            "host": platform.node(),
            "command": "uv run --group tools tools/emit/sad_reference.py",
        },
        "bases": bases,
        "atoms": atoms,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
