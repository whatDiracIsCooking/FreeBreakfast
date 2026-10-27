#!/usr/bin/env python3
"""Write data/aux_basis/cc-pV{D,T}Z-JKFIT.json: Weigend's JK fitting sets, from PySCF.

These are the auxiliary (density-fitting) sets for Coulomb and exchange that
pair with cc-pVDZ and cc-pVTZ: F. Weigend, Phys. Chem. Chem. Phys. 4, 4285
(2002), as Molpro's `weigend_jkfit.libmol` carries them. PySCF ships that
library as `pyscf/gto/basis/cc-pv{d,t}z-jkfit.dat`, and its density fitting
picks them by default, so a PySCF oracle and this tree fit over one auxiliary
basis. PySCF is Apache-2.0 licensed; its version is recorded in each document.

The Basis Set Exchange serves cc-pVTZ-JKFIT but has no cc-pVDZ-JKFIT at all:
the double-zeta set was never published. Molpro forms it by dropping the
highest angular momentum of the triple-zeta set, and PySCF's file is that
truncation. Both sets are taken from PySCF, so the pair stays one source.

Every element PySCF's files define is written: H, B-F, Ne, Al-Cl, Ar, Ga-Br,
Kr. Ne, Ar and Kr are Molpro's extrapolations, not published coefficients.

Each shell is one uncontracted primitive. Values are written as their
shortest round-trip decimal (`repr`), so the parser reads back exactly the
double PySCF does. s and p shells are `gto`, l >= 2 `gto_spherical`, as
tools/minao_from_pyscf.py writes them.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv \\
        uv run --group tools tools/jkfit_from_pyscf.py

writes both files. Re-run it after a PySCF bump; they are never hand-edited.
"""

from __future__ import annotations

import json
import pathlib

import pyscf
from pyscf import gto
from pyscf.data.elements import ELEMENTS

ELEMENT_RANGE = [1, *range(5, 11), *range(13, 19), *range(31, 37)]

OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "aux_basis"

# The BSE name each document calls itself, and the PySCF file it comes from.
SETS = {
    "cc-pVDZ-JKFIT": "cc-pvdz-jkfit.dat",
    "cc-pVTZ-JKFIT": "cc-pvtz-jkfit.dat",
}


def bse_shells(source: str, symbol: str) -> list[dict]:
    """PySCF's @p source for one element, as BSE `electron_shells`."""
    shells = []
    for entry in gto.basis.load(source.removesuffix(".dat"), symbol):
        ang, rows = entry[0], entry[1:]
        if not isinstance(rows[0], list):  # [l, kappa, rows...]
            rows = rows[1:]
        exponents = [repr(float(row[0])) for row in rows]
        columns = [
            [repr(float(row[k])) for row in rows] for k in range(1, len(rows[0]))
        ]
        shells.append(
            {
                "function_type": "gto" if ang <= 1 else "gto_spherical",
                "region": "",
                "angular_momentum": [ang],
                "exponents": exponents,
                "coefficients": columns,
            }
        )
    return shells


def document(name: str, source: str) -> dict:
    reference = {
        "reference_description": (
            "Weigend JK fitting set (F. Weigend, Phys. Chem. Chem. Phys. 4, "
            "4285 (2002), doi:10.1039/b204199p), from Molpro's "
            f"weigend_jkfit.libmol as pyscf/gto/basis/{source} carries it"
        ),
        "reference_keys": [],
    }
    return {
        "molssi_bse_schema": {"schema_type": "complete", "schema_version": "0.1"},
        "revision_description": (
            f"Converted from PySCF {pyscf.__version__} pyscf/gto/basis/{source} "
            "by tools/jkfit_from_pyscf.py"
        ),
        "elements": {
            str(z): {
                "electron_shells": bse_shells(source, ELEMENTS[z]),
                "references": [reference],
            }
            for z in ELEMENT_RANGE
        },
        "version": "1",
        "function_types": ["gto", "gto_spherical"],
        "names": [name],
        "tags": [],
        "family": "dunning",
        "description": f"{name}, Weigend's JK fitting set for {name[:-6]}",
        "role": "jkfit",
        "auxiliaries": {},
        "name": name,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, source in SETS.items():
        out = OUT_DIR / f"{name}.json"
        out.write_text(
            json.dumps(document(name, source), indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
