#!/usr/bin/env python3
"""Write data/aux_basis/cc-pV{D,T}Z-JKFIT.json and def2-universal-JKFIT.json:
Weigend's JK fitting sets, from PySCF.

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

Every element PySCF's cc files define is written: H, B-F, Ne, Al-Cl, Ar,
Ga-Br, Kr. Ne, Ar and Kr are Molpro's extrapolations, not published
coefficients.

def2-universal-JKFIT (F. Weigend, J. Comput. Chem. 29, 167 (2008)) is the one
JK fitting set for every def2 orbital set; PySCF maps def2-svp-jkfit and
def2-tzvp-jkfit onto the same file, which is BSE's copy of Turbomole's. It is
written for H-Kr, the range data/basis/ carries def2 for, all-electron.

The cc shells are each one uncontracted primitive; def2-universal-JKFIT has a
few contracted ones. Values are written as their
shortest round-trip decimal (`repr`), so the parser reads back exactly the
double PySCF does. s and p shells are `gto`, l >= 2 `gto_spherical`, as
tools/minao_from_pyscf.py writes them.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv \\
        uv run --group tools tools/jkfit_from_pyscf.py

writes all three. Re-run it after a PySCF bump; they are never hand-edited.
"""

from __future__ import annotations

import json
import pathlib

import pyscf
from pyscf import gto
from pyscf.data.elements import ELEMENTS

CC_ELEMENTS = [1, *range(5, 11), *range(13, 19), *range(31, 37)]

OUT_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "aux_basis"

CC_REFERENCE = (
    "Weigend JK fitting set (F. Weigend, Phys. Chem. Chem. Phys. 4, "
    "4285 (2002), doi:10.1039/b204199p), from Molpro's "
    "weigend_jkfit.libmol as pyscf/gto/basis/{source} carries it"
)
DEF2_REFERENCE = (
    "Weigend universal JK fitting set (F. Weigend, J. Comput. Chem. 29, "
    "167 (2008), doi:10.1002/jcc.20702), from Turbomole as "
    "pyscf/gto/basis/{source} carries it"
)

# The BSE name each document calls itself: the PySCF file it comes from, its
# elements, its reference, its BSE family, and the orbital sets it fits.
SETS = {
    "cc-pVDZ-JKFIT": (
        "cc-pvdz-jkfit.dat",
        CC_ELEMENTS,
        CC_REFERENCE,
        "dunning",
        "cc-pVDZ",
    ),
    "cc-pVTZ-JKFIT": (
        "cc-pvtz-jkfit.dat",
        CC_ELEMENTS,
        CC_REFERENCE,
        "dunning",
        "cc-pVTZ",
    ),
    "def2-universal-JKFIT": (
        "def2-universal-jkfit.dat",
        list(range(1, 37)),
        DEF2_REFERENCE,
        "ahlrichs",
        "the def2 sets",
    ),
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


def document(
    name: str,
    source: str,
    elements: list[int],
    reference: str,
    family: str,
    fits: str,
) -> dict:
    reference = {
        "reference_description": reference.format(source=source),
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
            for z in elements
        },
        "version": "1",
        "function_types": ["gto", "gto_spherical"],
        "names": [name],
        "tags": [],
        "family": family,
        "description": f"{name}, Weigend's JK fitting set for {fits}",
        "role": "jkfit",
        "auxiliaries": {},
        "name": name,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in SETS.items():
        out = OUT_DIR / f"{name}.json"
        out.write_text(
            json.dumps(document(name, *spec), indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
