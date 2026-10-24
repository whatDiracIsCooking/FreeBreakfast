#!/usr/bin/env python3
"""Write data/basis/minao.json: the MINAO minimal basis, taken from PySCF.

MINAO is Gerald Knizia's free-atom minimal basis: for each element, the first
n_occ contracted functions of each angular momentum of cc-pVTZ, which are
spherically averaged ground-state Hartree-Fock atomic orbitals. It is the B2
of the intrinsic atomic orbital construction (Knizia, J. Chem. Theory Comput.
9, 4834 (2013), Appendix B), where Molpro introduced it under that name.

The Basis Set Exchange does not serve it (its `mini` is Huzinaga's MINI, a
different set), so this tree takes it from PySCF, which ships Molpro's table
as `pyscf/gto/basis/minao.py`. PySCF's own IAO (`pyscf.lo.iao`) defaults to
that same table, so a PySCF oracle and this tree build their IAOs over one
B2. PySCF is Apache-2.0 licensed; its version is recorded in the document.

Only Z 1-18 and 20-36 are written. PySCF's table has no potassium (cc-pVTZ
defines none), and from rubidium on it is cut from cc-pVTZ-PP, a valence set
for an effective core potential, which this all-electron tree cannot use.

The coefficients are copied as PySCF holds them: raw, over primitive
normalised Gaussians, as the Basis Set Exchange stores them. Each value is
written as its shortest round-trip decimal (`repr`), so the parser reads back
exactly the double PySCF does. s and p shells are `gto`, d shells
`gto_spherical`, the Basis Set Exchange's own convention for cc-pVTZ.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv \\
        uv run --group tools tools/minao_from_pyscf.py

writes data/basis/minao.json. Re-run it after a PySCF bump, then regenerate
data/sad/ (data/sad/README.md); the file is never hand-edited.
"""

from __future__ import annotations

import json
import pathlib

import pyscf
from pyscf import gto
from pyscf.data.elements import ELEMENTS

ELEMENT_RANGE = [*range(1, 19), *range(20, 37)]

OUT = pathlib.Path(__file__).resolve().parents[1] / "data" / "basis" / "minao.json"


def bse_shells(symbol: str) -> list[dict]:
    """PySCF's MINAO for one element, as BSE `electron_shells`."""
    shells = []
    for entry in gto.basis.load("minao", symbol):
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


def main() -> None:
    reference = {
        "reference_description": (
            "MINAO: the first n_occ contracted functions of cc-pVTZ "
            "(G. Knizia, J. Chem. Theory Comput. 9, 4834 (2013), "
            "doi:10.1021/ct400687b), as pyscf/gto/basis/minao.py carries them"
        ),
        "reference_keys": [],
    }
    document = {
        "molssi_bse_schema": {"schema_type": "complete", "schema_version": "0.1"},
        "revision_description": (
            f"Converted from PySCF {pyscf.__version__} pyscf/gto/basis/minao.py "
            "by tools/minao_from_pyscf.py"
        ),
        "elements": {
            str(z): {
                "electron_shells": bse_shells(ELEMENTS[z]),
                "references": [reference],
            }
            for z in ELEMENT_RANGE
        },
        "version": "1",
        "function_types": ["gto", "gto_spherical"],
        "names": ["MINAO"],
        "tags": [],
        "family": "minao",
        "description": "MINAO, Knizia's cc-pVTZ free-atom minimal basis",
        "role": "orbital",
        "auxiliaries": {},
        "name": "MINAO",
    }
    OUT.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
