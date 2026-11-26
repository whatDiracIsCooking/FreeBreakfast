"""A molecule in PySCF as free_breakfast reads it, for fb-steps' reference
scripts (`b_reference.py`, `pm_reference.py`,
`open_localization_reference.py`), which load it by path. Moved here from
the old three-center crate's `tests/data/reference.py`, its integrals left
behind:

- Both sets are the BSE JSON documents the binary compiles in (data/basis/
  for the orbital set, data/aux_basis/ for the fitting set), found by the
  same folded name (lowercased, with `-`, `_` and spaces dropped), and
  converted by tools/atomic_cd/eri_quartet.py's basis_from_bse.
- The coordinates are converted to bohr here, with the tree's CODATA 2018
  Bohr radius, and handed to PySCF in bohr.
- The contraction normalization is free_breakfast's, which never
  renormalizes a contraction, while PySCF rescales every one to exactly 1:
  a quantity's element is PySCF's times D_ii for each index i, D_ii the
  published contraction's norm, on both sets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from pyscf import gto

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from tools.atomic_cd.eri_quartet import basis_from_bse  # noqa: E402

# kAngstromPerBohr, src/core/constants/constants.h (CODATA 2018).
ANGSTROM_PER_BOHR = 0.529177210903


def fold(name: str) -> str:
    """resolve_basis's folding: lowercased, `-`, `_` and spaces dropped."""
    return "".join(c for c in name.lower() if c not in "-_ ")


def bse_document(spec: str, directory: str) -> dict:
    """The carried set `spec` names in data/<directory>/."""
    for path in sorted((ROOT / "data" / directory).glob("*.json")):
        doc = json.loads(path.read_text())
        if fold(doc.get("name", "")) == fold(spec) or fold(path.stem) == fold(spec):
            return doc
    raise SystemExit(f"{spec!r} is not a carried set of data/{directory}/")


def read_xyz(path: Path) -> list[tuple[str, tuple[float, float, float]]]:
    """The atoms of an XYZ file, positions in bohr."""
    lines = path.read_text().splitlines()
    count = int(lines[0])
    atoms = []
    for line in lines[2 : 2 + count]:
        symbol, *xyz = line.split()[:4]
        atoms.append((symbol, tuple(float(v) / ANGSTROM_PER_BOHR for v in xyz)))
    return atoms


def contraction_norms(shells: list) -> list[float]:
    """Each function's norm under the published coefficients, per function
    in PySCF's order for one atom."""
    norms = []
    for ell, *rows in shells:
        exps = np.array([row[0] for row in rows])
        prim = np.array([gto.gto_norm(ell, a) for a in exps])
        pair = np.array(
            [[gto.gaussian_int(2 * ell + 2, a + b) for b in exps] for a in exps]
        )
        for col in range(1, len(rows[0])):
            c = np.array([row[col] for row in rows]) * prim
            norms += [float(np.sqrt(c @ pair @ c))] * (2 * ell + 1)
    return norms


def named(xyz: Path, orbital: str, aux: str) -> tuple:
    """The molecule in both carried sets, and each set's D."""
    atoms = read_xyz(xyz)
    symbols = sorted({symbol for symbol, _ in atoms})
    mols, ds = [], []
    for spec, directory in [(orbital, "basis"), (aux, "aux_basis")]:
        doc = bse_document(spec, directory)
        basis = {s: basis_from_bse(doc, s) for s in symbols}
        mol = gto.M(atom=atoms, basis=basis, unit="Bohr", cart=False, verbose=0)
        d = np.array(
            [n for symbol, _ in atoms for n in contraction_norms(basis[symbol])]
        )
        if d.size != mol.nao:
            raise SystemExit(f"{d.size} contraction norms for {mol.nao} functions")
        mols.append(mol)
        ds.append(d)
    return mols[0], mols[1], ds[0], ds[1]
