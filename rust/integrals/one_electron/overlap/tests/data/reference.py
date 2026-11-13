"""PySCF's spherical AO overlap S for the Rust overlap crate's tests.

Usage (from the checkout root; PySCF is in the `tools` dependency group):

    uv run --group tools python \
        rust/integrals/one_electron/overlap/tests/data/reference.py \
        rust/integrals/one_electron/overlap/tests/data/h2o.xyz cc-pvdz \
        h2o_cc-pvdz.npy

Two things are pinned to what free_breakfast itself reads, so the comparison
measures the integrals and nothing else:

- The basis is the same BSE JSON document the binary compiles in
  (data/basis/), found by the same folded name (lowercased, with `-`, `_` and
  spaces dropped; src/io/basis/builtin_basis.cpp.in), and converted by
  tools/atomic_cd/eri_quartet.py's basis_from_bse. PySCF's own copy of a
  named basis can differ in its digits. A path to a BSE JSON file works too.
- The coordinates are converted to bohr here, with the tree's CODATA 2018
  Bohr radius (src/core/constants), and handed to PySCF in bohr. PySCF's own
  Angstrom conversion uses CODATA 2010's, about 2e-10 apart, which would move
  S by more than the bar.

- The contraction normalization is free_breakfast's. Its basis folds each
  primitive's norm into the coefficients (BasisStamp::primitive_normalized)
  but never renormalizes a contraction, so an AO's norm is that of the BSE
  coefficients as published: 1 only to their printed digits (about 1e-10 off
  in STO-3G). PySCF rescales every contraction to exactly 1. So the two are
  related by S_fb = D S_pyscf D, D_ii the published contraction's norm, which
  this computes on PySCF's own primitives (gto_norm, gaussian_int: the sum
  PySCF normalizes by) and applies. It prints how far D is from 1.

S is written as a float64 .npy, nao x nao, in PySCF's spherical AO order (the
order free_breakfast's spherical layout follows).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from pyscf import gto

ROOT = Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT))

from tools.atomic_cd.eri_quartet import basis_from_bse  # noqa: E402

# kAngstromPerBohr, src/core/constants/constants.h (CODATA 2018).
ANGSTROM_PER_BOHR = 0.529177210903


def fold(name: str) -> str:
    """resolve_basis's folding: lowercased, `-`, `_` and spaces dropped."""
    return "".join(c for c in name.lower() if c not in "-_ ")


def bse_document(spec: str) -> dict:
    """The carried basis `spec` names, or the BSE JSON file it is a path to."""
    for path in sorted((ROOT / "data" / "basis").glob("*.json")):
        doc = json.loads(path.read_text())
        if fold(doc.get("name", "")) == fold(spec):
            return doc
    path = Path(spec)
    if path.is_file():
        return json.loads(path.read_text())
    raise SystemExit(f"{spec!r} is neither a carried basis nor a file")


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
    """Each AO's norm under the published coefficients, per AO in PySCF's
    order for one atom: shell by shell, column by column, 2l+1 per column.

    `shells` is PySCF's internal format, [l, [exponent, c_1, c_2, ...], ...].
    A column's norm^2 is sum_ij c_i c_j n_i n_j int r^(2l+2) e^(-(a_i+a_j) r^2),
    n the primitive radial norm: what PySCF divides the column by.
    """
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


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    xyz, spec, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    doc = bse_document(spec)
    atoms = read_xyz(xyz)
    symbols = sorted({symbol for symbol, _ in atoms})
    basis = {s: basis_from_bse(doc, s) for s in symbols}
    mol = gto.M(
        atom=atoms,
        basis=basis,
        unit="Bohr",
        cart=False,
        verbose=0,
    )
    d = np.array([n for symbol, _ in atoms for n in contraction_norms(basis[symbol])])
    if d.size != mol.nao:
        raise SystemExit(f"{d.size} contraction norms for {mol.nao} AOs")
    s = d[:, None] * mol.intor("int1e_ovlp") * d[None, :]
    np.save(out, s)
    print(
        f"{out}: {mol.nao} x {mol.nao} S of {xyz.name} in {doc.get('name', spec)};"
        f" max |D - 1| = {np.abs(d - 1).max():.3e}"
    )


if __name__ == "__main__":
    main()
