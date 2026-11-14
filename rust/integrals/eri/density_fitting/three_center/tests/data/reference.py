"""PySCF's spherical three-center integrals (mn|P), `int3c2e`, for the Rust
three-center crate's tests: the metric crate's reference.py over an orbital
set and a fitting set.

Usage (from the checkout root; PySCF is in the `tools` dependency group):

    D=rust/integrals/eri/density_fitting/three_center/tests/data
    OMP_NUM_THREADS=8 uv run --group tools python $D/reference.py \\
        $D/h2o.xyz cc-pvdz cc-pvdz-jkfit $D/h2o_cc-pvdz_cc-pvdz-jkfit.npy
    OMP_NUM_THREADS=8 uv run --group tools python $D/reference.py \\
        --manifest $D/df_high_aux.json $D/df_high_aux.npy

The first form pins three things to what free_breakfast itself reads, so
the comparison measures the integrals and nothing else:

- Both sets are the BSE JSON documents the binary compiles in (data/basis/
  for the orbital set, data/aux_basis/ for the fitting set), found by the
  same folded name (lowercased, with `-`, `_` and spaces dropped), and
  converted by tools/atomic_cd/eri_quartet.py's basis_from_bse.
- The coordinates are converted to bohr here, with the tree's CODATA 2018
  Bohr radius, and handed to PySCF in bohr.
- The contraction normalization is free_breakfast's, which never
  renormalizes a contraction, while PySCF rescales every one to exactly 1:
  (mn|P)_fb = D_m D_n D_P (mn|P)_pyscf, D_ii the published contraction's
  norm, on both sets.

The second form reads the C++ tree's hand-built df_high_aux manifest
(test/gtest/data/df_high_aux/df_high_aux.json, copied): one primitive per
shell, its coefficient the primitive's own normaliser, so both sides are
unit-normalised and no D applies. Its centres are already bohr.

The tensor is written as a float64 .npy of shape (naux, nao, nao), element
(P, n, m) = (mn|P): in the Fortran order fb-io-npy reads, P fastest, which
is the order the tree's output row holds. It prints the largest momenta,
the shape and max |(mn|P)|.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from pyscf import df, gto

ROOT = Path(__file__).resolve().parents[7]
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
    in PySCF's order for one atom (the metric's reference.py has the why)."""
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


def manifest(path: Path) -> tuple:
    """The manifest's two hand-built bases on its two centres, D = 1.

    One PySCF atom per shell centre would merge nothing, but PySCF shares a
    basis per symbol, so each centre keeps the manifest's own symbol and its
    shells in the manifest's order (centre-major, l ascending)."""
    doc = json.loads(path.read_text())
    atoms = [(a["symbol"], tuple(a["coord"])) for a in doc["atoms"]]
    mols = []
    for key in ("orbital_shells", "aux_shells"):
        basis = {}
        for shell in doc[key]:
            symbol = atoms[shell["atom"]][0]
            basis.setdefault(symbol, []).append([shell["l"], [shell["exponent"], 1.0]])
        mols.append(gto.M(atom=atoms, basis=basis, unit="Bohr", cart=False, verbose=0))
    orb, aux = mols
    return orb, aux, np.ones(orb.nao), np.ones(aux.nao)


def main() -> None:
    if len(sys.argv) == 4 and sys.argv[1] == "--manifest":
        orb, aux, d, d_aux = manifest(Path(sys.argv[2]))
        out = Path(sys.argv[3])
    elif len(sys.argv) == 5:
        orb, aux, d, d_aux = named(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
        out = Path(sys.argv[4])
    else:
        raise SystemExit(__doc__)
    eri = df.incore.aux_e2(orb, aux, intor="int3c2e", aosym="s1")
    eri = eri * d[:, None, None] * d[None, :, None] * d_aux[None, None, :]
    # (P, n, m) in Fortran order: P fastest, then n, then m.
    np.save(out, np.asfortranarray(eri.transpose(2, 1, 0)))
    lmax = max(int(orb.bas_angular(b)) for b in range(orb.nbas))
    lmax_aux = max(int(aux.bas_angular(b)) for b in range(aux.nbas))
    print(
        f"{out}: (mn|P) {orb.nao} x {orb.nao} x {aux.nao}; orbital l <= {lmax},"
        f" aux l <= {lmax_aux}, max |(mn|P)| = {np.abs(eri).max():.6f}"
    )


if __name__ == "__main__":
    main()
