"""libcint's Cartesian Schwarz diagonal, sqrt((mn|mn)) from PySCF's
`int2e_cart`, for the Rust Schwarz crate's tests.

Usage (from the checkout root; PySCF is in the `tools` dependency group):

    D=rust/integrals/schwarz/tests/data
    OMP_NUM_THREADS=12 uv run --group tools python $D/reference.py \\
        $D/h2o.xyz cc-pvtz $D/h2o_cc-pvtz.npy

It pins three things to what free_breakfast itself reads, so the
comparison measures the integrals and nothing else:

- The set is the BSE JSON document the binary compiles in (data/basis/),
  found by resolve_basis's folded name (lowercased, with `-`, `_` and
  spaces dropped), and converted by tools/atomic_cd/eri_quartet.py's
  basis_from_bse.
- The coordinates are converted to bohr here, with the tree's CODATA 2018
  Bohr radius, and handed to PySCF in bohr.
- The normalisation is the tree's, reached through the overlap: a tree
  Cartesian function is its shell's coefficients, each the published one
  times the (l,0,0) primitive's normaliser
  (ShellStamp::primitive_normalized), on every component, the contraction
  never renormalised; PySCF's differs from it by one positive factor per
  function, lambda_m = sqrt(S_fb[m,m] / S_pyscf[m,m]), S_fb the tree's
  Cartesian overlap diagonal in closed form and S_pyscf `int1e_ovlp_cart`'s.
  So sqrt((mn|mn))_fb = lambda_m lambda_n sqrt((mn|mn))_pyscf. The script
  also checks that lambda is one value across each shell's components,
  which is the statement that both codes give every component the same
  coefficients.

The array is written as a float64 .npy of shape (nao_cart, nao_cart),
element (m, n) = sqrt((mn|mn)) over the tree's Cartesian AOs (symmetric,
so the order of its two indices does not matter). It prints the shape,
the largest momentum and the extreme elements.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
from pyscf import gto

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from tools.atomic_cd.eri_quartet import basis_from_bse  # noqa: E402

# kAngstromPerBohr, src/core/constants/constants.h (CODATA 2018).
ANGSTROM_PER_BOHR = 0.529177210903


def fold(name: str) -> str:
    """resolve_basis's folding: lowercased, `-`, `_` and spaces dropped."""
    return "".join(c for c in name.lower() if c not in "-_ ")


def bse_document(spec: str) -> dict:
    """The carried set `spec` names in data/basis/."""
    for path in sorted((ROOT / "data" / "basis").glob("*.json")):
        doc = json.loads(path.read_text())
        if fold(doc.get("name", "")) == fold(spec) or fold(path.stem) == fold(spec):
            return doc
    raise SystemExit(f"{spec!r} is not a carried set of data/basis/")


def read_xyz(path: Path) -> list[tuple[str, tuple[float, float, float]]]:
    """The atoms of an XYZ file, positions in bohr."""
    lines = path.read_text().splitlines()
    count = int(lines[0])
    atoms = []
    for line in lines[2 : 2 + count]:
        symbol, *xyz = line.split()[:4]
        atoms.append((symbol, tuple(float(v) / ANGSTROM_PER_BOHR for v in xyz)))
    return atoms


def odd_double_factorial(n: int) -> float:
    """n!! for odd n >= -1, (-1)!! = 1."""
    return float(math.prod(range(n, 0, -2))) if n > 0 else 1.0


def cartesian_powers(ell: int) -> list[tuple[int, int, int]]:
    """fb_basis::cartesian_powers: x's power highest, then y's (PySCF's
    Cartesian order too)."""
    return [
        (a, b, ell - a - b) for a in range(ell, -1, -1) for b in range(ell - a, -1, -1)
    ]


def tree_overlap_diagonal(shells: list) -> list[float]:
    """S_fb[m,m] for one atom's functions, in PySCF's order (contraction
    columns outer, components inner), in closed form:
    sum_ij c_i c_j prod_axis (2k-1)!! / (2p)^k (pi/p)^(3/2), p = a_i + a_j,
    c the published coefficient times the (l,0,0) normaliser
    (2a/pi)^(3/4) (4a)^(l/2) / sqrt((2l-1)!!)."""
    out = []
    for ell, *rows in shells:
        exps = np.array([row[0] for row in rows])
        norm = (
            (2 * exps / np.pi) ** 0.75
            * (4 * exps) ** (ell / 2)
            / math.sqrt(odd_double_factorial(2 * ell - 1))
        )
        p = exps[:, None] + exps[None, :]
        for col in range(1, len(rows[0])):
            c = np.array([row[col] for row in rows]) * norm
            for powers in cartesian_powers(ell):
                angular = math.prod(odd_double_factorial(2 * k - 1) for k in powers)
                radial = (np.pi / p) ** 1.5 / (2 * p) ** ell
                out.append(float(c @ (angular * radial) @ c))
    return out


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    xyz, spec, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    atoms = read_xyz(xyz)
    doc = bse_document(spec)
    basis = {s: basis_from_bse(doc, s) for s in sorted({s for s, _ in atoms})}
    mol = gto.M(atom=atoms, basis=basis, unit="Bohr", cart=True, verbose=0)

    s_fb = np.array(
        [v for symbol, _ in atoms for v in tree_overlap_diagonal(basis[symbol])]
    )
    if s_fb.size != mol.nao:
        raise SystemExit(f"{s_fb.size} overlap diagonals for {mol.nao} functions")
    lam = np.sqrt(s_fb / np.diag(mol.intor("int1e_ovlp_cart")))
    for b in range(mol.nbas):
        lo, hi = mol.ao_loc_nr()[b], mol.ao_loc_nr()[b + 1]
        ncart = (mol.bas_angular(b) + 1) * (mol.bas_angular(b) + 2) // 2
        for k in range(lo, hi, ncart):
            block = lam[k : k + ncart]
            if np.abs(block / block[0] - 1).max() > 1e-12:
                raise SystemExit(
                    f"shell {b}: lambda is not one value across its components"
                )

    n = mol.nao
    eri = mol.intor("int2e_cart", aosym="s1").reshape(n, n, n, n)
    diag = np.einsum("mnmn->mn", eri)
    if diag.min() < 0:
        raise SystemExit(f"a negative (mn|mn): {diag.min()}")
    schwarz = np.sqrt(diag) * lam[:, None] * lam[None, :]
    np.save(out, np.asfortranarray(schwarz))
    lmax = max(int(mol.bas_angular(b)) for b in range(mol.nbas))
    print(
        f"{out}: sqrt((mn|mn)) {n} x {n}, l <= {lmax}, min {schwarz.min():.6e},"
        f" max {schwarz.max():.6e}"
    )


if __name__ == "__main__":
    main()
