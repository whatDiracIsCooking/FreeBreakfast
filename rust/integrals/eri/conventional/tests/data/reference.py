"""PySCF's exact (not density-fitted) J and K of the SAD guess density, for
the Rust conventional-ERI crate's test: `scf.RHF(mol).get_jk` over the same
orbital documents the tree compiles in.

Usage (from the checkout root; PySCF is in the `tools` dependency group):

    D=rust/integrals/eri/conventional/tests/data
    for xyz in h2 h2o; do for basis in sto-3g cc-pvdz; do
        OMP_NUM_THREADS=8 uv run --group tools python $D/reference.py \\
            $D/$xyz.xyz $basis $D
    done; done

It writes three float64 .npy files into the directory, each nao x nao in
Fortran order, AOs in the tree's spherical order and normalization:

- `<xyz>_<basis>_d.npy`: the SAD guess D: the compiled-in stamps
  `data/sad/<basis>/<Z>.npy` on the diagonal blocks of the atoms in XYZ
  order, which is what the Guess step assembles.
- `<xyz>_<basis>_j.npy` and `..._k.npy`: J and K of D, `J_mn = sum_ls
  (mn|ls) D_ls` and `K_mn = sum_ls (ml|ns) D_ls`, by libcint's integrals
  with no fit.

The conventions are the three-center crate's reference.py, whose helpers
this script imports: the BSE JSON documents of data/basis/, the
coordinates in bohr by CODATA 2018, and the tree's contraction
normalization. The tree's functions are N_i times PySCF's unit-normalised
ones, so D_pyscf = N D N in, and J = N J_pyscf N, K = N K_pyscf N out. It
prints the shapes, max |J| and max |K|.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from pyscf import gto, scf

ROOT = Path(__file__).resolve().parents[6]
THREE_CENTER = (
    ROOT / "rust/integrals/eri/density_fitting/three_center/tests/data/reference.py"
)


def three_center_reference():
    """The three-center crate's reference.py, as a module."""
    spec = importlib.util.spec_from_file_location(
        "three_center_reference", THREE_CENTER
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sad_density(atoms: list, basis_name: str) -> np.ndarray:
    """The compiled-in stamps on the atoms' diagonal blocks."""
    from pyscf.data.elements import charge

    blocks = [
        np.load(ROOT / "data/sad" / basis_name / f"{charge(symbol):03d}.npy")
        for symbol, _ in atoms
    ]
    n = sum(b.shape[0] for b in blocks)
    d = np.zeros((n, n))
    at = 0
    for b in blocks:
        k = b.shape[0]
        d[at : at + k, at : at + k] = b
        at += k
    return d


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    xyz, orbital, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    ref = three_center_reference()
    atoms = ref.read_xyz(xyz)
    doc = ref.bse_document(orbital, "basis")
    symbols = sorted({symbol for symbol, _ in atoms})
    basis = {s: ref.basis_from_bse(doc, s) for s in symbols}
    mol = gto.M(atom=atoms, basis=basis, unit="Bohr", cart=False, verbose=0)
    norms = np.array(
        [n for symbol, _ in atoms for n in ref.contraction_norms(basis[symbol])]
    )
    if norms.size != mol.nao:
        raise SystemExit(f"{norms.size} contraction norms for {mol.nao} functions")

    d = sad_density(atoms, doc["name"])
    if d.shape != (mol.nao, mol.nao):
        raise SystemExit(f"D is {d.shape}, the basis has {mol.nao} functions")
    dm = norms[:, None] * d * norms[None, :]
    vj, vk = scf.RHF(mol).get_jk(mol, dm, hermi=1)
    j = norms[:, None] * vj * norms[None, :]
    k = norms[:, None] * vk * norms[None, :]

    stem = f"{xyz.stem}_{orbital}"
    for name, a in [(f"{stem}_d.npy", d), (f"{stem}_j.npy", j), (f"{stem}_k.npy", k)]:
        np.save(out / name, np.asfortranarray(a))
    print(
        f"{stem}: nao {mol.nao} (cartesian {mol.nao_cart()}),"
        f" max |J| = {np.abs(j).max():.6f}, max |K| = {np.abs(k).max():.6f}"
    )


if __name__ == "__main__":
    main()
