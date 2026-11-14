"""PySCF's density-fitted J and K of the SAD guess density, for the Rust
ERI step's test: `scf.RHF(mol).density_fit(auxbasis=...).get_jk` over the
same orbital and fitting documents the tree compiles in.

Usage (from the checkout root; PySCF is in the `tools` dependency group):

    D=rust/pipeline/eri/tests/data
    OMP_NUM_THREADS=8 uv run --group tools python $D/reference.py \\
        $D/h2o.xyz cc-pvdz cc-pvdz-jkfit $D

It writes four float64 .npy files into the directory, each in Fortran
order, AOs in the tree's spherical order and normalization:

- `h2o_cc-pvdz_d.npy`: the SAD guess D, nao x nao: the compiled-in stamps
  `data/sad/<basis>/<Z>.npy` on the diagonal blocks of the atoms in XYZ
  order, which is what the Guess step assembles.
- `h2o_cc-pvdz_c.npy`: its factor C, nao x nocc, D = 2 C C^T: D's
  eigenvectors scaled by sqrt(w / 2) over the eigenvalues w above 1e-12
  of the largest (D is a sum of atomic densities, positive semidefinite).
- `h2o_cc-pvdz_cc-pvdz-jkfit_j.npy` and `..._k.npy`: J and K of D.

The conventions are the three-center crate's reference.py, whose helpers
this script imports: the BSE JSON documents of data/basis/ and
data/aux_basis/, the coordinates in bohr by CODATA 2018, and the tree's
contraction normalization. The tree's functions are D_i times PySCF's
unit-normalised ones, so D_pyscf = N D N in, and J = N J_pyscf N,
K = N K_pyscf N out; the fitting set's norms cancel in the fit. It prints
the shapes, max |2 C C^T - D|, max |J| and max |K|.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from pyscf import scf

ROOT = Path(__file__).resolve().parents[5]
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
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    xyz, orbital, aux_spec, out = (
        Path(sys.argv[1]),
        sys.argv[2],
        sys.argv[3],
        Path(sys.argv[4]),
    )
    ref = three_center_reference()
    orb, _, norms, _ = ref.named(xyz, orbital, aux_spec)
    atoms = ref.read_xyz(xyz)
    doc = ref.bse_document(orbital, "basis")
    aux_doc = ref.bse_document(aux_spec, "aux_basis")
    auxbasis = {s: ref.basis_from_bse(aux_doc, s) for s in {a for a, _ in atoms}}

    d = sad_density(atoms, doc["name"])
    if d.shape != (orb.nao, orb.nao):
        raise SystemExit(f"D is {d.shape}, the basis has {orb.nao} functions")
    w, v = np.linalg.eigh(d)
    keep = w > 1e-12 * w.max()
    c = v[:, keep] * np.sqrt(w[keep] / 2.0)
    factor_miss = np.abs(2.0 * c @ c.T - d).max()

    mf = scf.RHF(orb).density_fit(auxbasis=auxbasis)
    dm = norms[:, None] * d * norms[None, :]
    vj, vk = mf.get_jk(orb, dm, hermi=1)
    j = norms[:, None] * vj * norms[None, :]
    k = norms[:, None] * vk * norms[None, :]

    stem = f"{xyz.stem}_{orbital}"
    for name, a in [
        (f"{stem}_d.npy", d),
        (f"{stem}_c.npy", c),
        (f"{stem}_{aux_spec}_j.npy", j),
        (f"{stem}_{aux_spec}_k.npy", k),
    ]:
        np.save(out / name, np.asfortranarray(a))
    print(
        f"nao {orb.nao}, naux {mf.with_df.get_naoaux()}, nocc {c.shape[1]}:"
        f" max |2 C C^T - D| = {factor_miss:.3e},"
        f" max |J| = {np.abs(j).max():.6f}, max |K| = {np.abs(k).max():.6f}"
    )


if __name__ == "__main__":
    main()
