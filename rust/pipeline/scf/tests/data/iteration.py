"""The fixtures of one closed-shell SCF iteration's dense nodes, for the Rust
pipeline/scf crate's `tests/iteration.rs`: water in cc-pVDZ, PySCF's
density-fitted RHF over cc-pVDZ-JKFIT, at two points.

Usage (from the checkout root; PySCF is in the `tools` dependency group):

    OMP_NUM_THREADS=8 uv run --group tools python \\
        rust/pipeline/scf/tests/data/iteration.py

It writes beside this script, every array float64 in Fortran order:

- `iteration_h.npy`, `iteration_s.npy`, `iteration_x.npy`: H, S and
  X = S^(-1/2) (symmetric, from S's eigendecomposition, no mode dropped),
  n x n, in the tree's spherical AO order and normalization.
- `iteration_<point>_c.npy`: the orbitals C, n x n, ascending.
- `iteration_<point>_d.npy`: D = 2 C_occ C_occ^T over C's first nocc = 5.
- `iteration_<point>_j.npy`, `..._k.npy`: PySCF's DF J and K of D.
- `iteration_<point>_scalars.npy`: [E, ||e||_F, E_nuc], with
  F = H + J - K/2, E = tr(D (H + F))/2 + E_nuc and
  e = X^T (F D S - S D F) X, all evaluated here in float64.

The points are `start`, the core Hamiltonian's orbitals (the residual far
from zero), and `converged`, PySCF's converged orbitals at conv_tol 1e-12
and conv_tol_grad 1e-9, its J and K rebuilt from their D. The conventions
are the ERI step's reference.py (`rust/pipeline/eri/tests/data/`), through
the three-center crate's helpers: the BSE JSON documents of data/basis/ and
data/aux_basis/, the coordinates in bohr by CODATA 2018, and the tree's
contraction normalization, the tree's functions D_i times PySCF's: so
H, S, J and K are N M N of PySCF's, D is N^-1 D N^-1 and C is N^-1 C, and
X is the tree's S^(-1/2). The geometry is the ERI step's `h2o.xyz`. It
prints, per point, E, ||e||_F and the norms the test's bars are built from.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
from pyscf import scf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
XYZ = ROOT / "rust/pipeline/eri/tests/data/h2o.xyz"
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


def save(name: str, a: np.ndarray) -> None:
    np.save(HERE / f"iteration_{name}.npy", np.asfortranarray(a))


def main() -> None:
    ref = three_center_reference()
    mol, _, norms, _ = ref.named(XYZ, "cc-pvdz", "cc-pvdz-jkfit")
    aux_doc = ref.bse_document("cc-pvdz-jkfit", "aux_basis")
    symbols = {a for a, _ in ref.read_xyz(XYZ)}
    auxbasis = {sym: ref.basis_from_bse(aux_doc, sym) for sym in symbols}
    nocc = mol.nelectron // 2
    mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    mf.kernel()
    if not mf.converged:
        raise SystemExit("PySCF did not converge")

    # The tree's normalization: M -> N M N for an operator, N^-1 on an
    # orbital's coefficients.
    def tree(m: np.ndarray) -> np.ndarray:
        return norms[:, None] * m * norms[None, :]

    h = tree(mf.get_hcore())
    s = tree(mf.get_ovlp())
    w, v = np.linalg.eigh(s)
    x = (v / np.sqrt(w)) @ v.T
    e_nuc = mol.energy_nuc()
    for name, a in [("h", h), ("s", s), ("x", x)]:
        save(name, a)

    _, c_core = np.linalg.eigh(x @ h @ x)
    points = {"start": x @ c_core, "converged": mf.mo_coeff / norms[:, None]}
    for point, c in points.items():
        c_occ = c[:, :nocc]
        d = 2.0 * c_occ @ c_occ.T
        dm = tree(d)  # PySCF's density is N D N
        j, k = (tree(m) for m in mf.get_jk(mol, dm, hermi=1))
        f = h + j - 0.5 * k
        energy = 0.5 * np.sum(d * (h + f)) + e_nuc
        fds = f @ d @ s
        e = x.T @ (fds - fds.T) @ x
        norm = np.linalg.norm(e)
        for name, a in [("c", c), ("d", d), ("j", j), ("k", k)]:
            save(f"{point}_{name}", a)
        save(f"{point}_scalars", np.array([energy, norm, e_nuc]))
        print(
            f"{point}: n {mol.nao}, nocc {nocc}, E {energy:.12f}"
            f" (PySCF e_tot {mf.e_tot:.12f}), ||e||_F {norm:.3e},"
            f" ||F||_F {np.linalg.norm(f):.6f}, ||D||_F {np.linalg.norm(d):.6f},"
            f" ||S||_F {np.linalg.norm(s):.6f}, ||X||_2 {1 / np.sqrt(w[0]):.6f}"
        )


if __name__ == "__main__":
    main()
