"""The reference of `tests/examples.rs`'s localization example: PySCF's
density-fitted RHF of water in cc-pVDZ over cc-pVDZ-JKFIT, its occupied
orbitals localized by PySCF's Pipek-Mezey over Mulliken populations at
exponent 2, and the sensitivity of that maximum to how far the SCF it starts
from stopped short of convergence.

Usage (from the checkout root; PySCF is in the `tools` dependency group):

    OMP_NUM_THREADS=8 uv run --group tools python \\
        rust/pipeline/config/tests/data/localization_reference.py

It writes `localization_h2o_cc-pvdz_cc-pvdz-jkfit.npy` beside itself, float64
of shape (2,): `[L, sens]`.

- `L`: the localized functional `L_2 = sum_i sum_A (Q^A_ii)^2`. The SCF is
  scf_reference.py's (the same documents, the coordinates in bohr by CODATA
  2018), converged to `conv_tol = 1e-12` and `conv_tol_grad = 1e-10`; the
  localization is the Pipek-Mezey crate's reference.py's `localize`, PySCF's
  kernel at `conv_tol = 1e-12` alternated with its Jacobi stability check
  until no uphill rotation is left. L is invariant under any transformation
  within an atom's AOs, so PySCF's normalization and order need no
  conversion.

- `sens`: the first-order bound on how far the tree's L can sit from `L`
  because its SCF stopped at a residual `r = ||X^T (F D S - S D F) X||_F`
  rather than at zero. Write the occupied space as the reference's rotated
  by `kappa`, an n_vir x n_occ block (`[C_o C_v] expm([[0, -k^T], [k, 0]])`,
  in the canonical basis). The tree's SCF stops when the residual of the
  density it built its last F from, `D(kappa)`, is under the tolerance, and
  publishes the eigenvectors of that F, whose occupied space is
  `kappa' = T kappa`. Its residual is `R kappa`, as a vector of the n x n
  matrix. L at the maximum moves by `g . kappa'` (the envelope theorem: the
  localization's own rotation is stationary there), so
  `|dL| <= ||g^T T R^+|| r = sens r`, R^+ R's pseudo-inverse. g, T and R are
  each formed by central differences at step 1e-4 about the reference, one
  column per element of kappa, every Fock build PySCF's density-fitted one.

It prints both, and n_occ, n_vir and the SCF and localization's counts.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pyscf
import scipy.linalg
from pyscf import scf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
THREE_CENTER = (
    ROOT / "rust/integrals/eri/density_fitting/three_center/tests/data/reference.py"
)
PIPEK_MEZEY = ROOT / "rust/localization/pipek_mezey/tests/data/reference.py"
XYZ = ROOT / "rust/examples/water.xyz"
OUT = HERE / "localization_h2o_cc-pvdz_cc-pvdz-jkfit.npy"
EXPONENT = 2
STEP = 1e-4


def module(name: str, path: Path):
    """The script at `path`, as a module."""
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def rotated(c: np.ndarray, n_occ: int, kappa: np.ndarray) -> np.ndarray:
    """`c`'s columns rotated by the occupied-virtual block `kappa`."""
    n = c.shape[1]
    k = np.zeros((n, n))
    k[n_occ:, :n_occ] = kappa
    k[:n_occ, n_occ:] = -kappa.T
    return c @ scipy.linalg.expm(k)


def main() -> None:
    ref = module("three_center_reference", THREE_CENTER)
    pm = module("pipek_mezey_reference", PIPEK_MEZEY)
    print(f"PySCF {pyscf.__version__}")
    mol, _, _, _ = ref.named(XYZ, "cc-pvdz", "cc-pvdz-jkfit")
    aux_doc = ref.bse_document("cc-pvdz-jkfit", "aux_basis")
    atoms = ref.read_xyz(XYZ)
    auxbasis = {s: ref.basis_from_bse(aux_doc, s) for s in {a for a, _ in atoms}}
    mf = scf.RHF(mol).density_fit(auxbasis=auxbasis)
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-10
    mf.max_cycle = 200
    mf.verbose = 0
    energy = mf.kernel()
    if not mf.converged:
        raise SystemExit("the SCF did not converge: nothing written")

    s = mol.intor_symmetric("int1e_ovlp")
    c = mf.mo_coeff
    n_occ = int((mf.mo_occ > 0).sum())
    n = c.shape[1]
    n_vir = n - n_occ
    c_loc, rounds = pm.localize(mol, c[:, :n_occ], EXPONENT)
    # The localization as a rotation of the canonical occupied orbitals.
    u_loc = c[:, :n_occ].T @ s @ c_loc
    functional = float((pm.populations(mol, s, c_loc) ** EXPONENT).sum())

    w, v = np.linalg.eigh(s)
    x = v @ np.diag(w**-0.5) @ v.T

    def residual_and_occupied(kappa: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The residual matrix of D(kappa) and its F's own occupied space,
        as the tree's loop publishes it, the rotation from the reference's."""
        occ = rotated(c, n_occ, kappa)[:, :n_occ]
        d = 2 * occ @ occ.T
        f = mf.get_fock(dm=d)
        e = x.T @ (f @ d @ s - s @ d @ f) @ x
        _, new = scipy.linalg.eigh(f, s)
        p = new[:, :n_occ] @ new[:, :n_occ].T
        return e.ravel(), c[:, n_occ:].T @ s @ p @ s @ c[:, :n_occ]

    def localized(kappa: np.ndarray) -> float:
        """L of the rotated occupied space at the reference's localization."""
        occ = rotated(c, n_occ, kappa)[:, :n_occ] @ u_loc
        return float((pm.populations(mol, s, occ) ** EXPONENT).sum())

    g = np.zeros(n_vir * n_occ)
    t = np.zeros((n_vir * n_occ, n_vir * n_occ))
    r = np.zeros((n * n, n_vir * n_occ))
    for k in range(n_vir * n_occ):
        kappa = np.zeros(n_vir * n_occ)
        kappa[k] = STEP
        kappa = kappa.reshape(n_vir, n_occ)
        plus, minus = residual_and_occupied(kappa), residual_and_occupied(-kappa)
        r[:, k] = (plus[0] - minus[0]) / (2 * STEP)
        t[:, k] = (plus[1] - minus[1]).ravel() / (2 * STEP)
        g[k] = (localized(kappa) - localized(-kappa)) / (2 * STEP)
    sens = float(np.linalg.norm(g @ t @ np.linalg.pinv(r)))
    sigma = np.linalg.svd(r, compute_uv=False)[-1]

    print(
        f"n_occ {n_occ}, n_vir {n_vir}: E = {energy:.12f}; L_{EXPONENT} = "
        f"{functional:.15f} in {rounds} round(s); |g| = {np.linalg.norm(g):.3e}, "
        f"sigma_min(R) = {sigma:.3e}, sens = {sens:.3e}"
    )
    np.save(OUT, np.array([functional, sens]))


if __name__ == "__main__":
    main()
