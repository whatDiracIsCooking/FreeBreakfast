"""PySCF's fitted tensor B = L^-1 (P|mn), for fb-steps' chain test:
`molecule.py`'s `named` over an orbital set and a fitting set, so the
molecule, both BSE documents, the CODATA 2018 bohr and the contraction
norms D are the ones free_breakfast reads.

Usage (from the checkout root; PySCF is in the `tools` dependency group):

    D=rust/steps/tests/data
    OMP_NUM_THREADS=8 uv run --group tools python $D/b_reference.py \\
        $D/h2o.xyz cc-pvdz cc-pvdz-jkfit $D/b_h2o_cc-pvdz_cc-pvdz-jkfit.npy

L is the lower Cholesky factor of the metric (P|Q), unique for a positive
definite metric, as both the tree and PySCF's `df.incore.cholesky_eri` take
it. In free_breakfast's normalization, which never renormalizes a
contraction, (mn|P)_fb = D_m D_n D_P (mn|P)_pyscf and (P|Q)_fb = D_P D_Q
(P|Q)_pyscf, so L_fb = D L_pyscf and B_fb = D_m D_n B_pyscf: the fitting
set's norms cancel, and they are applied anyway, so the script states the
tree's tensor and nothing is left to the test.

It writes one float64 .npy of shape (naux, nao (nao + 1) / 2), Fortran
order, so the aux index is fastest: element (Q, p) is B[Q][m n] at the
packed lower-triangle index p = m (m + 1) / 2 + n, n <= m. It prints the
shape and max |B|.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import scipy.linalg
from pyscf import df

HERE = Path(__file__).resolve().parent
MOLECULE = HERE / "molecule.py"


def molecule_reference():
    """`molecule.py`, beside this script, as a module."""
    spec = importlib.util.spec_from_file_location("molecule_reference", MOLECULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if len(sys.argv) != 5:
        raise SystemExit(__doc__)
    ref = molecule_reference()
    orb, aux, d, d_aux = ref.named(Path(sys.argv[1]), sys.argv[2], sys.argv[3])
    j3c = df.incore.aux_e2(orb, aux, intor="int3c2e", aosym="s1")
    j3c = j3c * d[:, None, None] * d[None, :, None] * d_aux[None, None, :]
    j2c = aux.intor("int2c2e") * d_aux[:, None] * d_aux[None, :]
    lower = scipy.linalg.cholesky(j2c, lower=True)
    nao, naux = orb.nao, aux.nao
    b = scipy.linalg.solve_triangular(lower, j3c.reshape(nao * nao, naux).T, lower=True)
    b = b.reshape(naux, nao, nao)
    m, n = np.tril_indices(nao)
    packed = b[:, m, n]
    np.save(Path(sys.argv[4]), np.asfortranarray(packed))
    print(f"{sys.argv[4]}: B {packed.shape}, max |B| = {np.abs(packed).max():.6f}")


if __name__ == "__main__":
    main()
