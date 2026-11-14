"""The reference orbital energies of the core Hamiltonian for the Rust
pipeline/scf crate's tests: LAPACK's generalized symmetric eigenvalues,
through scipy.linalg.eigh(H, S), of each fixture's PySCF S and H committed in
rust/integrals/one_electron/overlap/tests/data/ and
rust/integrals/one_electron/core_hamiltonian/tests/data/, under the
conventions those directories' README.md pin (the tree's basis digits, its
CODATA 2018 bohr, its unrenormalized contractions).

Usage (from the checkout root; numpy and scipy are in the `tools` dependency
group, scipy through PySCF):

    uv run --group tools python rust/pipeline/scf/tests/data/reference.py

writes beside this script, for each fixture:

- <stem>_epsilon.npy: float64, the n orbital energies in ascending order;
- <stem>_scales.npy: float64, [kappa_2(S), ||H||_2], S's condition number
  (lambda_max / lambda_min of its eigenvalues) and H's spectral norm (its
  largest eigenvalue in magnitude), which the test's bars are built from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.linalg

HERE = Path(__file__).resolve().parent
ONE_ELECTRON = HERE.parents[3] / "integrals" / "one_electron"
OVERLAP = ONE_ELECTRON / "overlap" / "tests" / "data"
CORE_HAMILTONIAN = ONE_ELECTRON / "core_hamiltonian" / "tests" / "data"
STEMS = ("h2o_sto-3g", "h2o_cc-pvdz", "co_cc-pvtz")


def main() -> None:
    for stem in STEMS:
        s = np.load(OVERLAP / f"{stem}.npy")
        h = np.load(CORE_HAMILTONIAN / f"{stem}.npy")
        eps = scipy.linalg.eigh(h, s, eigvals_only=True)
        w = np.linalg.eigvalsh(s)
        kappa = w[-1] / w[0]
        norm_h = np.abs(np.linalg.eigvalsh(h)).max()
        np.save(HERE / f"{stem}_epsilon.npy", eps)
        np.save(HERE / f"{stem}_scales.npy", np.array([kappa, norm_h]))
        print(
            f"{stem}: n {len(eps)}, eps in [{eps[0]:.6e}, {eps[-1]:.6e}],"
            f" kappa_2(S) {kappa:.6e}, ||H||_2 {norm_h:.6e}"
        )


if __name__ == "__main__":
    main()
