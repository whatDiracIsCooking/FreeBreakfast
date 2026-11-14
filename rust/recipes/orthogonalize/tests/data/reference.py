"""The reference eigenvalues of the overlap fixtures' S for the Rust
orthogonalize crate's tests: LAPACK's, through numpy.linalg.eigvalsh, of each
S committed in rust/integrals/one_electron/overlap/tests/data/.

Usage (from the checkout root; numpy is in the `tools` dependency group):

    uv run --group tools python rust/recipes/orthogonalize/tests/data/reference.py

writes <stem>_eigenvalues.npy beside this script for each fixture: float64,
the n eigenvalues in ascending order. The test reads κ₂(S) = λ_max / λ_min
from them, and checks that none is at or below the orthogonalizer's floor.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OVERLAP = HERE.parents[3] / "integrals" / "one_electron" / "overlap" / "tests" / "data"
STEMS = ("h2o_sto-3g", "h2o_cc-pvdz", "co_cc-pvtz")


def main() -> None:
    for stem in STEMS:
        s = np.load(OVERLAP / f"{stem}.npy")
        w = np.linalg.eigvalsh(s)
        np.save(HERE / f"{stem}_eigenvalues.npy", w)
        cond = w[-1] / w[0]
        print(f"{stem}: n {len(w)}, w in [{w[0]:.6e}, {w[-1]:.6e}], cond {cond:.6e}")


if __name__ == "__main__":
    main()
