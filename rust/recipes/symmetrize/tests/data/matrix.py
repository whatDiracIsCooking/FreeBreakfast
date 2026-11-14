"""The nonsymmetric 37 x 37 the Rust symmetrize crate's tests run on.

Usage (from the checkout root; numpy is in the `tools` dependency group):

    uv run --group tools python rust/recipes/symmetrize/tests/data/matrix.py

writes a37.npy beside this script: float64, 37 x 37, standard normal from
numpy's PCG64 at seed 1508, so no element equals its mirror. 37 is one full
32-wide tile and a partial one, so the kernels' diagonal tiles, their one
off-diagonal pair and the edge guards all run. The test's expectations are
computed from these values on the host; nothing else is a reference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
N = 37


def main() -> None:
    a = np.random.default_rng(1508).standard_normal((N, N))
    off_diagonal = ~np.eye(N, dtype=bool)
    assert not np.any((a == a.T) & off_diagonal), "an element equals its mirror"
    np.save(HERE / "a37.npy", a)
    print(f"a37.npy: {N} x {N}, max |A - A^T| {np.abs(a - a.T).max():.6e}")


if __name__ == "__main__":
    main()
