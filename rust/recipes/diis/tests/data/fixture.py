"""The vectors and the linear map the Rust DIIS crate's tests run on.

Usage (from the checkout root; numpy is in the `tools` dependency group):

    uv run --group tools python rust/recipes/diis/tests/data/fixture.py

writes beside this script, all float64 from numpy's PCG64 at seed 1507:

- iterates.npy, residuals.npy: 11 x 37 each, uniform on [-1, 1), row k the
  k-th push's iterate and residual. The tests push them in order; nothing
  else is a reference, the expectations are solved on the host from these
  values.
- linear_a.npy: 6 x 6, A = Q diag(0.9, -0.8, 0.5, 0.3, -0.2, 0.7) Q^T with Q a
  Householder reflection, so its spectral radius is exactly 0.9 and the plain
  iteration x <- A x + b converges slowly; linear_b.npy: b, 6 entries,
  uniform on [-1, 1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PUSHES = 11
LEN = 37
LAMBDA = np.array([0.9, -0.8, 0.5, 0.3, -0.2, 0.7])


def main() -> None:
    rng = np.random.default_rng(1507)
    iterates = rng.uniform(-1.0, 1.0, (PUSHES, LEN))
    residuals = rng.uniform(-1.0, 1.0, (PUSHES, LEN))
    v = rng.uniform(-1.0, 1.0, LAMBDA.size)
    q = np.eye(LAMBDA.size) - 2.0 * np.outer(v, v) / (v @ v)
    a = q @ np.diag(LAMBDA) @ q.T
    b = rng.uniform(-1.0, 1.0, LAMBDA.size)
    np.save(HERE / "iterates.npy", iterates)
    np.save(HERE / "residuals.npy", residuals)
    np.save(HERE / "linear_a.npy", a)
    np.save(HERE / "linear_b.npy", b)
    print(f"spectral radius of A: {np.abs(np.linalg.eigvals(a)).max():.15f}")


if __name__ == "__main__":
    main()
