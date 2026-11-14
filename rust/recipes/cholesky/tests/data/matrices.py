"""The symmetric matrices the Rust Cholesky crate's tests factor.

Usage (from the checkout root; numpy is in the `tools` dependency group):

    uv run --group tools python rust/recipes/cholesky/tests/data/matrices.py

writes, beside this script, float64 .npy files from numpy's PCG64 at seed
1506 (the committed files were written by numpy 2.5.4):

- spd_<n>.npy for n in 1, 63, 64, 65, 128 and 200: A = B B^T / n + I, B
  standard normal n x n, so A is symmetric positive definite with every
  eigenvalue at least 1. The orders sit around the 64-column tile: one
  element, a partial tile, exactly one, one and a single column, two, and
  three and a partial one.
- minor_<k>.npy for k in 3 and 67, order 100: A = L D L^T, L unit lower
  triangular with standard normal entries scaled by 1/4 below the diagonal,
  D = I but D[k-1, k-1] = -1. The leading minors are det(A_k) = prod D[:k],
  so minors 1 to k-1 are 1 and minor k is -1: the first that is not
  positive is k, in the first tile for 3 and in the second for 67.

Each matrix is symmetrized as (A + A^T) / 2 before it is saved, so it is
symmetric bit for bit. A itself is the reference: the test's bar is the
backward error |L L^T - A|, so nothing else is computed here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SPD_ORDERS = (1, 63, 64, 65, 128, 200)
MINOR_ORDER = 100
MINORS = (3, 67)


def symmetric(a: np.ndarray) -> np.ndarray:
    a = (a + a.T) / 2
    assert np.array_equal(a, a.T)
    return a


def main() -> None:
    rng = np.random.default_rng(1506)
    for n in SPD_ORDERS:
        b = rng.standard_normal((n, n))
        a = symmetric(b @ b.T / n + np.eye(n))
        np.linalg.cholesky(a)
        np.save(HERE / f"spd_{n}.npy", a)
        print(f"spd_{n}.npy: max |A| {np.abs(a).max():.6e}")
    for k in MINORS:
        lower = np.tril(rng.standard_normal((MINOR_ORDER, MINOR_ORDER)) / 4, -1)
        lower += np.eye(MINOR_ORDER)
        d = np.ones(MINOR_ORDER)
        d[k - 1] = -1.0
        a = symmetric(lower @ np.diag(d) @ lower.T)
        np.linalg.cholesky(a[: k - 1, : k - 1])
        assert np.linalg.det(a[:k, :k]) < 0
        np.save(HERE / f"minor_{k}.npy", a)
        print(f"minor_{k}.npy: det(A_{k}) {np.linalg.det(a[:k, :k]):.6e}")


if __name__ == "__main__":
    main()
