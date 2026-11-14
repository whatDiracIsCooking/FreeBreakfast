"""The metrics and the tensor the Rust density-fit crate's tests run on.

Usage (from the checkout root; numpy is in the `tools` dependency group):

    uv run --group tools python \
        rust/integrals/eri/density_fitting/fit/tests/data/inputs.py

writes, beside this script, float64 .npy files from numpy's PCG64 at seed
1514 (the committed files were written by numpy 2.5.4):

- m41.npy: the metric, order 41, M = Q diag(lam) Q^T with Q the orthogonal
  factor of a standard normal 41 x 41 and lam log-spaced from 1e-3 to 10,
  so cond(M) is 1e4, about what a fitting set's metric carries per decade
  of exponents, and the condition number enters the test's bar.
- sigma_l41.npy: the singular values of numpy's Cholesky factor of m41,
  descending: the test's kappa_2(L) is their ratio, sigma[0] / sigma[-1].
- b41x25.npy: the tensor (P|mn), naux = 41 by nao^2 = 5^2, standard normal,
  stored 41 x 25 so that the test's column-major read is the tensor's
  [m][n][P] with P fastest. It is not symmetric in m and n: the fit is a
  solve over P and never looks at the pair index.
- minor_17.npy: order 41, M = L D L^T with L unit lower triangular, its
  entries standard normal scaled by 1/4 below the diagonal, and D = I but
  D[16, 16] = -1, so the leading minors 1 to 16 are 1 and minor 17 is -1:
  the first that is not positive is 17.

Each metric is symmetrized as (M + M^T) / 2 before it is saved, so it is
symmetric bit for bit. M and b are themselves the reference: the test's bar
is the solve's backward error |L B - b|, so nothing else is computed here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NAUX = 41
NAO = 5
MINOR = 17


def symmetric(a: np.ndarray) -> np.ndarray:
    a = (a + a.T) / 2
    assert np.array_equal(a, a.T)
    return a


def main() -> None:
    rng = np.random.default_rng(1514)
    q, _ = np.linalg.qr(rng.standard_normal((NAUX, NAUX)))
    lam = np.logspace(-3, 1, NAUX)
    m = symmetric(q @ np.diag(lam) @ q.T)
    sigma = np.linalg.svd(np.linalg.cholesky(m), compute_uv=False)
    np.save(HERE / "m41.npy", m)
    np.save(HERE / "sigma_l41.npy", sigma)
    print(
        f"m41.npy: max |M| {np.abs(m).max():.6e}, kappa_2(L) {sigma[0] / sigma[-1]:.6e}"
    )

    b = rng.standard_normal((NAUX, NAO * NAO))
    np.save(HERE / "b41x25.npy", b)
    print(f"b41x25.npy: max |b| {np.abs(b).max():.6e}")

    lower = np.tril(rng.standard_normal((NAUX, NAUX)) / 4, -1) + np.eye(NAUX)
    d = np.ones(NAUX)
    d[MINOR - 1] = -1.0
    a = symmetric(lower @ np.diag(d) @ lower.T)
    np.linalg.cholesky(a[: MINOR - 1, : MINOR - 1])
    assert np.linalg.det(a[:MINOR, :MINOR]) < 0
    np.save(HERE / f"minor_{MINOR}.npy", a)
    print(f"minor_{MINOR}.npy: det(M_{MINOR}) {np.linalg.det(a[:MINOR, :MINOR]):.6e}")


if __name__ == "__main__":
    main()
