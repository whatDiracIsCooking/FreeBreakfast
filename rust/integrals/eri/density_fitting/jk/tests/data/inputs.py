"""The factor, density and orbitals the Rust DF J/K crate's tests run on.

Usage (from the checkout root; numpy is in the `tools` dependency group):

    uv run --group tools python \
        rust/integrals/eri/density_fitting/jk/tests/data/inputs.py

writes, beside this script, float64 .npy files from numpy's PCG64 at seed
1515 (the committed files were written by numpy 2.5.4):

- b23x49.npy: the fitted factor B, naux = 23 by nao^2 = 7^2, standard
  normal, each B_Q symmetrized as (B_Q + B_Q^T) / 2 so that it is symmetric
  in m and n bit for bit, as (P|mn) and so B are. Stored 23 x 49 so that the
  test's column-major read is B's [m][n][Q], Q fastest: element
  (Q, n + nao m).
- c7x3.npy: an occupied factor C, nao = 7 by nocc = 3, standard normal.
- d7.npy: the density D = 2 C C^T, symmetrized as (D + D^T) / 2 so that it
  is symmetric bit for bit, J's input as K's C is.

Nothing else is computed here: the test's references are the host's
B^T (B vec(D)) and sum_Q B_Q D B_Q, each held to the products' rounding.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
NAUX = 23
NAO = 7
NOCC = 3


def symmetric(a: np.ndarray) -> np.ndarray:
    a = (a + np.swapaxes(a, -1, -2)) / 2
    assert np.array_equal(a, np.swapaxes(a, -1, -2))
    return a


def main() -> None:
    rng = np.random.default_rng(1515)
    b = symmetric(rng.standard_normal((NAUX, NAO, NAO)))
    # Row Q, column m * nao + n: the column-major read's n + nao m.
    np.save(HERE / f"b{NAUX}x{NAO * NAO}.npy", b.reshape(NAUX, NAO * NAO))
    print(f"b{NAUX}x{NAO * NAO}.npy: max |B| {np.abs(b).max():.6e}")

    c = rng.standard_normal((NAO, NOCC))
    d = symmetric(2 * c @ c.T)
    np.save(HERE / f"c{NAO}x{NOCC}.npy", c)
    np.save(HERE / f"d{NAO}.npy", d)
    print(f"c{NAO}x{NOCC}.npy, d{NAO}.npy: max |D| {np.abs(d).max():.6e}")


if __name__ == "__main__":
    main()
