# The three-center integrals' reference data

PySCF's spherical three-center integrals (mn|P), `int3c2e`, for the two
fixtures `tests/three_center.rs` compares against, so the tests need no
PySCF.

| File | What |
|---|---|
| `h2o.xyz` | Water, in Angstrom: the metric's (`../../metric/tests/data/`), byte for byte |
| `h2o_cc-pvdz_cc-pvdz-jkfit.npy` | (mn\|P) of water in cc-pVDZ / cc-pVDZ-JKFIT, shape (116, 24, 24): orbitals through d, aux through f |
| `df_high_aux.json` | The C++ tree's hand-built fixture, `test/gtest/data/df_high_aux/df_high_aux.json`, byte for byte: one primitive shell per l on two skew centres (bohr), orbitals through f, aux through i, its coefficients the primitives' own normalisers |
| `df_high_aux.npy` | (mn\|P) over it, shape (98, 32, 32): all 70 classes through (ff\|i). Identical, element for element, to the C++ tree's `int3c2e_aux*.npy` pair |
| `reference.py` | The script that wrote both |

Each array is float64, element (P, n, m) = (mn|P) in Fortran order, so P
fastest: the order of the tree's output row, `[m][n][P]`. Both were written
by PySCF 2.14.0 (`uv.lock`).

## What the reference pins, and why

For water, the metric's conventions
(`../../metric/tests/data/README.md`), on both sets, so the comparison
measures the integrals and nothing else: the BSE JSON documents of
`data/basis/` and `data/aux_basis/` that `fb_io_basis` compiles in,
converted with `tools/atomic_cd/eri_quartet.py`'s `basis_from_bse`; the
coordinates converted with CODATA 2018's Bohr radius; and the tree's
contraction normalization, `(mn|P) = D_m D_n D_P (mn|P)_pyscf`, D_ii the
published contraction's norm.

For `df_high_aux`, both sides are unit-normalised by construction, so no D
applies.

## A new reference

From the checkout root (PySCF is in the `tools` dependency group; inside the
devcontainer, set `UV_PROJECT_ENVIRONMENT` to a scratch path first, so the
group does not install into the image's `/opt/venv`):

```bash
D=rust/integrals/eri/density_fitting/three_center/tests/data
OMP_NUM_THREADS=8 uv run --group tools python $D/reference.py \
    $D/h2o.xyz cc-pvdz cc-pvdz-jkfit $D/h2o_cc-pvdz_cc-pvdz-jkfit.npy
OMP_NUM_THREADS=8 uv run --group tools python $D/reference.py \
    --manifest $D/df_high_aux.json $D/df_high_aux.npy
```
