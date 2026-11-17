# The Schwarz diagonal's reference data

libcint's Cartesian `sqrt((mn|mn))`, from PySCF's `int2e_cart`, for the
fixture `tests/libcint.rs` compares against, so the test needs no PySCF.

| File | What |
|---|---|
| `h2o.xyz` | Water, in Angstrom: the three-center integrals' (`../../eri/density_fitting/three_center/tests/data/`), byte for byte; `tests/projection.rs` reads it too |
| `h2o_cc-pvtz.npy` | `sqrt((mn\|mn))` of water in cc-pVTZ over the tree's Cartesian AOs, shape (65, 65): O through f, H through d, so every class through (ff\|ff) |
| `reference.py` | The script that wrote it |

The array is float64, element (m, n) = `sqrt((mn|mn))`, symmetric. It was
written by PySCF 2.14.0 (`uv.lock`).

## What the reference pins, and why

The set is the BSE JSON document of `data/basis/` that `fb_io_basis`
compiles in, converted with `tools/atomic_cd/eri_quartet.py`'s
`basis_from_bse`; the coordinates are converted with CODATA 2018's Bohr
radius. The normalisation is the tree's, reached through the overlap: each
function's factor against PySCF's is `sqrt(S_fb[m,m] / S_pyscf[m,m])`,
`S_fb` the tree's Cartesian overlap diagonal in closed form (the published
coefficients times the (l,0,0) primitive normaliser on every component, no
contraction renormalised) and `S_pyscf` `int1e_ovlp_cart`'s. The script
fails unless that factor is one value across each shell's components.

## A new reference

From the checkout root (PySCF is in the `tools` dependency group; inside the
devcontainer, set `UV_PROJECT_ENVIRONMENT` to a scratch path first, so the
group does not install into the image's `/opt/venv`):

```bash
D=rust/integrals/schwarz/tests/data
OMP_NUM_THREADS=12 uv run --group tools python $D/reference.py \
    $D/h2o.xyz cc-pvtz $D/h2o_cc-pvtz.npy
```

It prints `sqrt((mn|mn)) 65 x 65, l <= 3, min 8.138706e-04, max
2.177428e+00`. Regenerate after a change to `data/basis/cc-pvtz.json` or
the host normalisation, and say why in the commit.
