# The metric's reference data

PySCF's spherical Coulomb metric (P|Q), `int2c2e`, for the two fixtures
`tests/metric.rs` compares against, so the tests need no PySCF.

| File | What |
|---|---|
| `h2o.xyz` | Water, in Angstrom: the overlap's (`rust/integrals/one_electron/overlap/tests/data/`), byte for byte |
| `feh2o.xyz` | Iron beside that water, the Fe–O bond off every axis: def2-universal-JKFIT reaches h and i only from Sc on, so water alone would stop at g |
| `h2o_cc-pvdz-jkfit.npy` | M of water in cc-pVDZ-JKFIT, 116 × 116, aux through f, every shell one primitive of coefficient 1 |
| `feh2o_def2-universal-jkfit.npy` | M of iron and water in def2-universal-JKFIT, 377 × 377, aux through i (Fe's 3h 1i), contracted shells on every atom |
| `reference.py` | The script that wrote them |

Both are float64, in the spherical order the basis's layout follows. Each
was written by PySCF 2.14.0 (`uv.lock`), which reported the smallest
eigenvalues 6.7e-5 and 1.1e-6, both above the 1e-8 the script asserts.

## What the reference pins, and why

The conventions the one-electron references pin, for the same reason
(`rust/integrals/one_electron/overlap/tests/data/README.md` has each in
full), so the comparison measures the integrals and nothing else:

- **The same basis digits**: the BSE JSON of `data/aux_basis/` that
  `fb_io_basis::resolve_aux_basis` compiles in, converted with
  `tools/atomic_cd/eri_quartet.py`'s `basis_from_bse`.
- **The same coordinates**: Angstrom converted to bohr with CODATA 2018's
  Bohr radius, and passed in bohr.
- **The same contraction normalization**: M = D M_pyscf D, D_ii the
  published contraction's norm, since the basis normalizes each primitive
  but never a contraction. D is 1 on every cc-pVDZ-JKFIT function of water;
  on the def2-universal-JKFIT fixture it is up to 0.80 from 1, so that case
  is the one that checks the convention.

## A new reference

From the checkout root (PySCF is in the `tools` dependency group; inside the
devcontainer, set `UV_PROJECT_ENVIRONMENT` to a scratch path first, so the
group does not install into the image's `/opt/venv`):

```bash
D=rust/integrals/eri/density_fitting/metric/tests/data
OMP_NUM_THREADS=8 uv run --group tools python $D/reference.py \
    $D/h2o.xyz cc-pvdz-jkfit $D/h2o_cc-pvdz-jkfit.npy
OMP_NUM_THREADS=8 uv run --group tools python $D/reference.py \
    $D/feh2o.xyz def2-universal-jkfit $D/feh2o_def2-universal-jkfit.npy
```
