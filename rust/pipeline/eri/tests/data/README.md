# The ERI step's reference data

PySCF's density-fitted J and K of water's SAD guess, for the one device
case `tests/jk.rs`, so the test needs no PySCF.

| File | What |
|---|---|
| `h2o.xyz` | Water, in Angstrom: the three-center crate's (`../../../../integrals/eri/density_fitting/three_center/tests/data/`), byte for byte |
| `h2o_cc-pvdz_d.npy` | The SAD guess D in cc-pVDZ, 24 × 24: the compiled-in stamps `data/sad/cc-pVDZ/{008,001}.npy` on the atoms' diagonal blocks, as the Guess step places them |
| `h2o_cc-pvdz_c.npy` | Its factor C, 24 × 7, `D = 2 C Cᵀ` to 1.3e-15 |
| `h2o_cc-pvdz_cc-pvdz-jkfit_j.npy`, `..._k.npy` | J and K of D, PySCF's `scf.RHF(mol).density_fit(auxbasis=...).get_jk` over cc-pVDZ-JKFIT, 24 × 24 |
| `reference.py` | The script that wrote all four |

Each array is float64 in Fortran order, in the tree's spherical AO order
and contraction normalization. They were written by PySCF 2.14.0
(`uv.lock`), over the BSE JSON documents of `data/basis/` and
`data/aux_basis/`, the coordinates in bohr by CODATA 2018, through the
three-center crate's `reference.py`, whose README has why each is pinned.

## A new reference

From the checkout root (inside the devcontainer, set
`UV_PROJECT_ENVIRONMENT` to a scratch path first, so the `tools` group does
not install into the image's `/opt/venv`):

```bash
D=rust/pipeline/eri/tests/data
OMP_NUM_THREADS=8 uv run --group tools python $D/reference.py \
    $D/h2o.xyz cc-pvdz cc-pvdz-jkfit $D
```
