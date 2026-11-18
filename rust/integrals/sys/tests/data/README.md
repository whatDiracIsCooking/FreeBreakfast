# The packed three-center kernels' reference data

libcint's Cartesian three-center integrals (mn|P), from PySCF's
`int3c2e_cart`, packed as `fb_three_center_packed_class` writes them, for
the fixture `tests/packed_witness.rs` compares against, so the test needs
no PySCF.

| File | What |
|---|---|
| `packed_witness.npy` | (mn\|P) over the test's own basis, shape (84, 946): every orbital shell pair once (shells through f on two centres, one centre's contracted) against an aux basis through i on two centres, so every class through (ff\|i) |
| `reference.py` | The script that wrote it, with the basis, the column order and the normalisation written out |

The array is float64, element (P, c) the packed column c's aux function P,
in the Fortran order fb-io-npy reads, so P fastest: the packed output row's
order. It was written by PySCF 2.14.0 (`uv.lock`).

## What the reference pins, and why

The basis is the test's, written out in both files: neither reads a carried
set, so the reference changes only when the test's basis does. The
normalisation is the test's, the coefficients as given on raw Cartesian
Gaussians, which is what the kernels compute from a `ShellTable` and an
`AuxTable`. PySCF reads a coefficient as a normalised primitive's, so the
script divides each by PySCF's `gto_norm(l, a)` before handing it over, and
then takes out PySCF's contraction renormalisation through the overlap, as
`../../../schwarz/tests/data/README.md` does: one factor per function,
`sqrt(S_test[m,m] / S_pyscf[m,m])`, checked to be one value across each
shell's components.

## A new reference

From the checkout root (PySCF is in the `tools` dependency group; inside the
devcontainer, set `UV_PROJECT_ENVIRONMENT` to a scratch path first, so the
group does not install into the image's `/opt/venv`):

```bash
D=rust/integrals/sys/tests/data
OMP_NUM_THREADS=12 uv run --group tools python $D/reference.py \
    $D/packed_witness.npy
```

It prints `(mn|P) packed, 84 aux x 946 columns, max |(mn|P)| 1.277703e+01`.
Regenerate after a change to the test's basis, and say why in the commit.
