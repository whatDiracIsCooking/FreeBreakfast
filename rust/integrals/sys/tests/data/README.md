# The packed three-center kernels' reference data

libcint's Cartesian three-center integrals (mn|P), from PySCF's
`int3c2e_cart`, packed by column as `fb_three_center_packed_class` writes
them but with the aux index Cartesian, for the fixture
`tests/packed_witness.rs` compares against, so the test needs no PySCF.
The kernels write the aux index spherical (FelizJueves #743); the test
applies the aux basis's `U` (`fb_basis::spherical_transform_matrix`) to
this array's aux index before comparing, so the array is unchanged.

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

## The fully spherical class

One class, (p d | d), of the fully spherical packed (μν|P) of water
(`rust/examples/water.xyz`) in cc-pVDZ against cc-pVDZ-JKFIT, with the
inputs it was computed from, for `tests/three_center_spherical.rs`: the
output of `experimental/rust/df_batched/single_gpu_spherical`'s own
`ssperi3` library (`fbx_ssperi3_packed_class`), the same kernel text as
`fb_three_center_spherical_class` launches, on the same input bits.

| File | What |
|---|---|
| `spherical_records.npy` | The class's 10 bra records, 11 fields of 10, field-major (ζ, P, P − A, P − B, `s = c_a c_b (π/ζ)^{3/2}`), every (p, d) shell pair's primitive pairs in turn |
| `spherical_aux.npy` | The aux basis's 42 records, 5 fields of 42 (centre, α, `c (π/α)^{3/2}`) |
| `spherical_cells.npy` | The 36 cells, 6 fields of 36 (bra record range, aux record range, destination, diagonal flag), each an integer held as a float64: the 4 pairs against the 9 d aux shells, each pair's 15 spherical columns side by side |
| `spherical_eri3.npy` | The batch, shape (116, 60), Fortran order (aux fastest), zero outside the d aux functions' rows |

Written on the RTX 3080 (CUDA) by the experiment's `ssperi3-class-golden`
binary (`src/class_golden.rs` there, which documents each input), from the
checkout root, after building rust/'s CMake project (`cmake --preset cuda
&& cmake --build --preset cuda`, from rust/):

```bash
E=experimental/rust/df_batched/single_gpu_spherical
cmake -S $E/kernels -B $E/kernels/build -G Ninja && cmake --build $E/kernels/build
cargo run --release --manifest-path $E/Cargo.toml --bin ssperi3-class-golden -- out
for f in records aux cells eri3; do
  cp out/$f.npy rust/integrals/sys/tests/data/spherical_$f.npy
done
```

A change to either basis file, to the record formula or to the class
leaves the test comparing old inputs with old outputs, still valid; a
change to the kernels (a `deps/FelizJueves` bump) wants it written again.
