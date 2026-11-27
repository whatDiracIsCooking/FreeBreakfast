# The fully spherical three-center reference data

## The fully spherical class

One class, (p d | d), of the fully spherical packed (μν|P) of water
(`rust/examples/water.xyz`) in cc-pVDZ against cc-pVDZ-JKFIT, with the
inputs it was computed from, for `tests/three_center_spherical.rs`: the
output of the prototype `experimental/rust/df_batched/single_gpu_spherical`'s own
`ssperi3` library (`fbx_ssperi3_packed_class`), the same kernel text as
`fb_three_center_spherical_class` launches, on the same input bits.

| File | What |
|---|---|
| `spherical_records.npy` | The class's 10 bra records, 11 fields of 10, field-major (ζ, P, P − A, P − B, `s = c_a c_b (π/ζ)^{3/2}`), every (p, d) shell pair's primitive pairs in turn |
| `spherical_aux.npy` | The aux basis's 42 records, 5 fields of 42 (centre, α, `c (π/α)^{3/2}`) |
| `spherical_cells.npy` | The 36 cells, 6 fields of 36 (bra record range, aux record range, destination, diagonal flag), each an integer held as a float64: the 4 pairs against the 9 d aux shells, each pair's 15 spherical columns side by side |
| `spherical_eri3.npy` | The batch, shape (116, 60), Fortran order (aux fastest), zero outside the d aux functions' rows |

Written on the RTX 3080 (CUDA) by the experiment's `ssperi3-class-golden`
binary (`src/class_golden.rs` there, which documents each input). The
prototype is removed, and the crates it builds against went before it, so
the recipe runs in a checkout of `99e2d21b`, the commit that wrote these
files: from its root, after building its rust/ CMake project (`cmake
--preset cuda && cmake --build --preset cuda`, from rust/), then copying
the four files into this directory of the current tree:

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
