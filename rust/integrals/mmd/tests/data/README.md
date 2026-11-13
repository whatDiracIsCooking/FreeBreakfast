# The C++ tree's records

`h2o_ccpvdz.bin` and `co_ccpvtz.bin` are the primitive-pair records the C++
tree's `primitive_pair_classes` forms for H2O/cc-pVDZ and CO/cc-pVTZ, copied
back from the device: the reference `tests/records.rs` compares this crate's
records against. Each basis is the BasisSet step's, the compiled-in set with
its primitives normalized, placed on the atoms of `tests/common/mod.rs`
(bohr, the geometries of the C++ tree's mmd fixtures).

Written on the RTX 3080 (CUDA) by `tools/mmd_records`, which the root build
builds with the tree. From the repository root, after `cmake --preset
default && cmake --build build`:

```bash
build/tools/mmd_records/free-breakfast-mmd-records rust/integrals/mmd/tests/data
```

It prints each file's class count, record count and size: 6 classes and 1629
records (234 660 bytes) for H2O, 10 classes and 7684 records (1 106 628
bytes) for CO. Two runs write the same bytes. SHA-256:

```text
ab539c23d3a6ced49c331d0f83e0122ca4ff09632cf21d33cd61a4c4ca7c73d4  h2o_ccpvdz.bin
c64abf037709e87e645fecec0d4c2a352852a82397b4b69fe1de57b1396662ad  co_ccpvtz.bin
```

## Format

Raw little-endian, no padding:

- the 8 bytes `fbmmd v1`, then the class count as a `u32`;
- per class, ascending in `(la, lb)`: `la`, `lb` and `n_rec` as `i32`; then
  16 fields of `n_rec` `f64` each, in record order: `zeta`, `p_x`, `p_y`,
  `p_z`, `pa_x`, `pa_y`, `pa_z`, `pb_x`, `pb_y`, `pb_z`, `s`, `ea`, `eb`,
  `bound`, `bound_ab`, `bound_ba`; then 4 fields of `n_rec` `i32` each,
  `off_a`, `off_b`, `rank_a`, `rank_b`.

Regenerate after a change to `data/basis/cc-pvdz.json` or `cc-pvtz.json`, the
host normalization, or the C++ mmd module, and say why in the commit.
