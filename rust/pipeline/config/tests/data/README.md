# The example's reference data

`h2o_cc-pvdz.npy` is PySCF's spherical AO overlap S of water in cc-pVDZ,
float64, 24 × 24, copied byte for byte from
`integrals/one_electron/overlap/tests/data/`, whose `README.md` says how it
was written and which conventions it pins. `rust/examples/water.xyz` is that
directory's `h2o.xyz`, byte for byte, so `tests/examples.rs` compares the S
`rust/examples/overlap.toml` writes against it at the overlap's bar.

A copy rather than a path into the overlap crate, so this crate's tests
read only files of their own (`docs/testing.md`, "Selection: the workspace
graph is the selection graph").
