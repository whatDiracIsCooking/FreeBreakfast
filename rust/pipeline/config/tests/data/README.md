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

`localization_h2o_cc-pvdz_cc-pvdz-jkfit.npy` is `[L, sens]`, written by
`localization_reference.py` (its docstring has the command): PySCF's
Pipek-Mezey maximum `L_2` of its DF-RHF of `rust/examples/water.xyz` in
cc-pVDZ over cc-pVDZ-JKFIT, and the first-order bound on how far that
maximum moves per unit of the SCF's residual, which `tests/examples.rs`'s
localization example builds its bar from.

`tests/open_localization.rs` reads `localization_rohf.json` and
`localization_rohf_bars.json` from `rust/steps/tests/data/`, where they
moved with `open_localization_reference.py` when the localization step
took the open shell.
