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

`localization_rohf.json` is the C++ tree's
`test/gtest/data/localization/rohf.json`, copied byte for byte (its
provenance inside): PySCF's DF-ROHF of H₂O⁺ and triplet CH₂ in cc-pVDZ
over cc-pVDZ-JKFIT, each occupied space localized among itself.
`localization_rohf_bars.json` holds, per case and exponent, that
reference's L and the first-order bars `tests/open_localization.rs` holds
the tree to, written by `open_localization_reference.py` (its docstring has
the analysis and the command), which first reproduces each space's L.
