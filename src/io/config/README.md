# io/config

The calculation-config domain: a TOML document describing the method, basis
set, charge, multiplicity and SCF controls, parsed into a `Config`. A
syntactically invalid document yields `ErrorKind::IO`; a document that parses
but has a wrong-typed key yields `ErrorKind::InvalidConfig`. Missing keys are
not an error — the `Config` field defaults stand.

The `aux_basis` key names the auxiliary (density-fitting) basis set, e.g.
`cc-pVDZ-RI` or `def2-universal-JKFIT`. It defaults to empty and is consulted
only by the density-fitting path; a method that does not fit its integrals
ignores it. An absent key is not a parse error — a DF method finding it empty is
the method's error to report, not the reader's.

## Files

| File | Role |
|------|------|
| `config.cppm` | Primary interface `free_breakfast.io.config`; `export import`s the partitions |
| `config_shared.cppm` | `:shared` — the `Config` struct and its defaults |
| `config_reader.cppm` | `:reader` — `parse_config(text)` / `read_config(path)` |
| `config_reader.cpp` | Implementation (TOML handling arrives through `io/toml`, not the third-party module) |
| `CMakeLists.txt` | The `free_breakfast.io.config` module target |
