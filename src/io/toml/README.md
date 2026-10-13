# io/toml

The TOML format module: the one place the backing TOML library (toml++) is
named. Domain modules (`config/`) import this instead of
`free_breakfast.third_party.tomlplusplus`, getting

* the vocabulary types — `free_breakfast::toml::table`, `node_view` and the
  rest, re-exported through `:shared`;
* `parse_toml(text)` / `read_toml(path)` — the one copy of the
  exception-to-`Result` translation: a syntactically invalid document is
  `ErrorKind::IO` naming the line and column, and no domain parser needs its
  own try/catch.

What a well-formed document *means* stays the domain's job — a wrong-typed
key parses fine here and fails there, classified `InvalidConfig`. Read-only
for now; a `:writer` partition joins when something needs to emit TOML.

## Files

| File | Role |
|------|------|
| `toml.cppm` | Primary interface `free_breakfast.io.toml`; `export import`s the partitions |
| `toml_shared.cppm` | `:shared` — re-exports the third-party module |
| `toml_reader.cppm` | `:reader` — `parse_toml` / `read_toml` |
| `toml_reader.cpp` | Implementation |
| `CMakeLists.txt` | The `free_breakfast.io.toml` module target |
