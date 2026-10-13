# io/json

The JSON format module: the one place the backing JSON library (nlohmann) is
named. Domain modules import this instead of
`free_breakfast.third_party.nlohmann`, getting

* the vocabulary type — `free_breakfast::json` (an alias for
  `free_breakfast::nlohmann::json`, which stays reachable too, since `:shared`
  re-exports the third-party module);
* `parse_json(text)` / `read_json(path)` — the one copy of the
  exception-to-`Result` translation: a syntactically invalid document is
  `ErrorKind::IO`, and no domain parser needs its own try/catch;
* `json_to_string(document)` / `write_json(path, document)` — pretty-printed
  serialization (doubles at full round-trip precision).

What a well-formed document *means* stays the domain's job — a wrong-shaped
document parses fine here and fails there, classified `InvalidConfig`.

## Files

| File | Role |
|------|------|
| `json.cppm` | Primary interface `free_breakfast.io.json`; `export import`s the partitions |
| `json_shared.cppm` | `:shared` — re-exports the third-party module; the `free_breakfast::json` alias |
| `json_reader.cppm` | `:reader` — `parse_json` / `read_json` |
| `json_writer.cppm` | `:writer` — `json_to_string` / `write_json` |
| `json_reader.cpp`, `json_writer.cpp` | Implementations |
| `CMakeLists.txt` | The `free_breakfast.io.json` module target |
