# io/base

The file-level primitives the rest of the I/O layer builds on: read a whole
file into a string, write a string back to a file. Both open in binary mode and
report failure through the project's `Result<T>` / `Status` (`ErrorKind::IO`,
naming the path) rather than a throw.

Every format module follows the same two-step shape — an in-memory
`parse_x(std::string_view)` / `x_to_json(...)` half, and an on-disk `read_x` /
`write_x` half that is just `read_file` / `write_file` piped into the first.
This directory is the one copy of the on-disk half.

## Files

| File | Role |
|------|------|
| `base.cppm` | Primary interface `free_breakfast.io.base`; `export import`s the two partitions |
| `base_reader.cppm` | `:reader` — `read_file(path) -> Result<std::string>` |
| `base_writer.cppm` | `:writer` — `write_file(path, contents) -> Status` |
| `base_reader.cpp`, `base_writer.cpp` | Implementations |
| `CMakeLists.txt` | The `free_breakfast.io.base` module target |
