# IO

The file-level input/output layer: reading a calculation's inputs off disk and writing its results back. Each sub-directory is one module, laid out the same way — a primary interface named after the directory plus `<name>_{reader,writer,shared}.cppm` partitions. The layer has three tiers: `base/` (whole-file primitives), the **format** modules (`json/`, `toml/`, `npy/`, each the only place its backing representation is named), and the **domain** modules (`basis/` and `xyz/`, which say what the bytes mean). The TOML config is read one layer up, by `free_breakfast.pipeline.config`, because what it means is the pipeline's steps. `io.cppm` (`free_breakfast.io`) re-exports all of them, so one import covers the layer and each sub-module stays importable on its own.

## Sub-modules

| Directory | Module | Role |
|-----------|--------|------|
| `base/` | `free_breakfast.io.base` | The whole-file primitives everything builds on: `read_file` and `write_file`, both through `Result`/`Status`. |
| `json/` | `free_breakfast.io.json` | The JSON format: the `free_breakfast::json` vocabulary type, `parse_json`/`read_json`, `json_to_string`/`write_json`. The only place nlohmann is named. |
| `toml/` | `free_breakfast.io.toml` | The TOML format: the `free_breakfast::toml` vocabulary types and `parse_toml`/`read_toml`. The only place toml++ is named. |
| `npy/` | `free_breakfast.io.npy` | The NumPy `.npy` binary array format (float64, C order): `NpyArray`, `parse_npy`/`read_npy`, `npy_to_bytes`/`write_npy`. Interop with the Python tooling layer. |
| `xyz/` | `free_breakfast.io.xyz` | Parse a standard XYZ geometry (Angstrom) into a `Molecule` (bohr). |
| `basis/` | `free_breakfast.io.basis` | Parse a Basis Set Exchange (BSE) JSON basis into a per-element `BasisStamp` and place it over a `Molecule` to build a `BasisSet`. |

Every parser returns through `free_breakfast.core.error_handling`'s `Result<T>` / `Status`, so a malformed file becomes a classified `Error` rather than a throw or a crash. The classification is split by tier: **syntactically invalid is `IO`** (reported by the format module), **wrong content is `InvalidConfig`** (reported by the domain module). Domain modules import the format modules, never `free_breakfast.third_party.nlohmann` / `free_breakfast.third_party.tomlplusplus` directly.

## Layout convention

Within a directory `x/`:

* `x.cppm` — the primary interface `free_breakfast.io.x`; `export import`s the partitions.
* `x_reader.cppm` / `x_writer.cppm` — the `:reader` / `:writer` partitions; a module carries only the directions it has.
* `x_shared.cppm` — the `:shared` partition for what both directions (or the caller) need, e.g. `npy/`'s `NpyArray`.
* `x_reader.cpp` / `x_writer.cpp` — implementation units.

The on-disk `read_x(path)` / `write_x(path, ...)` entry points are the in-memory parsers piped through `io/base`'s `read_file` / `write_file` — no other module opens files itself.

## Tests

Host-only, in `test/gtest/io_*_host.cpp`, linked into `free_breakfast_tests_host`.

## Files

| File | Role |
|------|------|
| `io.cppm` | Aggregate module interface; `export import`s every sub-module |
| `CMakeLists.txt` | Build configuration — `add_subdirectory` per module, and the `free_breakfast.io` module target |
