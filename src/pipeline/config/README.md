# Pipeline: config

The TOML front end: a config file read into the steps it names.

## Module

`free_breakfast.pipeline.config`, target `free_breakfast::pipeline::config`. It
re-exports the step modules whose types it returns.

## The config file

One table per step, named by the step's `kName`. Its keys are the step's
options, spelled as the step's members:

```toml
[molecule]
xyz = "water.xyz"   # required
charge = 0          # default 0
multiplicity = 1    # default 1

[basis_set]
basis = "sto-3g.json"   # required, a BSE JSON basis

[orthogonalizer]          # optional: every option has a default
eigenvalue_floor = 1e-10  # default 1e-10
```

```cpp
using ConfigSteps = std::tuple<MoleculeStep, BasisSetStep, OrthogonalizerStep>;

Result<ConfigSteps> parse_config(std::string_view toml_text, const std::filesystem::path &base_dir);
Result<ConfigSteps> read_config(const std::filesystem::path &path);
```

**Relative paths are relative to the config file**, not the working directory.
`read_config` resolves them against the file's own directory, and
`parse_config` against `base_dir`. An empty `base_dir` leaves them relative to
the working directory. The steps document that their paths "arrive
resolved", and this is where that happens.

**A table is required only if its step has a required option.** The
`[orthogonalizer]` table may be left out, since its one option has a default.
Steps with no options at all (`SphericalTransform`, `PrimitivePairClasses`,
`Overlap`, `CoreHamiltonian`) have no table and are not in `ConfigSteps`:
`main` constructs them.

**Strict.** Each of these is `InvalidConfig`, naming the
table or key:

- a missing table for a step with a required option, or a missing required key;
- a table that names no step;
- a key the step has no option for;
- a value of the wrong type. Integers are read exactly, so `charge = 1.5` is
  refused, not truncated, and so is a value outside `int32_t`. A `double`
  option accepts an integer, so `eigenvalue_floor = 0` reads as `0.0`.

A misspelt `multiplicty` would otherwise run silently as a singlet. (The
`io.config` this replaced ignored unknown keys, and read a flat `Config`
that restated every step option a second time.) A syntactically invalid document is `IO`,
as `io.toml` reports it.

## Until the planner

`ConfigSteps` is a fixed tuple because there is no `StepVariant` yet. When
`Pipeline<Steps...>` lands, `read_config` returns the step vector, and the
walk over the document becomes a lookup of each table's name in the
registry. Every table then becomes optional, and `Pipeline::create` reports
a missing producer.

toml++ keeps a table's keys in a sorted `std::map`, so this module never sees
the order the tables were written in. Pipeline order comes from each step's
`View`, not the file.

## Tests

`test/gtest/pipeline_config_host.cpp`, host-only: every option read, the step
defaults kept, path resolution against the file's directory, and each refusal
above.

## Files

| File | Role |
|------|------|
| `config.cppm` | Module interface: `ConfigSteps`, `parse_config`, `read_config` |
| `config.cpp` | The document walk |
| `CMakeLists.txt` | The module library |
