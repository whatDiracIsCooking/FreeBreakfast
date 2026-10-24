# Pipeline: config

The TOML front end: a config file read into the pipeline it asks for, and the
registry of every step a config may name.

## Module

`free_breakfast.pipeline.config`, target `free_breakfast::pipeline::config`. It
re-exports `pipeline.pipeline` and every step module, since `Calculation`
names them all. It is the one library that imports every step, so only it and
`main` pay that build fan-in.

```cpp
using Calculation = Pipeline<MoleculeStep, BasisSetStep, SphericalTransformStep,
                             PrimitivePairClassesStep, OverlapStep, OrthogonalizerStep,
                             CoreHamiltonianStep, GuessStep, ScfStep, LocalizationStep>;

Result<Calculation> parse_config(std::string_view toml_text, const std::filesystem::path &base_dir);
Result<Calculation> read_config(const std::filesystem::path &path);
```

A new step joins the registry here: its type in `Calculation`, and, if it has
options, a branch in `read_step` reading them. A step with options and no
branch fails a `static_assert`, rather than having every key refused.

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

[orthogonalizer]          # every option has a default
eigenvalue_floor = 1e-10  # default 1e-10

[core_hamiltonian]        # no options: the table is the request

[guess]
type = "sad"              # default "sad", the only kind so far

[scf]                     # every option has a default
max_iterations = 100
molden = "water.molden"   # default none: the converged orbitals, written

[localization]            # every option has a default
mode = "pipek_mezey"      # default "pipek_mezey", the only method so far
exponent = 4              # default 2; 2 or 4
molden = "water_pm.molden"  # default none: the localized orbitals, written
```

`[localization]`'s method options sit in the table beside `mode`, not in
a subtable named after `LocalizationStep::pipek_mezey`: Pipek-Mezey is the only
method so far.

**A table is a request, and the steps producing what it reads are filled
in.** The file above plans all ten steps: `[orthogonalizer]` reads the
overlap, which reads the spherical transform and the pair classes. Each
filled-in step is **read from an empty table**, so its defaults apply and a
required option it lacks is the same error a written table would give:
`[overlap]` alone is refused with `[molecule] needs 'xyz'`. So only a step
with a required option ever has to be written out for its own sake, and there
is no second list of which options are required. Nothing is pruned: every
planned step is upstream of a request.

The filling in is `Calculation::create`'s insert hook, called with the
missing producer's type. `Pipeline::create` without the hook stays strict, so
a pipeline built by hand lists every step.

**`from` loads a product instead of making it.** For a step with a `load`
(`step/README.md`), `from` names the file its product is read from, and the
step is planned by its load, so only what the load reads is filled in:

```toml
[guess]
from = "water_d.npy"   # the density, read in place of SAD

[scf]                  # starts from it
```

`from` stands alone: the step's options configure a run that does not happen,
so any other key beside it is refused. A step with no `load` refuses `from`.
It is a path, resolved like every other.

**Relative paths are relative to the config file**, not the working directory.
`read_config` resolves them against the file's own directory, and
`parse_config` against `base_dir`. An empty `base_dir` leaves them relative to
the working directory. The steps document that their paths "arrive
resolved", and this is where that happens.

**Strict.** Each of these is `InvalidConfig`, naming the table or key:

- a document naming no step;
- a table that names no step, or a step written as a key instead of a table;
- a missing required key, in a written table or in a filled-in step's empty one;
- a key the step has no option for (a step with no options has no keys);
- an enumerated option naming no value it has (`[guess] type`, which knows
  only `"sad"`; `[localization] mode`, which knows only `"pipek_mezey"`; and
  `[localization] exponent`, 2 or 4);
- `from` beside another key, or on a step with no `load`;
- a value of the wrong type. Integers are read exactly, so `charge = 1.5` is
  refused, not truncated, and so is a value outside `int32_t`. A `double`
  option accepts an integer, so `eigenvalue_floor = 0` reads as `0.0`;
- what `Pipeline::create` refuses (a cycle).

A misspelt `multiplicty` would otherwise run silently as a singlet. (The
`io.config` this replaced ignored unknown keys, and read a flat `Config`
that restated every step option a second time.) A syntactically invalid
document is `IO`, as `io.toml` reports it.

toml++ keeps a table's keys in a sorted `std::map`, so this module never sees
the order the tables were written in. Pipeline order comes from each step's
`View`, not the file.

## Tests

`test/gtest/pipeline_config_host.cpp`, host-only: every option read, the step
defaults kept, path resolution against the file's directory, which steps are
filled in and which are not, `from`, and each refusal above. Nothing is run.
`test/gtest/pipeline_calculation_gpu.cpp` runs a config end to end.

## Files

| File | Role |
|------|------|
| `config.cppm` | Module interface: `Calculation`, `parse_config`, `read_config` |
| `config.cpp` | The document walk and each step's `read_step` |
| `CMakeLists.txt` | The module library |
