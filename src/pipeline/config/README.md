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
                             PrimitivePairClassesStep, EriStep, OverlapStep,
                             OrthogonalizerStep, CoreHamiltonianStep, GuessStep, ScfStep,
                             LocalizationStep, CasciStep, CasscfStep, FciStep, Nevpt2Step,
                             CisStep>;

Result<Calculation> parse_config(std::string_view toml_text, const std::filesystem::path &input_dir,
                                 const std::filesystem::path &output_dir);
Result<Calculation> read_config(const std::filesystem::path &path,
                                const std::filesystem::path &input_dir,
                                const std::filesystem::path &output_dir);
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

[eri]                     # every option has a default
mode = "integral_direct"  # default "integral_direct"; or "density_fitting"

[orthogonalizer]          # every option has a default
eigenvalue_floor = 1e-10  # default 1e-10

[core_hamiltonian]        # no options: the table is the request

[guess]
type = "sad"              # default "sad", the only kind so far

[scf]                     # every option has a default
max_iterations = 100

[localization]            # every option has a default
mode = "pipek_mezey"      # default "pipek_mezey"; or "ibo"
exponent = 4              # 2 or 4; default 2, or 4 for "ibo"
livvo = true              # default false; the LIVVOs too, in either mode

[casci]
ncas = 4                  # required
nelecas = 4               # required
nroots = 3                # default 1
active = [2, 4, 5, 7]     # 0-based MO indices; default the frontier orbitals
```

Or, in place of `ncas`, `nelecas` and `active`, AVAS picks the active space
from MINAO AO labels, and refuses any of the three beside it:

```toml
[casci]
avas = ["O 2p", "H 1s"]   # "<symbol> <n><l>"; every m, every atom of the element
avas_threshold = 0.2      # default 0.2; needs avas
```

`[eri]` picks the two-electron integrals for the **whole pipeline**: the
`EriStep` builds them once, and `[scf]` and `[casscf]` both read its product
(`eri/README.md`). `mode = "integral_direct"`, the default, is J's and K's
pair classes. `"density_fitting"` fits J and K over an aux basis:

```toml
[eri]
mode = "density_fitting"
aux_basis = "cc-pvdz-jkfit"   # optional: a compiled-in fitting set, or a BSE JSON path
```

Left out, `aux_basis` is `<basis>-acCD-6-6` when the basis is a compiled-in
set that has one carried; `EriStep::run` picks it from `BasisSet::name()`, so
this front end never loads a basis, and a basis with none is `InvalidConfig`
then, asking for `aux_basis`. Written, it follows `[basis_set]`'s rule over
the fitting sets: joined to the input directory only when it is not a
compiled-in fitting set's name, and an input file like any other. A missing
`[eri]` is filled in, integral-direct, as every defaulted table is.

Each mode refuses the other's options: `aux_basis` under `"integral_direct"`
is `InvalidConfig`, and so is `[scf] fock_rebuild_interval` beside `mode =
"density_fitting"`, since every DF build is full. That one spans two tables,
so `parse_config` checks it once the pipeline is planned (`[eri]` may be
filled in, or written after `[scf]`). `[scf]` and `[casscf]` take no `eri` or
`aux_basis` of their own, and there is no `[density_fitting]`: one pipeline
cannot mix a fitted SCF with a direct CASSCF (one aux basis per pipeline; a
mix is two runs).

`[localization]`'s method options sit in the table beside `mode`, not in
a subtable named after `LocalizationStep::pipek_mezey`: both modes run the
same Pipek-Mezey sweep, IBO over intrinsic-atomic-orbital populations, so they
take the same options. `livvo` adds the localized valence virtual orbitals
(`pipeline/localization`'s README) in either mode.

`[casci]`'s keys are read for their types only. The active-space keys
(`ncas`, `nelecas`, `active`, `avas`, `avas_threshold`) and their refusals are
`read_active_space`'s, one reader for every step whose config derives from
`ActiveSpaceSelection`. Whether the active space fits needs the molecule's
electrons and the basis's orbitals, so `ActiveSpaceSelection::orbital_order`
checks the ranges when the step runs (`active_space/README.md`).

```toml
[casscf]
ncas = 3                  # the start: the same selection keys as [casci], avas too
nelecas = 4
nroots = 2                # default 1
weights = [0.5, 0.5]      # default equal; nroots of them, >= 0, summing to 1
conv_tol = 1e-10          # default 1e-10: |dE| at convergence
conv_tol_grad = 1e-6      # default 1e-6: ||g|| at convergence
max_cycle = 50            # default 50 macro iterations
```

`[casscf]` reads the selection as `[casci]` does, and its own keys are checked
when the file is read: `weights` of the wrong count, with a negative entry or
not summing to 1 within 1e-12 (`mcscf::solve`'s own bar), a tolerance not
positive, or `max_cycle` below 1 is InvalidConfig. Unlike the active space's
ranges, none of these needs the molecule.

`[casscf]` runs the DF-CASSCF under a density-fitted `[eri]`
(`casscf/README.md`).

```toml
[eri]
mode = "density_fitting"  # required by [nevpt2]
[casscf]
ncas = 6
nelecas = 6
[nevpt2]                  # no options: the table is the request
```

`[nevpt2]` runs SC- and PC-NEVPT2 on every root of the `[casscf]` it reads
(`nevpt2/README.md`), and only over density fitting. Under an
integral-direct `[eri]`, written so or filled in, it is `Unsupported`,
naming `[eri]`: across two tables, so `parse_config` checks it once the
pipeline is planned, before a GPU is touched rather than after the SCF and
the CASSCF have converged. The step refuses it again, for a pipeline built
by hand.

```toml
[fci]
nroots = 3                # default 1
```

`[fci]` takes no active space: it is every orbital and every electron. Whether
the determinant space fits is the device's to say, so `FciStep::run` estimates
it against the free memory when the step runs (`fci/README.md`).

```toml
[cis]
n_states = 5              # default 5: the lowest excitations
spin = "singlet"          # default "singlet"; or "triplet"
residual_tolerance = 1e-6 # default 1e-6, relative (cis::Options)
max_iterations = 100      # default 100
max_subspace = 40         # default 40; clamped to [2 n_states, n_ov] when run
```

`[cis]` is CIS over the RHF (`cis/README.md`). Its keys are `CisStep`'s
`options` (`cis::Options`: `n_states`, `spin`, and two of its `davidson`'s),
flattened into the table as `[localization]`'s are, and `max_subspace`.
`n_states` below 1, a `residual_tolerance` not positive or `max_iterations`
below 1 is `InvalidConfig` when the file is read; `n_states` past the single
excitations needs the molecule and the basis, so the step refuses it when it
runs. It runs over integral-direct ERIs only so far: under a density-fitted
`[eri]` it is `Unsupported`, naming `[eri]`, checked once the pipeline is
planned as `[nevpt2]`'s is, and refused again by the step (#200 lifts both).

**A table is a request, and the steps producing what it reads are filled
in.** The file above plans all twelve steps: `[orthogonalizer]` reads the
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

**Every path in a config is an input, relative to the input directory**:
`[molecule] xyz`, a `[basis_set] basis` that is not a compiled-in name, a
`[eri] aux_basis` that is not a compiled-in fitting set's name
(`is_builtin_aux_basis`, the other table), and every `from`. The input directory is the run's, not the config file's:
`free-breakfast -i <dir>`, default `.`, so a config can be shared between
runs that each bring their own inputs. An empty `input_dir` leaves the paths
relative to the working directory. The steps document that their paths
"arrive resolved", and this is where that happens.

**Outputs are named by the run, not the config.** Every output a planned step
can write is written, into the output directory (`free-breakfast -o <dir>`,
default `fbf_output`) under the step's `kName`. A pipeline holds one of each
step, so no two collide:

| Step | File |
|------|------|
| `[overlap]` with `write_npy = true` | `overlap.npy`: S, `nao_sphe` square, float64 |
| `[scf]` | `scf.molden`: every MO, the occupied at 2 |
| `[localization]` | `localization.molden`: the localized occupied orbitals in ascending energy, then the canonical virtuals |
| `[localization]` with `livvo = true` | `localization_livvo.molden`: the LIVVOs, at 0 |
| `[casci]` | `casci.molden`: the lowest root's natural orbitals |
| `[casscf]` | `casscf.molden`: the state-averaged natural orbitals |

There is no `molden` key, nor a `livvo_molden` or `npy` one: a config names
no output, so a later config can rely on what an earlier run wrote.
`write_npy` is a switch, off by default, not a path. A file already there is
overwritten; a directory per run is what `-o` is for. The steps keep their
`molden` member (and `LocalizationStep` its `livvo_molden`, `OverlapStep` its
`npy`), empty meaning not written, for a pipeline built by hand.

**The outputs are kept apart from the inputs.** Both are `InvalidConfig`,
comparing the directories named rather than their spelling (`.`, `''` and the
working directory spelt out are one; a trailing `/` or a `..` changes
nothing):

- an output directory that is the input directory, which would mix a run's
  outputs into its inputs;
- an input file directly in the output directory, which a step's output could
  overwrite. An absolute path, or one through `..`, reaches there however the
  two directories differ. A file in a directory *below* the output directory
  is apart, and so is an output directory below the input one, as the
  defaults are.

`read_config` then creates the output directory, before anything runs: a
directory that cannot be made is `IO`, refused before a GPU is touched rather
than after the SCF has converged. `parse_config` touches no disk.

**Strict.** Each of these is `InvalidConfig`, naming the table or key:

- a document naming no step;
- a table that names no step, or a step written as a key instead of a table;
- a missing required key, in a written table or in a filled-in step's empty one;
- a key the step has no option for (a step with no options has no keys);
- an enumerated option naming no value it has (`[guess] type`, which knows
  only `"sad"`; `[eri] mode`, which knows `"integral_direct"` and `"density_fitting"`; `[localization] mode`, which knows `"pipek_mezey"` and `"ibo"`;
  `[localization] exponent`, 2 or 4; and `[cis] spin`, which knows
  `"singlet"` and `"triplet"`);
- `from` beside another key, or on a step with no `load`;
- an option of the other ERI mode (above): `[eri] aux_basis` under
  `"integral_direct"`, or `[scf] fock_rebuild_interval` under
  `"density_fitting"`; or an empty `aux_basis`;
- a value of the wrong type. A boolean (`[localization] livvo`) is `true` or
  `false`, not `1` or `"true"`. Integers are read exactly, so `charge = 1.5` is
  refused, not truncated, and so is a value outside `int32_t`. A `double`
  option accepts an integer, so `eigenvalue_floor = 0` reads as `0.0`. An
  integer-array option (`[casci] active`) holds integers only, each read the
  same way, and a string-array option (`[casci] avas`) strings only;
- what `Pipeline::create` refuses (a cycle).

Two refusals are not `InvalidConfig` but `Unsupported`, since the request is
well formed and the method is what is missing: `[nevpt2]` under an
integral-direct `[eri]`, and `[cis]` under a density-fitted one (above).

A misspelt `multiplicty` would otherwise run silently as a singlet. (The
`io.config` this replaced ignored unknown keys, and read a flat `Config`
that restated every step option a second time.) A syntactically invalid
document is `IO`, as `io.toml` reports it.

toml++ keeps a table's keys in a sorted `std::map`, so this module never sees
the order the tables were written in. Pipeline order comes from each step's
`View`, not the file.

## Tests

`test/gtest/pipeline_config_host.cpp`, host-only: every option read, the step
defaults kept, path resolution against the input directory, the output
names, the inputs kept apart from the outputs, which steps are filled in and
which are not, `from`, and each refusal above. Nothing is run.
`test/gtest/pipeline_calculation_gpu.cpp` runs a config end to end.

## Files

| File | Role |
|------|------|
| `config.cppm` | Module interface: `Calculation`, `parse_config`, `read_config` |
| `config.cpp` | The document walk, each step's `read_step`, and `read_active_space` |
| `CMakeLists.txt` | The module library |
