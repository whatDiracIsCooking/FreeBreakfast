//! `src/pipeline/config` and `src/main.cpp`: the `free-breakfast` binary's
//! config, read from TOML, and the plan it asks for. The binary is this
//! crate's `src/main.rs`: `free-breakfast <config.toml>` reads the config
//! ([`read_config`]), plans it ([`plan`], syevd's work asked of device 0
//! when the calculation diagonalizes), admits every worker's tables against
//! the `[memory]` caps ([`admit_plan`]), makes the output directory, runs
//! the plan once on device 0, and writes `timing.json`, the run's
//! [`Record`](fb_core_thread_pool::Record), beside the products. Its `main`
//! is the one place an error is logged, exiting with
//! [`ErrorKind::exit_code`] (`docs/diagnostics.md`, "The one boundary").
//!
//! [`Config`] is the reference: each key is a field, its doc comment its
//! default and unit (`docs/documentation.md`, "Configuration: the reference
//! is emitted from the type"). `rust/examples/` holds one config per
//! calculation, each run through the binary by `tests/examples.rs`.
//!
//! Four calculations exist. [`Calculation::Overlap`]: the overlap of
//! [`fb_integrals_one_electron_overlap`], then its write nodes
//! ([`add_write_npy`]), `download overlap.npy` on the stream and
//! `write overlap.npy` on the CPU (`docs/design.md`, "What it writes").
//! [`Calculation::Guess`]: the initial-guess density D of
//! [`fb_pipeline_guess`], SAD or loaded as [`GuessStep`] says, then
//! `write d.npy` on the CPU ([`add_write_host_npy`]): D is a host row, so
//! it needs no download. [`Calculation::Rhf`]: the DF-RHF of
//! [`fb_pipeline_scf::add_scf`], ROHF at a multiplicity above 1, over
//! [`fb_pipeline_eri::add_df_eri`]'s fitted integrals, from that guess,
//! then the write nodes of `c.npy` and
//! `d.npy` and of `scf.json`, a stream download of ε and a CPU write of it
//! beside the last build's energy, residual and iteration count.
//! [`Calculation::Localization`]: that RHF and its writes, then the
//! Pipek-Mezey localization of its occupied orbitals
//! ([`fb_localization_pipek_mezey::add_pm_localize`]) over its S and C,
//! the decision loop after the SCF's, then the write nodes of
//! `c_local.npy` and of `localization.json`, a CPU write of the
//! localization's heap row `[L before, L after, sweeps]`.
//!
//! **Tables by calculation.** A step's table is its step crate's type,
//! used from there (no umbrella re-exports them): [`GuessStep`],
//! [`EriStep`], [`ScfStep`](fb_pipeline_scf::ScfStep),
//! [`LocalizationStep`], each with its own `check` of its values. Which of these optional tables a calculation requires and which
//! it accepts is one row per calculation, and [`read_config`] holds a
//! config to its row in one loop: a table set beside a calculation that
//! does not read it is refused, naming those that do, and a required one
//! missing is refused. A calculation is one row; a table is one entry of
//! the list of optional tables beside it.
//!
//! **Charge and multiplicity.** `[molecule]`'s `charge` and
//! `multiplicity` ([`MoleculeStep`]) are the C++ keys, defaulted to the
//! neutral singlet. The SCF reads them: an ion's SAD guess is rescaled to
//! its electron count, and a multiplicity above 1 is ROHF, both in
//! [`fb_pipeline_scf::add_scf`]. The overlap and the guess calculations
//! read neither. The localization is a closed shell's alone, and refuses a
//! multiplicity above 1 as `Unsupported`.
//!
//! **Products are `.npy`.** A matrix is written as a float64 `.npy`,
//! `n × n` in Fortran order ([`fb_io_npy::write_npy`]): the C++ tree's
//! `overlap.npy` is the same file, a plain `np.load` reads it with no
//! reader of ours, and the tests' references (PySCF's, the SAD stamps) are
//! already that format, so a product compares against a fixture with the
//! reader the tree has.
//!
//! **Memory caps.** `[memory]` ([`MemoryCaps`]) caps each GPU worker's
//! device and pinned blocks, the C++ keys with the C++ size strings, and
//! the plan's heap block, a key of its own. Each
//! key's compiled-in default is also its floor: a config with no key runs
//! at it, and one that sets a key under it is refused, so a key can only
//! raise a cap. A floor is what lets a compiled-in size be relied on to
//! fit without a search. [`admit_plan`] holds the plan to the caps after
//! [`plan`] and before the blocks are made, a device cap bounded by the
//! card and less the device's measured reserve and the kernels' stack,
//! and the node's pinned blocks and the heap block together against the
//! host's available memory, so a plan that does not fit is refused naming
//! its rows before the first launch (`docs/memory.md`, "Admission").
//!
//! **Paths.** The geometry and a basis path are relative to the config
//! file's directory, which [`read_config`] joins; a compiled-in basis name
//! is not a path and is never joined ([`fb_io_basis::is_builtin_basis`]).
//! The output directory is relative to the working directory, as the C++
//! binary's `-o` is.
//!
//! | C++ (`free_breakfast`) | Here |
//! |---|---|
//! | `read_config(config, input_dir, output_dir)`, a `Calculation` of steps | [`read_config`], a [`Config`], and [`plan`], its tasks as one [`Planned`] |
//! | toml++'s table walk, `[step] has no option 'key'` | serde, `deny_unknown_fields`: `<file>:<line>: unknown field ...` |
//! | `MoleculeStep::xyz`, `BasisSetStep::basis` | [`MoleculeStep::xyz`], [`BasisSetStep::basis`] |
//! | `[molecule] charge`, `multiplicity`, `MoleculeStep`'s, default 0 and 1 | [`MoleculeStep::charge`], [`MoleculeStep::multiplicity`], the same defaults; a count of the wrong parity refused at planning by [`fb_pipeline_scf::occupied_orbitals`] |
//! | `[overlap] write_npy`, `OverlapStep::npy` | [`Calculation::Overlap`], which always writes `overlap.npy` |
//! | `[guess] type`, `GuessStep::type`, `GuessType::Sad` | [`GuessStep::kind`], [`GuessKind::Sad`] |
//! | `[guess] from`, `Load<GuessStep>`, `GuessStep::load` | [`GuessStep::from`], [`fb_pipeline_guess::add_loaded_guess`], its file's header checked at planning |
//! | `[guess] from` with another key: `loads 'from' a file, so takes no option` | the same refusal, `InvalidConfig`, from [`GuessStep::check`] |
//! | `[eri] mode`, `EriStep::mode`, `EriMode` | [`EriStep::mode`], [`EriMode::DensityFitting`] alone and required: integral-direct is not ported |
//! | `[eri] aux_basis`, a fitting set's name or a path joined to the input directory, `<basis>-acCD-6-6` when empty | [`EriStep::aux_basis`], a compiled-in name alone and required, never joined, refused by [`read_config`] when not carried |
//! | `[eri]` beside an SCF, the integral-direct default when absent | required beside [`Calculation::Rhf`], refused beside every other, by its row of tables |
//! | `[scf]`, `ScfStep` and its keys | [`Config::scf`], [`fb_pipeline_scf::ScfStep`] itself, its range checked by [`read_config`] |
//! | the guess rescaled by `N / Σ Z` for an ion | ported in [`fb_pipeline_scf::add_scf`] |
//! | the SCF of an open shell, ROHF | [`Calculation::Rhf`] at a multiplicity above 1; `scf.json` carries `n_alpha` and `n_beta` |
//! | the localization of an open shell, each occupied space among itself | not ported: refused `Unsupported` |
//! | the RHF calculation, `ScfStep::run` over the steps before it | [`Calculation::Rhf`], [`plan`] composing the steps' tasks; `scf.json`, `c.npy`, `d.npy` its products |
//! | `ScfStep::molden`, `scf.molden` | not ported |
//! | the localization calculation, `[scf]` then `[localization]`, `LocalizationStep::run` | [`Calculation::Localization`], `calculation = "localization"`: the RHF's tasks and products, then [`add_pm_localize`]; `c_local.npy` and `localization.json` (L before and after, the sweeps) its own products |
//! | `[localization]`'s `mode`, `max_sweeps`, `functional_tolerance`, `rotation_gain_threshold`, `exponent` | [`Config::localization`], [`LocalizationStep`] itself, its values checked by [`read_config`] |
//! | `[localization] mode = "ibo"`, `livvo` | not ported: `"ibo"` refused `Unsupported`, `livvo` an unknown key |
//! | `LocalizationStep::molden`, `localization.molden`, its orbitals sorted by `<i\|F\|i>` | not ported: `c_local.npy` holds the orbitals in the order the sweeps leave them |
//! | the density the C++ pipeline names `d.npy` | [`Calculation::Guess`], which always writes `d.npy` |
//! | `write_overlap_npy`, after a stream sync inside the step | [`add_write_npy`]: a download node and a write node that feeds nothing |
//! | `[memory] max_device_mem_per_gpu`, `max_pinned_mem_per_gpu`, `read_byte_size`, each off when unset | [`MemoryCaps`], the same keys and size strings, each defaulted at its floor |
//! | `preflight`'s fold and `check_plan`, the pool capped at the cap less `kDeviceReserve`, the pre-warm against the card | [`admit_plan`]: [`fb_core_arena::admit`] on each table, the device's cap the smaller of the key and the card, less the reserve measured at planning (at least [`DEVICE_RESERVE`]) and the kernels' stack ([`KERNEL_STACK_BYTES`]) |
//! | `check_pinned_node` against `/proc/meminfo`'s `MemAvailable`, under a pinned cap | the same check in [`admit_plan`], at every cap, the default's included, against [`fb_core_arena::host_available`], `MemAvailable` within the cgroup's limit |
//! | nothing: host data too large to pin goes to pageable memory, outside every count | `[memory] max_host_mem` ([`MemoryCaps::max_host_mem`]): the plan's heap block, admitted against it, and with the pinned blocks against the host's available memory ([`fb_core_arena::admit_host`]) |
//! | `-o <output_dir>`, default `fbf_output` | [`Config::output`], the same default |
//! | `-i <input_dir>` | nothing: paths are relative to the config file |
//! | `main`, `print_products` | `src/main.rs`: the error logged once, the exit code, `timing.json` |
//!
//! [`ErrorKind::exit_code`]: fb_core_error_handling::ErrorKind::exit_code
//! [`GuessStep`]: fb_pipeline_guess::GuessStep
//! [`GuessStep::kind`]: fb_pipeline_guess::GuessStep::kind
//! [`GuessStep::from`]: fb_pipeline_guess::GuessStep::from
//! [`GuessStep::check`]: fb_pipeline_guess::GuessStep::check
//! [`GuessKind::Sad`]: fb_pipeline_guess::GuessKind::Sad
//! [`EriStep`]: fb_pipeline_eri::EriStep
//! [`EriStep::mode`]: fb_pipeline_eri::EriStep::mode
//! [`EriStep::aux_basis`]: fb_pipeline_eri::EriStep::aux_basis
//! [`EriMode::DensityFitting`]: fb_pipeline_eri::EriMode::DensityFitting
//! [`Planned`]: fb_core_planner::Planned
//! [`LocalizationStep`]: fb_localization_pipek_mezey::LocalizationStep
//! [`add_pm_localize`]: fb_localization_pipek_mezey::add_pm_localize

#![forbid(unsafe_code)]

mod admit;
mod plan;
mod read;
mod schema;
mod write;

pub use admit::{DEVICE_RESERVE, KERNEL_STACK_BYTES, admit_plan};
pub use plan::plan;
pub use read::read_config;
pub use schema::{BasisSetStep, Calculation, Config, MemoryCaps, MoleculeStep};
pub use write::{add_write_host_npy, add_write_npy};
