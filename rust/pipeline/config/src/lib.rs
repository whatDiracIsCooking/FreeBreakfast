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
//! Three calculations exist. [`Calculation::Overlap`]: the overlap of
//! [`fb_integrals_one_electron_overlap`], then its write nodes
//! ([`add_write_npy`]), `download overlap.npy` on the stream and
//! `write overlap.npy` on the CPU (`docs/design.md`, "What it writes").
//! [`Calculation::Guess`]: the initial-guess density D of
//! [`fb_pipeline_guess`], SAD or loaded as [`GuessStep`] says, then
//! `write d.npy` on the CPU ([`add_write_host_npy`]): D is a host row, so
//! it needs no download. [`Calculation::Rhf`]: the closed-shell DF-RHF of
//! [`fb_pipeline_scf::add_scf`] over [`fb_pipeline_eri::add_df_eri`]'s
//! fitted integrals, from that guess, then the write nodes of `c.npy` and
//! `d.npy` and of `scf.json`, a stream download of ε and a CPU write of it
//! beside the last build's energy, residual and iteration count.
//!
//! **Neutral molecules alone.** `[molecule]` has no charge or multiplicity
//! key, so every molecule is the neutral singlet: the C++ guess's rescale
//! by `N / Σ Z` for an ion has nothing to do here, and a key that admits
//! an ion must port it or refuse a charge.
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
//! | `[overlap] write_npy`, `OverlapStep::npy` | [`Calculation::Overlap`], which always writes `overlap.npy` |
//! | `[guess] type`, `GuessStep::type`, `GuessType::Sad` | [`GuessStep::kind`], [`GuessKind::Sad`] |
//! | `[guess] from`, `Load<GuessStep>`, `GuessStep::load` | [`GuessStep::from`], [`fb_pipeline_guess::add_loaded_guess`], its file's header checked at planning |
//! | `[guess] from` with another key: `loads 'from' a file, so takes no option` | the same refusal, `InvalidConfig` |
//! | `[eri] mode`, `EriStep::mode`, `EriMode` | [`EriStep::mode`], [`EriMode::DensityFitting`] alone and required: integral-direct is not ported |
//! | `[eri] aux_basis`, a fitting set's name or a path joined to the input directory, `<basis>-acCD-6-6` when empty | [`EriStep::aux_basis`], a compiled-in name alone and required, never joined, refused by [`read_config`] when not carried |
//! | `[eri]` beside an SCF, the integral-direct default when absent | required beside [`Calculation::Rhf`], refused beside every other |
//! | `[scf]`, `ScfStep` and its keys | [`Config::scf`], [`fb_pipeline_scf::ScfStep`] itself, its range checked by [`read_config`] |
//! | `[molecule] charge`, `multiplicity`; the guess rescaled by `N / Σ Z` | not ported: every molecule is the neutral singlet |
//! | the RHF calculation, `ScfStep::run` over the steps before it | [`Calculation::Rhf`], [`plan`] composing the steps' tasks; `scf.json`, `c.npy`, `d.npy` its products |
//! | `ScfStep::molden`, `scf.molden` | not ported |
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
//! [`Planned`]: fb_core_planner::Planned

#![forbid(unsafe_code)]

mod admit;
mod plan;
mod read;
mod schema;
mod write;

pub use admit::{DEVICE_RESERVE, KERNEL_STACK_BYTES, admit_plan};
pub use plan::plan;
pub use read::read_config;
pub use schema::{
    BasisSetStep, Calculation, Config, EriMode, EriStep, GuessKind, GuessStep, MemoryCaps,
    MoleculeStep,
};
pub use write::{add_write_host_npy, add_write_npy};
