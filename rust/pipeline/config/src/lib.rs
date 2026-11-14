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
//! device and pinned blocks, the C++ keys with the C++ size strings. Each
//! key's compiled-in default is also its floor: a config with no key runs
//! at it, and one that sets a key under it is refused, so a key can only
//! raise a cap. A floor is what lets a compiled-in size be relied on to
//! fit without a search. [`admit_plan`] holds the plan to the caps after
//! [`plan`] and before the blocks are made, a device cap bounded by the
//! card's free memory at planning, so a plan that does not fit is refused
//! naming its rows before the first launch (`docs/memory.md`,
//! "Admission").
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
//! | `preflight`'s fold and `check_plan`, the pool capped at the cap less `kDeviceReserve`, the pre-warm against the card | [`admit_plan`]: [`fb_core_arena::admit`] on each table, the device's cap the smaller of the key and the card's free memory, less [`DEVICE_RESERVE`] |
//! | `check_pinned_node` against `/proc/meminfo`'s `MemAvailable`, under a pinned cap | the same check in [`admit_plan`], at every cap, the default's included |
//! | `-o <output_dir>`, default `fbf_output` | [`Config::output`], the same default |
//! | `-i <input_dir>` | nothing: paths are relative to the config file |
//! | `main`, `print_products` | `src/main.rs`: the error logged once, the exit code, `timing.json` |

#![forbid(unsafe_code)]

mod write;

use std::path::{Path, PathBuf};

use fb_core_arena::{Pinned, RowId, admit};
use fb_core_error_handling::{Error, ErrorKind, Result, Status};
use fb_core_planner::{Dag, Planned};
use fb_integrals_mmd::add_records;
use fb_integrals_one_electron_core_hamiltonian::add_core_hamiltonian;
use fb_integrals_one_electron_overlap::add_overlap;
use fb_molecule::Molecule;
use fb_molecule_nuclei::add_nuclei;
use fb_pipeline_eri::add_df_eri;
use fb_pipeline_guess::{add_guess_upload, add_loaded_guess, add_sad_guess};
use fb_pipeline_scf::{ScfStep, add_scf, occupied_orbitals};
use fb_recipes_orthogonalize::{ORTHOGONALIZE_EIGENVALUE_FLOOR, add_orthogonalize};
use serde::{Deserialize, Deserializer};

pub use write::{add_write_host_npy, add_write_npy};

/// The binary's config: one TOML file. Every table and key not listed here
/// is refused, naming the key, the file and the line.
///
/// ```text
/// calculation = "overlap"
/// output = "overlap"
///
/// [molecule]
/// xyz = "water.xyz"
///
/// [basis_set]
/// basis = "cc-pvdz"
/// ```
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// `calculation`: what the run computes and writes. Required; no
    /// default.
    pub calculation: Calculation,
    /// `output`: the directory the products and `timing.json` are written
    /// into, made if missing. A path, relative to the working directory.
    /// Default `"fbf_output"`.
    #[serde(default = "default_output")]
    pub output: PathBuf,
    /// `[molecule]`: the geometry. Required.
    pub molecule: MoleculeStep,
    /// `[basis_set]`: the orbital basis. Required.
    pub basis_set: BasisSetStep,
    /// `[guess]`: the initial guess, read by [`Calculation::Guess`] and
    /// [`Calculation::Rhf`] and refused beside the overlap. Optional;
    /// default a SAD guess.
    #[serde(default)]
    pub guess: Option<GuessStep>,
    /// `[eri]`: the two-electron integrals, density-fitted. Required by
    /// [`Calculation::Rhf`], which has no default for it, and refused
    /// beside every other calculation.
    #[serde(default)]
    pub eri: Option<EriStep>,
    /// `[scf]`: the SCF loop's options, [`ScfStep`]'s fields as its keys
    /// (`max_iterations`, `energy_tolerance` in hartree,
    /// `commutator_tolerance`, `diis_history`), each optional at the C++
    /// default its doc gives. Read by [`Calculation::Rhf`] alone and
    /// refused beside any other. Optional; default every key's default.
    #[serde(default)]
    pub scf: Option<ScfStep>,
    /// `[memory]`: the caps on what each GPU worker's blocks may hold.
    /// Optional; default every key's default, which is also its floor.
    #[serde(default)]
    pub memory: MemoryCaps,
}

/// `calculation`: what a run computes, and the products it writes into
/// [`Config::output`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Calculation {
    /// `"overlap"`: the spherical AO overlap S
    /// ([`fb_integrals_one_electron_overlap`]), written as `overlap.npy`,
    /// `nao_sphe × nao_sphe`.
    Overlap,
    /// `"guess"`: the initial-guess density D, spin-summed, in the
    /// spherical AO layout ([`fb_pipeline_guess`]), SAD or loaded as
    /// [`Config::guess`] says, written as `d.npy`, `nao_sphe × nao_sphe`.
    /// Computes on the CPU alone.
    Guess,
    /// `"rhf"`: the closed-shell restricted Hartree-Fock energy over
    /// density-fitted integrals ([`fb_pipeline_scf::add_scf`]), from the
    /// guess [`Config::guess`] says, over [`Config::eri`]'s fitting set,
    /// with [`Config::scf`]'s options. Writes `scf.json`, an object of the
    /// converged `energy` (hartree, electronic plus nuclear repulsion), its
    /// `residual` `‖Xᵀ (F D S - S D F) X‖_F`, the Fock builds as
    /// `iterations` and the `epsilon` array of orbital energies ascending
    /// (hartree); `c.npy`, the orbitals, column `j` that of `epsilon[j]`;
    /// and `d.npy`, `D = 2 C_occ C_occᵀ`, which `[guess] from` reads back;
    /// each matrix `nao_sphe × nao_sphe`. An odd electron count is refused.
    Rhf,
}

impl Calculation {
    /// The value as the config spells it.
    fn key(self) -> &'static str {
        match self {
            Calculation::Overlap => "overlap",
            Calculation::Guess => "guess",
            Calculation::Rhf => "rhf",
        }
    }
}

/// `[molecule]`: the geometry.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoleculeStep {
    /// `xyz`: the XYZ file, coordinates in Angstrom
    /// ([`fb_io_xyz::read_xyz`]). A path, relative to the config file's
    /// directory. Required; no default.
    pub xyz: PathBuf,
}

/// `[basis_set]`: the orbital basis.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BasisSetStep {
    /// `basis`: a compiled-in name, `"cc-pvdz"` in any spelling
    /// [`fb_io_basis::is_builtin_basis`] folds, or else a BSE JSON path,
    /// relative to the config file's directory
    /// ([`fb_io_basis::resolve_basis`]). Required; no default.
    pub basis: String,
}

/// `[guess]`: the initial-guess density D, computed or loaded.
///
/// ```text
/// [guess]
/// type = "sad"        # or, in its place:
/// from = "d.npy"      # D read back, as a run of "guess" wrote it
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuessStep {
    /// `type`: the guess computed. Default `"sad"`, the only kind. Refused
    /// beside [`GuessStep::from`].
    #[serde(rename = "type", default)]
    pub kind: Option<GuessKind>,
    /// `from`: a float64 `.npy` file of D, `nao_sphe × nao_sphe` in the
    /// basis's spherical AO order, read in place of computing one
    /// ([`fb_pipeline_guess::add_loaded_guess`]), its header checked when
    /// the run is planned. A path, relative to the config file's directory.
    /// Optional; no default.
    pub from: Option<PathBuf>,
}

/// `[guess] type`: the kind of guess computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuessKind {
    /// `"sad"`: the superposition of atomic densities
    /// ([`fb_pipeline_guess::add_sad_guess`]), a spherically averaged atomic
    /// RHF per distinct (Z, shells), a compiled-in stamp when the basis is
    /// named and carries one.
    Sad,
}

/// `[eri]`: the two-electron integrals, built once for every reader of the
/// plan ([`fb_pipeline_eri::add_df_eri`]).
///
/// ```text
/// [eri]
/// mode = "density_fitting"
/// aux_basis = "cc-pvdz-jkfit"
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EriStep {
    /// `mode`: how the integrals are built. Required: `"density_fitting"`
    /// is the only mode, and the C++ default, `"integral_direct"`, is not
    /// ported, so no default stands in for it.
    pub mode: EriMode,
    /// `aux_basis`: the fitting set, a compiled-in name in any spelling
    /// [`fb_io_basis::is_builtin_aux_basis`] folds (`"cc-pvdz-jkfit"`),
    /// placed over the molecule when the run is planned. Never a path.
    /// Required; no default (the C++ `<basis>-acCD-6-6` is not ported).
    pub aux_basis: String,
}

/// `[eri] mode`: how the two-electron integrals are built.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EriMode {
    /// `"density_fitting"`: the fitted factor B = L⁻¹ (P|mn) over
    /// [`EriStep::aux_basis`], with M = (P|Q) = L Lᵀ
    /// (`fb_pipeline_eri::add_df_eri`).
    DensityFitting,
}

/// `[memory]`: the caps on each GPU worker's blocks, the keys of the C++
/// tree's memory caps (its `src/pipeline/README.md`). Each key's default
/// is also its floor: [`read_config`] refuses a value under it, naming the
/// key and the value, so a key can only raise a cap. [`admit_plan`] holds
/// the plan to them.
///
/// A value is a byte size: an integer of bytes, or a string of a number
/// and a unit, `B`; `kB`, `MB`, `GB`, `TB`, powers of 1000; `KiB`, `MiB`,
/// `GiB`, `TiB`, powers of 1024; rounded down to a whole byte. Anything
/// else is refused, `<file>:<line>: ...`: a string with no unit or an
/// unknown one, an exponent (`"1e9B"` reads as the unit `e9B`), a value
/// below 1 byte or from 2^63 bytes up, a float or any other type. A cap
/// is never guessed.
///
/// ```text
/// [memory]
/// max_device_mem_per_gpu = "10GiB"
/// max_pinned_mem_per_gpu = "2GiB"
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct MemoryCaps {
    /// `max_device_mem_per_gpu`: one GPU worker's device memory, bytes:
    /// its block and the reserve the context and the handles hold beside
    /// it ([`DEVICE_RESERVE`]). Bounded by the card: a block is admitted
    /// against the smaller of this and the card's free memory at planning,
    /// less the reserve. Default and floor `"8GiB"`, so a card with more
    /// is used past 8 GiB only when a config says so.
    #[serde(deserialize_with = "byte_size")]
    pub max_device_mem_per_gpu: usize,
    /// `max_pinned_mem_per_gpu`: one GPU worker's pinned host block,
    /// bytes. The workers' caps together must fit the node's
    /// `MemAvailable`: pinned memory is not pageable. Default and floor
    /// `"2GiB"`.
    #[serde(deserialize_with = "byte_size")]
    pub max_pinned_mem_per_gpu: usize,
}

impl Default for MemoryCaps {
    /// Each key at its floor: 8 GiB of device memory and 2 GiB pinned.
    fn default() -> Self {
        MemoryCaps {
            max_device_mem_per_gpu: 8 << 30,
            max_pinned_mem_per_gpu: 2 << 30,
        }
    }
}

/// What a card holds outside any block, bytes: the context, the library
/// handles and the loaded modules, the C++ `kDeviceReserve` of 256 MiB.
/// A device block is admitted against its cap less this. Measuring it per
/// device at process start, with this its floor, and adding the kernels'
/// stack are not built (`docs/memory.md`, "Admission").
pub const DEVICE_RESERVE: usize = 256 << 20;

/// A `[memory]` byte size, as the C++ `read_byte_size` reads one.
fn byte_size<'de, D: Deserializer<'de>>(deserializer: D) -> std::result::Result<usize, D::Error> {
    use serde::de::Visitor;

    struct ByteSize;

    impl Visitor<'_> for ByteSize {
        type Value = usize;

        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str(
                "a byte size: an integer of bytes, or a string with a unit, such as \"10GiB\"",
            )
        }

        fn visit_i64<E: serde::de::Error>(self, count: i64) -> std::result::Result<usize, E> {
            usize::try_from(count)
                .ok()
                .filter(|&b| b >= 1)
                .ok_or_else(|| E::custom(format!("{count} is below 1 byte")))
        }

        fn visit_u64<E: serde::de::Error>(self, count: u64) -> std::result::Result<usize, E> {
            self.visit_i64(i64::try_from(count).unwrap_or(i64::MAX))
        }

        fn visit_str<E: serde::de::Error>(self, text: &str) -> std::result::Result<usize, E> {
            // A sign, digits and a point: an exponent is not a byte size's
            // spelling, so "1e9B" stops at 'e' and reads as an unknown unit.
            let digits = text.strip_prefix('-').unwrap_or(text);
            let end = text.len() - digits.len()
                + digits
                    .find(|c: char| !(c.is_ascii_digit() || c == '.'))
                    .unwrap_or(digits.len());
            let (number, unit) = text.split_at(end);
            let value = number
                .parse::<f64>()
                .map_err(|_| E::custom(format!("\"{text}\" is not a number and a unit")))?;
            let unit = unit.trim_start_matches(' ');
            if unit.is_empty() {
                return Err(E::custom(format!(
                    "\"{text}\" has no unit: bytes are an integer, and a string needs a unit, \
                     such as \"10GiB\""
                )));
            }
            let scale = match unit {
                "B" => 1.0,
                "kB" => 1e3,
                "MB" => 1e6,
                "GB" => 1e9,
                "TB" => 1e12,
                "KiB" => 2f64.powi(10),
                "MiB" => 2f64.powi(20),
                "GiB" => 2f64.powi(30),
                "TiB" => 2f64.powi(40),
                _ => {
                    return Err(E::custom(format!(
                        "\"{text}\" has no unit '{unit}' (B, kB, MB, GB, TB, KiB, MiB, GiB or TiB)"
                    )));
                }
            };
            let bytes = value * scale;
            if bytes < 1.0 {
                return Err(E::custom(format!("\"{text}\" is below 1 byte")));
            }
            // 2^63: past every machine's memory, and still exact in a double.
            if bytes >= 2f64.powi(63) {
                return Err(E::custom(format!("\"{text}\" is past 2^63 bytes")));
            }
            // In [1, 2^63): the cast rounds down to a whole byte.
            Ok(bytes as usize)
        }
    }

    deserializer.deserialize_any(ByteSize)
}

/// `bytes` in GiB, for a message.
fn gib(bytes: usize) -> f64 {
    bytes as f64 / f64::from(1u32 << 30)
}

fn default_output() -> PathBuf {
    PathBuf::from("fbf_output")
}

/// Read the config at `path`, its geometry and any basis path joined to
/// `path`'s directory. Touches nothing the config names.
///
/// # Errors
/// - `Io`, naming the file: it cannot be read, or is not TOML, the message
///   `<file>:<line>: ...`.
/// - `InvalidConfig`, `<file>:<line>: ...` naming the key: a key or table
///   [`Config`] does not have, a required one missing, a value of the
///   wrong type, or a `calculation` or `[guess] type` that is not one.
/// - `InvalidConfig`, `<file>:<line>: ...`: a `[memory]` value that is not
///   a byte size ([`MemoryCaps`]).
/// - `InvalidConfig`, `<file>: ...` naming the table or the key: a
///   `[guess]` beside [`Calculation::Overlap`], or one with both `type`
///   and `from`; an `[eri]` or an `[scf]` beside a calculation other than
///   [`Calculation::Rhf`], or no `[eri]` beside it; an `[eri] aux_basis`
///   that is not a compiled-in fitting set; an `[scf]` option out of range
///   ([`ScfStep::check`]); a `[memory]` key under its floor, its default,
///   naming the value.
pub fn read_config(path: &Path) -> Result<Config> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::io(format!("cannot open '{}': {e}", path.display())))?;
    // Syntax first, so a file that is not TOML is the reader's error and a
    // TOML file the type refuses is the config's.
    toml::from_str::<toml::Table>(&text).map_err(|e| refusal(ErrorKind::Io, path, &text, &e))?;
    let mut config: Config =
        toml::from_str(&text).map_err(|e| refusal(ErrorKind::InvalidConfig, path, &text, &e))?;
    let file = path.display();
    let rhf = config.calculation == Calculation::Rhf;
    match &config.eri {
        Some(_) if !rhf => {
            return Err(Error::config(format!(
                "{file}: [eri] is read by calculation = \"rhf\" alone, and calculation = \
                 \"{}\" builds no two-electron integrals",
                config.calculation.key()
            )));
        }
        Some(eri) if !fb_io_basis::is_builtin_aux_basis(&eri.aux_basis) => {
            return Err(Error::config(format!(
                "{file}: [eri] aux_basis: '{}' is not a compiled-in fitting set",
                eri.aux_basis
            )));
        }
        None if rhf => {
            return Err(Error::config(format!(
                "{file}: calculation = \"rhf\" needs an [eri] table: it has no default"
            )));
        }
        _ => {}
    }
    if let Some(scf) = &config.scf {
        if !rhf {
            return Err(Error::config(format!(
                "{file}: [scf] is read by calculation = \"rhf\" alone"
            )));
        }
        scf.check()
            .map_err(|e| Error::config(format!("{file}: {}", e.message)))?;
    }
    let floor = MemoryCaps::default();
    for (key, value, floor) in [
        (
            "max_device_mem_per_gpu",
            config.memory.max_device_mem_per_gpu,
            floor.max_device_mem_per_gpu,
        ),
        (
            "max_pinned_mem_per_gpu",
            config.memory.max_pinned_mem_per_gpu,
            floor.max_pinned_mem_per_gpu,
        ),
    ] {
        if value < floor {
            return Err(Error::config(format!(
                "{file}: [memory] {key} = {value} bytes ({:.3} GiB) is under its floor of \
                 {floor} bytes ({} GiB), which is also its default: a key can only raise a cap",
                gib(value),
                gib(floor)
            )));
        }
    }
    let dir = path.parent().unwrap_or(Path::new(""));
    config.molecule.xyz = dir.join(&config.molecule.xyz);
    if !fb_io_basis::is_builtin_basis(&config.basis_set.basis) {
        config.basis_set.basis = dir.join(&config.basis_set.basis).display().to_string();
    }
    if let Some(guess) = &mut config.guess {
        if config.calculation == Calculation::Overlap {
            return Err(Error::config(format!(
                "{}: [guess] is read by calculation = \"guess\" and \"rhf\" alone",
                path.display()
            )));
        }
        if guess.from.is_some() && guess.kind.is_some() {
            return Err(Error::config(format!(
                "{}: [guess] loads 'from' a file, so takes no option 'type'",
                path.display()
            )));
        }
        if let Some(from) = &mut guess.from {
            *from = dir.join(&*from);
        }
    }
    Ok(config)
}

/// The TOML reader's refusal as an error of `kind`: `<file>:<line>: `
/// and its message, the line the one its span starts on.
#[track_caller]
fn refusal(kind: ErrorKind, path: &Path, text: &str, e: &toml::de::Error) -> Error {
    let at = match e.span() {
        Some(span) => format!(
            "{}:{}",
            path.display(),
            text[..span.start].matches('\n').count() + 1
        ),
        None => path.display().to_string(),
    };
    let message = format!("{at}: {}", e.message().trim_end());
    match kind {
        ErrorKind::Io => Error::io(message),
        _ => Error::config(message),
    }
}

/// Plan the run `config` asks for, on GPU worker 0: the geometry read, the
/// basis resolved, normalized and placed on it, then the calculation's
/// tasks and its write nodes, each product written into
/// [`Config::output`]. `lwork` maps the number of spherical AOs `n` to
/// syevd's work at `n` on the device the plan will run on
/// ([`syevd_lwork`](fb_recipes_diagonalize::syevd_lwork) on a solver handle
/// there); it is asked once, by [`Calculation::Rhf`] alone, after every
/// refusal planning can make of the config. Touches no device unless
/// `lwork` does; the output directory need not exist until the plan runs.
///
/// For [`Calculation::Overlap`]: the records ([`add_records`]) and
/// [`add_overlap`] over them, then [`add_write_npy`] of S to
/// `overlap.npy`. For [`Calculation::Guess`]: [`add_sad_guess`], or
/// [`add_loaded_guess`] when [`GuessStep::from`] is set, then
/// [`add_write_host_npy`] of D to `d.npy`. For [`Calculation::Rhf`]: the
/// nuclei ([`add_nuclei`]), the records, S, H ([`add_core_hamiltonian`])
/// and the fitted integrals ([`add_df_eri`]) over one staging of the
/// orbital basis's spherical transform, that guess uploaded ([`add_guess_upload`]),
/// `X = S^(-1/2)` ([`add_orthogonalize`], at the default floor) and the
/// loop ([`add_scf`]); then [`add_write_npy`] of C to `c.npy` and of D to
/// `d.npy`, and `download scf.json` (stream: ε into a pinned row) and
/// `write scf.json` (CPU: ε beside the last build's energy, residual and
/// count).
///
/// # Errors
/// The readers' refusals, the message led by the key that named the file
/// (`[molecule] xyz: `, `[basis_set] basis: `, `[guess] from: `,
/// `[eri] aux_basis: `): an XYZ file that cannot be read or parsed (`Io`),
/// a basis that is neither a compiled-in name nor a readable document
/// (`Io`), an element the basis or the fitting set lacks; a loaded guess's
/// file that cannot be opened or whose header cannot be parsed (`Io`), or
/// whose shape is not the basis's (`InvalidConfig`), checked from its
/// header alone ([`add_loaded_guess`]); for [`Calculation::Rhf`], no
/// `[eri]`, an odd electron count ([`occupied_orbitals`]) and an option out
/// of range ([`ScfStep::check`]), each `InvalidConfig`; `lwork`'s error;
/// and the kernel libraries' load checks ([`add_records`], [`add_df_eri`],
/// [`add_scf`]). The guess's data is read when the plan runs.
pub fn plan(
    config: &Config,
    lwork: impl FnOnce(usize) -> Result<usize>,
) -> Result<Planned<'static>> {
    // Charge and multiplicity: [molecule] has no key for either, so every
    // molecule is the neutral singlet.
    let molecule =
        fb_io_xyz::read_xyz(&config.molecule.xyz, 0, 1).map_err(|e| keyed(e, "[molecule] xyz"))?;
    let basis = fb_io_basis::resolve_basis(&config.basis_set.basis)
        .and_then(|stamp| stamp.primitive_normalized().place(&molecule))
        .map_err(|e| keyed(e, "[basis_set] basis"))?;
    let n = basis.nao_sphe() as usize;
    let mut dag = Dag::new();
    match config.calculation {
        Calculation::Overlap => {
            dag.blas_workspace(0);
            let classes = add_records(&mut dag, 0, &basis)?;
            let s = add_overlap(&mut dag, 0, &basis, &classes, &mut None);
            add_write_npy(&mut dag, 0, s, n, config.output.join("overlap.npy"));
        }
        Calculation::Guess => {
            let d = add_guess(&mut dag, config, &basis, &molecule)?;
            add_write_host_npy(&mut dag, 0, d, n, config.output.join("d.npy"));
        }
        Calculation::Rhf => {
            let Some(eri) = &config.eri else {
                return Err(Error::config(
                    "calculation = \"rhf\" needs an [eri] table: it has no default",
                ));
            };
            let step = config.scf.clone().unwrap_or_default();
            // Refused here, before lwork can touch a device; add_scf checks
            // both again.
            step.check()?;
            occupied_orbitals(&molecule)?;
            dag.blas_workspace(0);
            let nuclei = add_nuclei(&mut dag, 0, &molecule);
            let mut u = None;
            let classes = add_records(&mut dag, 0, &basis)?;
            let s = add_overlap(&mut dag, 0, &basis, &classes, &mut u);
            let h = add_core_hamiltonian(&mut dag, 0, &basis, &classes, nuclei, &mut u)?;
            let eri = add_df_eri(
                &mut dag,
                0,
                &molecule,
                &basis,
                &classes,
                &mut u,
                &eri.aux_basis,
            )
            .map_err(|e| match e.kind {
                ErrorKind::InvalidConfig => keyed(e, "[eri] aux_basis"),
                _ => e,
            })?;
            let guess = add_guess(&mut dag, config, &basis, &molecule)?;
            let guess = add_guess_upload(&mut dag, 0, &basis, guess);
            let lwork = lwork(n)?;
            let x = add_orthogonalize(
                &mut dag,
                0,
                "X",
                n,
                lwork,
                s,
                ORTHOGONALIZE_EIGENVALUE_FLOOR,
            )?;
            let (scalars, [eps, c, d]) = add_scf(
                &mut dag,
                0,
                &step,
                &molecule,
                n,
                lwork,
                &eri,
                [h, s, x],
                guess,
            )?;
            add_write_npy(&mut dag, 0, c, n, config.output.join("c.npy"));
            add_write_npy(&mut dag, 0, d, n, config.output.join("d.npy"));
            write::add_write_scf_json(&mut dag, 0, scalars, eps, n, config.output.join("scf.json"));
        }
    }
    Ok(dag.plan())
}

/// Admit `planned` against `memory`'s caps, as the binary does after
/// [`plan`] and before it makes the blocks (`docs/memory.md`,
/// "Admission"). In order:
///
/// 1. The node: every GPU worker's pinned cap together must fit
///    `mem_available`, the node's `MemAvailable` in bytes, since pinned
///    memory is not pageable (the C++ `check_pinned_node`).
/// 2. Each worker's device table, [`admit`]ted against the smaller of
///    [`MemoryCaps::max_device_mem_per_gpu`] and its card's free memory,
///    less [`DEVICE_RESERVE`]: so a card smaller than the cap, or one
///    another process holds part of, refuses the plan before the first
///    launch. `device_free(worker)` is that card's free memory, in bytes;
///    it is asked only of a worker whose device table holds a row, so a
///    plan with none touches no device.
/// 3. Each worker's pinned table, against
///    [`MemoryCaps::max_pinned_mem_per_gpu`].
///
/// Each worker admitted is logged at info, its blocks and their caps.
///
/// # Errors
/// - `InvalidConfig`, naming the key, the workers and `MemAvailable`: the
///   pinned caps together past it.
/// - `device_free`'s error.
/// - [`admit`]'s refusal, `Allocation`, naming the worker (`gpu<w>`), the
///   kind, the bytes over, every row over the cap alone and the rows live
///   at the peak; then the cap's terms: the key, and for a device block
///   the card's free memory and the reserve.
pub fn admit_plan(
    planned: &Planned<'_>,
    memory: &MemoryCaps,
    device_free: impl Fn(usize) -> Result<usize>,
    mem_available: usize,
) -> Status {
    let workers = planned.n_workers();
    let pinned_cap = memory.max_pinned_mem_per_gpu;
    // workers * cap > available, without the product's overflow.
    if workers != 0 && pinned_cap > mem_available / workers {
        return Err(Error::config(format!(
            "{workers} GPU workers, each allowed [memory] max_pinned_mem_per_gpu = {pinned_cap} \
             bytes ({:.3} GiB) of pinned memory, can pin more than the node's MemAvailable of \
             {mem_available} bytes ({:.3} GiB); pinned memory is not pageable",
            gib(pinned_cap),
            gib(mem_available)
        )));
    }
    for w in 0..workers {
        let worker = format!("gpu{w}");
        let device = planned.device_table(w);
        let admitted = if device.total() == 0 {
            "no device block".to_owned()
        } else {
            let cap = memory.max_device_mem_per_gpu;
            let free = device_free(w)?;
            let room = cap.min(free).saturating_sub(DEVICE_RESERVE);
            admit(device, room, &worker).map_err(|mut e| {
                e.message = format!(
                    "{}; the cap is the smaller of [memory] max_device_mem_per_gpu = {cap} bytes \
                     and the card's {free} bytes free at planning, less the reserve of \
                     {DEVICE_RESERVE} bytes",
                    e.message
                );
                e
            })?;
            format!(
                "a device block of {} bytes against a cap of {room}",
                device.total()
            )
        };
        let host = planned.host_table(w);
        admit(host, pinned_cap, &worker).map_err(|mut e| {
            e.message = format!(
                "{}; the cap is [memory] max_pinned_mem_per_gpu = {pinned_cap} bytes",
                e.message
            );
            e
        })?;
        log::info!(
            "{worker} admitted: {admitted}, a pinned block of {} bytes against a cap of \
             {pinned_cap}",
            host.total()
        );
    }
    Ok(())
}

/// The guess [`Config::guess`] asks for, on GPU worker 0's host table:
/// [`add_loaded_guess`] when it sets `from`, else [`add_sad_guess`].
fn add_guess(
    dag: &mut Dag<'static>,
    config: &Config,
    basis: &fb_basis::BasisSet,
    molecule: &Molecule,
) -> Result<RowId<Pinned, f64>> {
    match config.guess.as_ref().and_then(|g| g.from.clone()) {
        Some(from) => add_loaded_guess(dag, 0, basis, from).map_err(|e| keyed(e, "[guess] from")),
        None => add_sad_guess(dag, 0, basis, molecule.atoms()),
    }
}

/// `error` with its message led by `key`, its origin kept.
fn keyed(mut error: Error, key: &str) -> Error {
    error.message = format!("{key}: {}", error.message);
    error
}
