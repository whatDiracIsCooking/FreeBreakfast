//! The MainBus: the config, what setup derives from it, the ledger, the
//! locks (each GPU's `GpuResources` and each manager's lock), and one slot
//! per step. A deliberate god object; a step's `run` is handed a View of
//! it and the guards the wrapper took, never the bus.
//!
//! Setup is the unbudgeted host phase: it parses the config, the geometry
//! and the bases, and **counts** every length a step's arrays need into
//! [`Dims`] by a walk over the shells that builds no array
//! ([`for_each_pair`]). Each `mem_budget` is then a function of the
//! dimensions, the step's options and the caps.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use fb_basis::BasisSet;
use fb_core_budget::{Backing, Budget, Caps, Ledger};
use fb_core_error_handling::{Error, Result};
use fb_core_flow::{Bus, Locks, Slot};
use fb_integrals_df_metric::HostPool;
use fb_molecule::Molecule;
use serde::Deserialize;

use crate::steps::core_hamiltonian::CoreHamiltonianOutput;
use crate::steps::eri::EriManager;
use crate::steps::guess::GuessOutput;
use crate::steps::localization::LocalizationOutput;
use crate::steps::metric::MetricOutput;
use crate::steps::orthogonalizer::OrthogonalizerOutput;
use crate::steps::overlap::OverlapOutput;
use crate::steps::scf::{DIIS_HISTORY_MAX, ScfOutput};

/// The config file: the calculation, one table per step family, plus the
/// run's resources. With no `calculation`, a step whose table is absent is
/// not in the DAG.
///
/// ```
/// use std::path::Path;
/// use fb_steps::Config;
///
/// let text = "[molecule]\nxyz = \"h2o.xyz\"\n[basis]\nname = \"cc-pvdz\"\n";
/// let config = Config::parse(text, Path::new("/data")).unwrap();
/// assert_eq!(config.molecule.xyz, Path::new("/data/h2o.xyz"));
/// assert!(config.eri.is_none());
/// assert_eq!(config.resources.devices, [0]);
/// ```
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Config {
    /// The calculation, the steps of its DAG and the products the run
    /// writes ([`Calculation`]); absent, the tables present pick the steps.
    pub calculation: Option<Calculation>,
    /// The geometry.
    pub molecule: MoleculeTable,
    /// The orbital basis.
    pub basis: BasisTable,
    /// The initial guess; absent, the SAD density.
    pub guess: Option<GuessTable>,
    /// The density-fitted integrals; absent, no `metric` step.
    pub eri: Option<EriTable>,
    /// The RHF loop, ROHF for an open shell; absent, no `scf` step.
    pub scf: Option<ScfTable>,
    /// The occupied orbitals' localization; absent, no `localization`
    /// step.
    pub localization: Option<LocalizationTable>,
    /// GPUs, threads and caps.
    #[serde(default)]
    pub resources: ResourcesTable,
}

/// `[molecule]`: the geometry, its net charge and its spin multiplicity.
///
/// ```
/// use std::path::Path;
/// use fb_steps::Config;
///
/// let text = "[molecule]\nxyz = \"h2o.xyz\"\ncharge = 1\nmultiplicity = 2\n\
///             [basis]\nname = \"cc-pvdz\"\n";
/// let molecule = Config::parse(text, Path::new(".")).unwrap().molecule;
/// assert_eq!(molecule.occupations(9).unwrap(), (5, 4));
/// let error = molecule.occupations(10).unwrap_err();
/// assert!(error.message.starts_with("config:4: [molecule] multiplicity = 2"));
/// ```
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoleculeTable {
    /// The XYZ file, relative to the config's directory.
    pub xyz: PathBuf,
    /// The net charge, in units of the elementary charge, positive for a
    /// cation: the electron count is `Σ Z - charge`. Default 0.
    #[serde(default)]
    pub charge: i32,
    /// The spin multiplicity `2S + 1`: at least 1, at most the electron
    /// count plus one, and of the count's parity plus one, or the run is
    /// refused (`InvalidConfig`, [`MoleculeTable::occupations`]). Above 1,
    /// an open shell, the SCF runs ROHF. Default 1.
    #[serde(default = "default_multiplicity")]
    pub multiplicity: i32,
    /// Where the spin was set, `<file>:<line>`: `multiplicity`'s line,
    /// else `charge`'s, else the table's.
    #[serde(skip)]
    at: String,
}

/// `[molecule] multiplicity`'s default: a singlet.
fn default_multiplicity() -> i32 {
    1
}

impl MoleculeTable {
    /// The occupations `(n_alpha, n_beta)` of `electrons` electrons at
    /// this multiplicity: `n_alpha - n_beta = multiplicity - 1`, a closed
    /// shell's both half the count.
    ///
    /// # Errors
    /// `InvalidConfig`, `<file>:<line>: [molecule] multiplicity = ...`
    /// naming the count and the charge: a count that is not positive, a
    /// multiplicity below 1 or past the count plus one, or one of the
    /// count's parity.
    pub fn occupations(&self, electrons: i32) -> Result<(usize, usize)> {
        let open = self.multiplicity - 1;
        if electrons <= 0 || open < 0 || electrons < open || (electrons - open) % 2 != 0 {
            return Err(Error::config(format!(
                "{}: [molecule] multiplicity = {} is impossible for {electrons} electrons \
                 (charge {})",
                self.at, self.multiplicity, self.charge
            )));
        }
        // Both non-negative, so the casts are exact.
        Ok((
            ((electrons + open) / 2).unsigned_abs() as usize,
            ((electrons - open) / 2).unsigned_abs() as usize,
        ))
    }
}

/// `calculation`: the steps its DAG holds, the tables it reads beside
/// `[molecule]`, `[basis]` and `[resources]`, and the products
/// [`Run::write`] puts beside the run's record. One row per calculation:
///
/// | `calculation` | Steps | Reads | Writes |
/// |---|---|---|---|
/// | `"overlap"` | overlap | none | `overlap.npy` |
/// | `"guess"` | guess | `[guess]` | `d.npy` |
/// | `"rhf"` | the chain to the SCF | `[guess]`, `[eri]` (required), `[scf]` | `scf.json`, `c.npy`, `d.npy` |
/// | `"localization"` | the chain to the SCF, and the localization | `[guess]`, `[eri]` (required), `[scf]`, `[localization]` | the SCF's, `localization.json`, `c_local.npy` |
///
/// An optional table it reads is at its defaults when absent. Parsing
/// refuses, `InvalidConfig`, naming the file and the line: a table it
/// does not read, at the table's line; a missing `[eri]`, at
/// `calculation`'s.
///
/// ```
/// use std::path::Path;
/// use fb_steps::{Calculation, Config};
///
/// let base = "[molecule]\nxyz = \"h2o.xyz\"\n[basis]\nname = \"cc-pvdz\"\n\
///             [eri]\naux_basis = \"cc-pvdz-jkfit\"\n";
/// let config = Config::parse(&format!("calculation = \"rhf\"\n{base}"), Path::new(".")).unwrap();
/// assert_eq!(config.calculation, Some(Calculation::Rhf));
/// assert!(config.scf.is_some() && config.localization.is_none());
/// let error = Config::parse(&format!("calculation = \"overlap\"\n{base}"), Path::new("."))
///     .unwrap_err();
/// assert!(error.message.starts_with("config:6: [eri] is set"), "{}", error.message);
/// ```
///
/// [`Run::write`]: crate::Run::write
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Calculation {
    /// `"overlap"`: S.
    Overlap,
    /// `"guess"`: the guess's density.
    Guess,
    /// `"rhf"`: the DF-RHF energy, ROHF at a multiplicity above 1.
    Rhf,
    /// `"localization"`: that SCF, then the Pipek-Mezey localization of its
    /// occupied orbitals.
    Localization,
}

impl Calculation {
    /// Its row: its name in the config and the optional tables it reads.
    fn row(self) -> (&'static str, &'static [&'static str]) {
        match self {
            Calculation::Overlap => ("overlap", &[]),
            Calculation::Guess => ("guess", &["guess"]),
            Calculation::Rhf => ("rhf", &["guess", "eri", "scf"]),
            Calculation::Localization => ("localization", &["guess", "eri", "scf", "localization"]),
        }
    }
}

/// The lines of `calculation`, of `[molecule]`'s spin keys and of the
/// optional tables, read again with their spans.
#[derive(Deserialize)]
struct Spans {
    calculation: Option<toml::Spanned<toml::Value>>,
    molecule: toml::Spanned<MoleculeSpans>,
    guess: Option<toml::Spanned<GuessSpans>>,
    eri: Option<toml::Spanned<toml::Value>>,
    scf: Option<toml::Spanned<toml::Value>>,
    localization: Option<toml::Spanned<toml::Value>>,
}

/// `[guess]`'s `from`, spanned; every other key ignored.
#[derive(Deserialize)]
struct GuessSpans {
    from: Option<toml::Spanned<toml::Value>>,
}

/// `[molecule]`'s spin keys, spanned; every other key ignored.
#[derive(Deserialize)]
struct MoleculeSpans {
    charge: Option<toml::Spanned<i32>>,
    multiplicity: Option<toml::Spanned<i32>>,
}

/// `[basis]`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BasisTable {
    /// A compiled-in name or a BSE JSON path.
    pub name: String,
}

/// `[guess]`: the SCF's starting density, the SAD density when the
/// table or both its keys are absent.
///
/// `from` loads a density saved as a float64 `.npy`, `n_ao × n_ao` in the
/// basis's spherical AO order (a run's `d.npy`), into the guess step's
/// staging in place of the SAD stamps, its bytes budgeted the same. The
/// file carries only the numbers, so only its shape is checked: a basis
/// of the same size in another order is not caught. Setup refuses, each
/// refusal naming the key, the file and the line:
///
/// - `type` and `from` together, `InvalidConfig`
///   (`<file>:<line>: [guess] type and from are both set ...`);
/// - a file that is not `n_ao × n_ao`, `InvalidConfig`
///   (`<file>:<line>: [guess] from = '<path>': the array is ...`), and one
///   that cannot be read or is no float64 `.npy`, `Io`, before anything
///   is allocated;
/// - `from` with an open shell, `Unsupported`
///   (`<file>:<line>: [guess] from = '<path>' with an open shell ...`):
///   ROHF's restart needs its alpha and beta densities, and `d.npy` holds
///   their sum.
///
/// ```
/// use std::path::Path;
/// use fb_steps::Config;
///
/// let text = "[molecule]\nxyz = \"h2o.xyz\"\n[basis]\nname = \"cc-pvdz\"\n\
///             [guess]\nfrom = \"run/d.npy\"\n";
/// let guess = Config::parse(text, Path::new("/data")).unwrap().guess.unwrap();
/// assert_eq!(guess.from.unwrap(), Path::new("/data/run/d.npy"));
/// assert!(guess.r#type.is_none());
/// ```
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GuessTable {
    /// The guess built: `"sad"`, the superposition of atomic densities
    /// from the compiled-in stamps (or computed, for a basis that carries
    /// none). Absent, `"sad"` unless `from` is set.
    pub r#type: Option<GuessType>,
    /// A density to load in place of the built one, relative to the
    /// config's directory.
    pub from: Option<PathBuf>,
    /// Where the guess was set, `<file>:<line>`: `from`'s line, else the
    /// table's.
    #[serde(skip)]
    at: String,
}

/// `[guess] type`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GuessType {
    /// `"sad"`: the superposition of atomic densities.
    #[default]
    Sad,
}

impl GuessTable {
    /// Refuse `from` beside `type`, with an open shell, or of a shape
    /// other than `n_ao × n_ao`, reading the file's header alone.
    fn check(&self, dims: &Dims, multiplicity: i32) -> Result<()> {
        let Some(path) = &self.from else {
            return Ok(());
        };
        let at = &self.at;
        if self.r#type.is_some() {
            return Err(Error::config(format!(
                "{at}: [guess] type and from are both set: from loads a density in place of \
                 the one type builds"
            )));
        }
        let from = path.display();
        if dims.n_alpha != dims.n_beta {
            return Err(Error::unsupported(format!(
                "{at}: [guess] from = '{from}' with an open shell (multiplicity \
                 {multiplicity}): ROHF's restart needs its alpha and beta densities, and a \
                 saved d.npy holds their sum"
            )));
        }
        let n = dims.n_ao;
        let shape = fb_io_npy::read_npy_shape(path).map_err(|mut e| {
            e.message = format!("{at}: [guess] from: {}", e.message);
            e
        })?;
        if shape != [n, n] {
            return Err(Error::config(format!(
                "{at}: [guess] from = '{from}': the array is {shape:?}, not [{n}, {n}], the \
                 basis's spherical AO count"
            )));
        }
        Ok(())
    }
}

/// `[eri]`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EriTable {
    /// The fitting set, a compiled-in name or a BSE JSON path.
    pub aux_basis: String,
    /// Each GPU's ERI buffer, bytes: the step limit the pair and aux batch
    /// widths are chosen within. Absent, [`DEFAULT_MEM_PER_GPU`]; under
    /// [`EriStep::least_mem_per_gpu`], the plan is refused.
    ///
    /// [`DEFAULT_MEM_PER_GPU`]: crate::DEFAULT_MEM_PER_GPU
    /// [`EriStep::least_mem_per_gpu`]: crate::EriStep::least_mem_per_gpu
    pub mem_per_gpu: Option<u64>,
    /// The depth of the build's ring and of a J/K pass's: 2, or 1.
    #[serde(default = "default_ring")]
    pub ring: usize,
    /// The Schwarz screen's threshold: a shell pair whose bound, the sum of
    /// its records' dressed Schwarz bounds, is under it is skipped in B's
    /// build, its columns left zero. At least 0, and 0 screens nothing.
    /// Default 1e-12, the old path's.
    #[serde(default = "default_schwarz_threshold")]
    pub schwarz_threshold: f64,
}

/// `[eri] ring`'s default.
fn default_ring() -> usize {
    2
}

/// `[eri] schwarz_threshold`'s default.
fn default_schwarz_threshold() -> f64 {
    1e-12
}

/// `[scf]`: every key at its default when absent.
///
/// ```
/// use std::path::Path;
/// use fb_steps::Config;
///
/// let text = "[molecule]\nxyz = \"h2o.xyz\"\n[basis]\nname = \"cc-pvdz\"\n[scf]\n";
/// let scf = Config::parse(text, Path::new(".")).unwrap().scf.unwrap();
/// assert_eq!((scf.max_iterations, scf.diis_history), (100, 8));
/// assert!(!scf.stage_sync);
/// ```
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ScfTable {
    /// The most Fock builds, at least 1; past them the step fails with
    /// `MaxIterationsExceeded`. Default 100.
    pub max_iterations: usize,
    /// `|ΔE|` between two builds under which the energy has converged,
    /// hartree, positive. Default 1e-10.
    pub energy_tolerance: f64,
    /// `‖Xᵀ (F D S - S D F) X‖_F` under which the density has converged,
    /// positive. Default 1e-7.
    pub commutator_tolerance: f64,
    /// The DIIS subspace's size, 1 to 64 pairs. Default 8.
    pub diis_history: usize,
    /// A measurement, off by default: synchronize gpu0's stream where an
    /// iteration would end a step were it three (after the J/K build,
    /// after F with its energy and residual, after the DIIS step), as the
    /// wrapper does at a step's end.
    pub stage_sync: bool,
}

impl Default for ScfTable {
    fn default() -> Self {
        ScfTable {
            max_iterations: 100,
            energy_tolerance: 1e-10,
            commutator_tolerance: 1e-7,
            diis_history: 8,
            stage_sync: false,
        }
    }
}

impl ScfTable {
    /// Refuse a value out of range.
    ///
    /// # Errors
    /// `InvalidConfig`, naming the key.
    pub fn check(&self) -> Result<()> {
        if self.max_iterations == 0 {
            return Err(Error::config("[scf] max_iterations is at least 1"));
        }
        if !(self.energy_tolerance > 0.0 && self.commutator_tolerance > 0.0) {
            return Err(Error::config(
                "[scf] energy_tolerance and commutator_tolerance are positive",
            ));
        }
        if !(1..=DIIS_HISTORY_MAX).contains(&self.diis_history) {
            return Err(Error::config(format!(
                "[scf] diis_history is {}, outside 1..={DIIS_HISTORY_MAX}",
                self.diis_history
            )));
        }
        Ok(())
    }
}

/// `[localization]`: Pipek-Mezey localization of the SCF's occupied
/// orbitals, every key at its default when absent.
///
/// ```
/// use std::path::Path;
/// use fb_steps::{Config, LocalizationMode};
///
/// let text = "[molecule]\nxyz = \"h2o.xyz\"\n[basis]\nname = \"cc-pvdz\"\n\
///             [localization]\nexponent = 4\n";
/// let table = Config::parse(text, Path::new(".")).unwrap().localization.unwrap();
/// assert_eq!((table.mode, table.exponent, table.max_sweeps), (LocalizationMode::PipekMezey, 4, 64));
/// assert!(table.check().is_ok());
/// ```
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct LocalizationTable {
    /// The localization: `"pipek_mezey"`, the default, over Mulliken
    /// populations; `"ibo"` is read and refused `Unsupported` by
    /// [`LocalizationTable::check`].
    pub mode: LocalizationMode,
    /// The power `p` of `L_p = Σ_i Σ_A (Q^A_ii)^p`: 2, Pipek and Mezey's,
    /// or 4, Knizia's IBO exponent. Default 2.
    pub exponent: i32,
    /// Sweeps before the step fails with `MaxIterationsExceeded`, each
    /// visiting every orbital pair once; at least 1. Default 64.
    pub max_sweeps: usize,
    /// Converged when a sweep moves `L` by less than this, in units of
    /// `L` (a sum of squared or fourth-power populations); non-negative.
    /// Default 1e-10.
    pub functional_tolerance: f64,
    /// A pair whose gain in `L` does not exceed this is left alone, in
    /// units of `L`; non-negative. Default 1e-14.
    pub rotation_gain_threshold: f64,
}

impl Default for LocalizationTable {
    fn default() -> Self {
        LocalizationTable {
            mode: LocalizationMode::PipekMezey,
            exponent: 2,
            max_sweeps: 64,
            functional_tolerance: 1e-10,
            rotation_gain_threshold: 1e-14,
        }
    }
}

impl LocalizationTable {
    /// Refuse a value out of range.
    ///
    /// # Errors
    /// `Unsupported` for `mode = "ibo"`; `InvalidConfig`, naming the key,
    /// for an exponent other than 2 or 4, a `max_sweeps` of 0, or a
    /// tolerance that is negative or NaN.
    pub fn check(&self) -> Result<()> {
        if self.mode == LocalizationMode::Ibo {
            return Err(Error::unsupported(
                "[localization] mode = \"ibo\": IBO needs the intrinsic atomic orbitals, not \
                 ported; \"pipek_mezey\" is the mode there is",
            ));
        }
        if self.exponent != 2 && self.exponent != 4 {
            return Err(Error::config(format!(
                "[localization] exponent is {}, not 2 or 4",
                self.exponent
            )));
        }
        if self.max_sweeps == 0 {
            return Err(Error::config("[localization] max_sweeps is at least 1"));
        }
        let refused = |t: f64| t.is_nan() || t < 0.0;
        if refused(self.functional_tolerance) || refused(self.rotation_gain_threshold) {
            return Err(Error::config(
                "[localization] functional_tolerance and rotation_gain_threshold are \
                 non-negative",
            ));
        }
        Ok(())
    }
}

/// `[localization] mode`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LocalizationMode {
    /// `"pipek_mezey"`: Pipek-Mezey over Mulliken populations.
    #[default]
    PipekMezey,
    /// `"ibo"`: intrinsic bond orbitals. Not ported, refused.
    Ibo,
}

/// `[resources]`.
#[derive(Clone, Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct ResourcesTable {
    /// The GPUs of the run, each by the device it is on; the first is
    /// gpu0, and the run's GPU count is the list's length. A device named
    /// twice is two logical GPUs on one physical device, a test facility.
    pub devices: Vec<i32>,
    /// The executor's width.
    pub threads: usize,
    /// The host integrals' pool, threads beside the executor's; absent, the
    /// host's `available_parallelism` less `threads`, floored at one. 0 is
    /// refused.
    pub host_threads: Option<usize>,
    /// The device cap per GPU, MiB; absent, the GPU's free memory after
    /// its handles exist, less its margin.
    pub device_cap_mib: Option<u64>,
    /// The host cap, heap and pinned together, MiB; absent, the host's
    /// available memory at setup less its margin.
    pub host_cap_mib: Option<u64>,
    /// Test mode: the wrapper fills each scratch lease with NaN before
    /// `run`.
    pub nan_fill: bool,
}

impl Default for ResourcesTable {
    #[allow(
        clippy::disallowed_macros,
        reason = "the config's default device list, one entry, made before any step"
    )]
    fn default() -> Self {
        ResourcesTable {
            devices: vec![0],
            threads: 4,
            host_threads: None,
            device_cap_mib: None,
            host_cap_mib: None,
            nan_fill: false,
        }
    }
}

impl Config {
    /// Parse `text`, resolving a relative `[molecule] xyz` and `[guess]
    /// from` against `dir`, and fill the optional tables `calculation`
    /// reads at their defaults. A refusal, and a later one of
    /// `[molecule]`'s spin or of `[guess]`, names the line as
    /// `config:<line>`.
    ///
    /// # Errors
    /// `InvalidConfig`, `config:<line>: ...` with the parser's message: a
    /// key or table the config does not have, a required one missing, a
    /// value of the wrong type; and [`Calculation`]'s refusals.
    pub fn parse(text: &str, dir: &Path) -> Result<Config> {
        Config::parse_in(text, dir, "config")
    }

    /// Read and parse the config file at `path`; a refusal names the file
    /// and the line, `<file>:<line>: ...`.
    ///
    /// # Errors
    /// `Io` if it cannot be read; [`Config::parse`]'s, naming the file.
    pub fn read(path: &Path) -> Result<Config> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| Error::io(format!("cannot read '{}': {e}", path.display())))?;
        Config::parse_in(
            &text,
            path.parent().unwrap_or(Path::new(".")),
            &path.display().to_string(),
        )
    }

    /// [`Config::parse`], its refusals, `[molecule]`'s spin and `[guess]`
    /// located in `file`.
    fn parse_in(text: &str, dir: &Path, file: &str) -> Result<Config> {
        let line = |offset: usize| text[..offset].matches('\n').count() + 1;
        let mut config: Config = toml::from_str(text).map_err(|e| {
            let at = e
                .span()
                .map_or(file.to_owned(), |s| format!("{file}:{}", line(s.start)));
            Error::config(format!("{at}: {}", e.message().trim_end()))
        })?;
        // The text parsed as a `Config`, so it parses again.
        let spans: Spans = toml::from_str(text).map_err(|e| Error::config(e.to_string()))?;
        let table = &spans.molecule;
        let set = table.get_ref();
        let offset = set
            .multiplicity
            .as_ref()
            .or(set.charge.as_ref())
            .map_or(table.span().start, |k| k.span().start);
        config.molecule.at = format!("{file}:{}", line(offset));
        if config.molecule.xyz.is_relative() {
            config.molecule.xyz = dir.join(&config.molecule.xyz);
        }
        if let Some(calculation) = config.calculation {
            let (name, reads) = calculation.row();
            let set = [
                ("guess", spans.guess.as_ref().map(toml::Spanned::span)),
                ("eri", spans.eri.as_ref().map(toml::Spanned::span)),
                ("scf", spans.scf.as_ref().map(toml::Spanned::span)),
                (
                    "localization",
                    spans.localization.as_ref().map(toml::Spanned::span),
                ),
            ];
            for (table, span) in set {
                if let Some(span) = span.filter(|_| !reads.contains(&table)) {
                    return Err(Error::config(format!(
                        "{file}:{}: [{table}] is set, but calculation = \"{name}\" does not \
                         read it",
                        line(span.start)
                    )));
                }
            }
            if reads.contains(&"eri") && config.eri.is_none() {
                let offset = spans.calculation.as_ref().map_or(0, |c| c.span().start);
                return Err(Error::config(format!(
                    "{file}:{}: calculation = \"{name}\" needs [eri], whose aux_basis has no \
                     default",
                    line(offset)
                )));
            }
            if reads.contains(&"scf") {
                config.scf.get_or_insert_with(ScfTable::default);
            }
            if reads.contains(&"localization") {
                config
                    .localization
                    .get_or_insert_with(LocalizationTable::default);
            }
        }
        if let (Some(guess), Some(table)) = (config.guess.as_mut(), spans.guess.as_ref()) {
            let offset = table
                .get_ref()
                .from
                .as_ref()
                .map_or(table.span().start, |k| k.span().start);
            guess.at = format!("{file}:{}", line(offset));
            if let Some(from) = guess.from.as_mut().filter(|f| f.is_relative()) {
                *from = dir.join(&*from);
            }
        }
        Ok(config)
    }
}

/// One class of orbital shell pairs, counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClassDim {
    /// The bra shells' momentum.
    pub la: i32,
    /// The ket shells', at least `la`.
    pub lb: i32,
    /// Its shell pairs.
    pub n_pair: usize,
    /// Its primitive pair records.
    pub n_rec: usize,
}

impl ClassDim {
    /// A pair's spherical columns.
    #[must_use]
    pub fn sph(&self) -> usize {
        n_sph_of(self.la) * n_sph_of(self.lb)
    }
}

/// The spherical functions of a shell of momentum `l`.
fn n_sph_of(l: i32) -> usize {
    (2 * l + 1).unsigned_abs() as usize
}

/// The dimensions: every length a step's arrays need, counted by setup and
/// built by none of it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Dims {
    /// Atoms.
    pub n_atom: usize,
    /// Spherical AOs.
    pub n_ao: usize,
    /// Doubly occupied orbitals, `n_beta`.
    pub n_occ: usize,
    /// Occupied alpha orbitals, `n_occ` plus the singly occupied ones.
    pub n_alpha: usize,
    /// Occupied beta orbitals.
    pub n_beta: usize,
    /// Orbital shells.
    pub n_shell: usize,
    /// Orbital primitives.
    pub n_prim: usize,
    /// Orbital shell pairs `i <= j`, unscreened.
    pub n_shell_pair: usize,
    /// The spherical pair columns of the packed pair layout, unscreened:
    /// the bound B is budgeted at.
    pub n_col: usize,
    /// The shell-pair classes, `(la, lb)` ascending.
    pub classes: Vec<ClassDim>,
    /// Spherical fitting functions; 0 with no `[eri]`.
    pub n_aux: usize,
    /// Fitting shells.
    pub n_aux_shell: usize,
    /// Fitting primitives: the aux records.
    pub n_aux_prim: usize,
    /// The fitting shells of each momentum.
    pub aux_shells_of_l: Vec<usize>,
}

impl Dims {
    /// Count the dimensions of `basis`, `aux` and the occupations
    /// `(n_alpha, n_beta)` ([`MoleculeTable::occupations`]), building no
    /// array.
    #[must_use]
    pub fn count(
        basis: &BasisSet,
        aux: Option<&BasisSet>,
        (n_alpha, n_beta): (usize, usize),
    ) -> Dims {
        let classes = count_classes(basis);
        let mut aux_shells_of_l = Vec::new();
        for shell in aux.iter().flat_map(|a| a.shells()) {
            let l = shell.angular_momentum().unsigned_abs() as usize;
            if aux_shells_of_l.len() <= l {
                #[allow(
                    clippy::disallowed_methods,
                    reason = "setup, the unbudgeted host phase: one count per angular momentum"
                )]
                aux_shells_of_l.resize(l + 1, 0);
            }
            aux_shells_of_l[l] += 1;
        }
        let prims =
            |b: &BasisSet| -> usize { b.shells().iter().map(|s| s.nprimitives() as usize).sum() };
        let n_shell = basis.nshells() as usize;
        Dims {
            n_atom: basis.natoms() as usize,
            n_ao: basis.nao_sphe() as usize,
            n_occ: n_beta,
            n_alpha,
            n_beta,
            n_shell,
            n_prim: prims(basis),
            n_shell_pair: n_shell * (n_shell + 1) / 2,
            n_col: classes.iter().map(|c| c.n_pair * c.sph()).sum(),
            classes,
            n_aux: aux.map_or(0, |a| a.nao_sphe() as usize),
            n_aux_shell: aux.map_or(0, |a| a.nshells() as usize),
            n_aux_prim: aux.map_or(0, prims),
            aux_shells_of_l,
        }
    }
}

/// Every shell pair `i <= j` of `basis` in the tree's sweep order, swapped
/// when `i` has the higher momentum: `f(la, lb, bra, ket)`. The pairs of
/// one class come in the order its tables list them. The walk builds
/// nothing: setup counts with it, and a step writes its tables with it
/// straight into pinned scratch.
pub fn for_each_pair(basis: &BasisSet, mut f: impl FnMut(i32, i32, u32, u32)) {
    let shells = basis.shells();
    let n = basis.nshells();
    for i in 0..n {
        for j in i..n {
            let (li, lj) = (
                shells[i as usize].angular_momentum(),
                shells[j as usize].angular_momentum(),
            );
            if li > lj {
                f(lj, li, j, i);
            } else {
                f(li, lj, i, j);
            }
        }
    }
}

/// The classes of `basis`, counted: `(la, lb)` ascending.
fn count_classes(basis: &BasisSet) -> Vec<ClassDim> {
    let shells = basis.shells();
    let mut classes: Vec<ClassDim> = Vec::new();
    for_each_pair(basis, |la, lb, bra, ket| {
        let n_rec =
            (shells[bra as usize].nprimitives() * shells[ket as usize].nprimitives()) as usize;
        match classes.iter_mut().find(|c| c.la == la && c.lb == lb) {
            Some(c) => {
                c.n_pair += 1;
                c.n_rec += n_rec;
            }
            None => classes.push(ClassDim {
                la,
                lb,
                n_pair: 1,
                n_rec,
            }),
        }
    });
    classes.sort_by_key(|c| (c.la, c.lb));
    classes
}

/// What setup fills, before budget mode. Every step may read it without
/// declaring anything: it is not another step's product.
pub struct Shared {
    /// The parsed config.
    pub config: Config,
    /// The geometry.
    pub molecule: Molecule,
    /// The orbital basis, primitive-normalized and placed.
    pub basis: BasisSet,
    /// The fitting set, likewise; `None` with no `[eri]`.
    pub aux: Option<BasisSet>,
    /// The dimensions.
    pub dims: Dims,
    /// The nuclear repulsion energy, hartree.
    pub e_nuc: f64,
    locks: Locks,
    host_pool: HostPool,
}

impl Shared {
    /// The host integrals' pool, `[resources] host_threads` threads: a host
    /// step calls a host integral crate inside its
    /// [`install`](HostPool::install), never outside it, so rayon's global
    /// pool is never built (`docs/programming_model.md`, "CPU workers").
    #[must_use]
    pub fn host_pool(&self) -> &HostPool {
        &self.host_pool
    }

    /// The GPUs of the run, whether or not a device is behind them.
    #[must_use]
    pub fn num_gpus(&self) -> usize {
        self.config.resources.devices.len()
    }

    /// syevd's work buffer at order `n` on gpu0, in doubles: a size query
    /// under gpu0's lock, which needs the live handle, so a plan with no
    /// device behind it reports 0 (`--budget-only`).
    ///
    /// # Errors
    /// The query's failure.
    pub fn syevd_lwork(&self, budget: &Budget<'_>, n: usize) -> Result<usize> {
        if self.locks.physical().is_empty() {
            return Ok(0);
        }
        self.locks.query(budget, 0, |gpu0| gpu0.syevd_lwork(n))
    }
}

/// The run's one shared object.
pub struct MainBus {
    shared: Shared,
    ledger: Ledger,
    pub(crate) overlap: Slot<OverlapOutput>,
    pub(crate) core_hamiltonian: Slot<CoreHamiltonianOutput>,
    pub(crate) metric: Slot<MetricOutput>,
    pub(crate) eri: Slot<EriManager>,
    /// The ERI's tests' reader of the manager.
    #[cfg(test)]
    pub(crate) probe: Slot<crate::steps::eri::tests::ProbeOutput>,
    pub(crate) guess: Slot<GuessOutput>,
    pub(crate) orthogonalizer: Slot<OrthogonalizerOutput>,
    pub(crate) scf: Slot<ScfOutput>,
    pub(crate) localization: Slot<LocalizationOutput>,
    /// The localization's tests' probe of its sweep body.
    #[cfg(test)]
    pub(crate) localization_probe: Slot<crate::steps::localization::tests::ProbeOutput>,
}

impl MainBus {
    /// Setup: set a linked host BLAS to one thread
    /// ([`fb_core_device::single_thread_host_blas`]), parse the geometry
    /// and the bases on the host, count the dimensions, start the host
    /// integrals' pool, make each GPU's
    /// `GpuResources`, and read and clamp the caps (a GPU's after its
    /// handles exist). Not budgeted.
    ///
    /// [`Backing::Real`] runs on the devices `[resources] devices` names.
    /// [`Backing::Host`] and [`Backing::Dry`] touch no device: a
    /// host-backed plan runs its host steps into heap "device" buffers, and
    /// a dry one can be budgeted and refused or accepted, not run, each
    /// solver work buffer counted as zero.
    ///
    /// # Errors
    /// The readers' refusals; [`MoleculeTable::occupations`]'s;
    /// [`GuessTable`]'s, before anything is allocated; a config
    /// with no GPU, no thread or no
    /// host thread, or a device the runtime does not see
    /// (`InvalidConfig`); the host integrals' threads not starting (`Io`);
    /// the recipes library's mismatch, and the integrals' and the
    /// localization's with their tables; a cap's reading.
    pub fn setup(config: Config, backing: Backing) -> Result<Arc<MainBus>> {
        fb_core_device::single_thread_host_blas();
        let spin = &config.molecule;
        let molecule = fb_io_xyz::read_xyz(&spin.xyz, spin.charge, spin.multiplicity)?;
        let occupations = spin.occupations(molecule.electron_count())?;
        let basis = fb_io_basis::resolve_basis(&config.basis.name)?
            .primitive_normalized()
            .place(&molecule)?;
        let aux = match &config.eri {
            Some(eri) => Some(
                fb_io_basis::resolve_aux_basis(&eri.aux_basis)?
                    .primitive_normalized()
                    .place(&molecule)?,
            ),
            None => None,
        };
        let dims = Dims::count(&basis, aux.as_ref(), occupations);
        if let Some(guess) = &config.guess {
            guess.check(&dims, spin.multiplicity)?;
        }
        let e_nuc = fb_molecule::nuclear_repulsion_energy(&molecule);

        let res = &config.resources;
        if res.devices.is_empty() || res.threads == 0 || res.host_threads == Some(0) {
            return Err(Error::config(
                "[resources] devices names a GPU, and threads and host_threads are at least 1",
            ));
        }
        let host_threads = res.host_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map_or(1, usize::from)
                .saturating_sub(res.threads)
                .max(1)
        });
        let host_pool = HostPool::new(host_threads).map_err(|e| {
            Error::io(format!(
                "cannot start the host integrals' {host_threads} threads: {e}"
            ))
        })?;
        let real = backing == Backing::Real;
        if real {
            fb_recipes_sys::check_library().map_err(Error::gpu)?;
            if config.eri.is_some() {
                fb_integrals_sys::check_library().map_err(Error::gpu)?;
            }
            if config.localization.is_some() {
                fb_localization_sys::check_library().map_err(Error::gpu)?;
            }
            let seen = fb_core_device::device_count()?;
            if let Some(d) = res.devices.iter().find(|&&d| d < 0 || d >= seen) {
                return Err(Error::config(format!(
                    "[resources] devices names device {d}, but {seen} are visible"
                )));
            }
        }
        // The handles first: a defaulted cap then already sees them.
        let locks = Locks::new(if real { &res.devices } else { &[] });
        let written = res.device_cap_mib.map(|mib| mib << 20);
        #[allow(
            clippy::disallowed_macros,
            reason = "setup, the unbudgeted host phase: one written cap per GPU"
        )]
        let caps = Caps::read(
            &res.devices,
            &vec![written; res.devices.len()],
            res.host_cap_mib.map(|mib| mib << 20),
            backing,
        )?;
        let ledger = Ledger::new(caps, backing, res.devices.clone(), res.nan_fill);
        Ok(Arc::new(MainBus {
            shared: Shared {
                config,
                molecule,
                basis,
                aux,
                dims,
                e_nuc,
                locks,
                host_pool,
            },
            ledger,
            overlap: Slot::new(),
            core_hamiltonian: Slot::new(),
            metric: Slot::new(),
            eri: Slot::new(),
            #[cfg(test)]
            probe: Slot::new(),
            guess: Slot::new(),
            orthogonalizer: Slot::new(),
            scf: Slot::new(),
            localization: Slot::new(),
            #[cfg(test)]
            localization_probe: Slot::new(),
        }))
    }
}

impl Bus for MainBus {
    type Shared = Shared;

    fn shared(&self) -> &Shared {
        &self.shared
    }

    fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    fn locks(&self) -> &Locks {
        &self.shared.locks
    }
}
