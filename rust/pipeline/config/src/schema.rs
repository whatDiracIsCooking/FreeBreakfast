//! The config's schema: [`Config`] and its tables, each key a field whose
//! doc comment is its default and unit, read by serde.

use std::path::PathBuf;

use fb_localization_pipek_mezey::LocalizationStep;
use fb_pipeline_eri::EriStep;
use fb_pipeline_guess::GuessStep;
use fb_pipeline_scf::ScfStep;
use serde::{Deserialize, Deserializer};

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
    /// `[guess]`: the initial guess, read by [`Calculation::Guess`],
    /// [`Calculation::Rhf`] and [`Calculation::Localization`], and refused
    /// beside the overlap. Optional;
    /// default a SAD guess.
    #[serde(default)]
    pub guess: Option<GuessStep>,
    /// `[eri]`: the two-electron integrals, density-fitted. Required by
    /// [`Calculation::Rhf`] and [`Calculation::Localization`], which have
    /// no default for it, and refused beside every other calculation.
    #[serde(default)]
    pub eri: Option<EriStep>,
    /// `[scf]`: the SCF loop's options, [`ScfStep`]'s fields as its keys
    /// (`max_iterations`, `energy_tolerance` in hartree,
    /// `commutator_tolerance`, `diis_history`), each optional at the C++
    /// default its doc gives. Read by [`Calculation::Rhf`] and
    /// [`Calculation::Localization`], and refused beside any other.
    /// Optional; default every key's default.
    #[serde(default)]
    pub scf: Option<ScfStep>,
    /// `[localization]`: the localization's options, [`LocalizationStep`]'s
    /// fields as its keys (`mode`, `max_sweeps`, `functional_tolerance`,
    /// `rotation_gain_threshold`, `exponent`), each optional at the C++
    /// default its doc gives. Read by [`Calculation::Localization`] alone
    /// and refused beside any other. Optional; default every key's
    /// default.
    #[serde(default)]
    pub localization: Option<LocalizationStep>,
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
    /// `"rhf"`: the restricted Hartree-Fock energy over density-fitted
    /// integrals ([`fb_pipeline_scf::add_scf`]), closed-shell at
    /// multiplicity 1 and ROHF above it, as the C++ SCF step is one step
    /// for both, from the guess [`Config::guess`] says, over
    /// [`Config::eri`]'s fitting set, with [`Config::scf`]'s options.
    /// Writes `scf.json`, an object of the converged `energy` (hartree,
    /// electronic plus nuclear repulsion), its `residual`
    /// `‖Xᵀ (F D S - S D F) X‖_F`, the Fock builds as `iterations`, the
    /// occupations `n_alpha` and `n_beta`, and the `epsilon` array of
    /// orbital energies (hartree), ascending (an open shell's by occupation
    /// class); `c.npy`, the orbitals, column `j` that of `epsilon[j]`; and
    /// `d.npy`, the spin-summed density, `D = 2 C_occ C_occᵀ` or `Da + Db`,
    /// which `[guess] from` reads back; each matrix `nao_sphe × nao_sphe`.
    /// An electron count of the wrong parity for the multiplicity is
    /// refused.
    Rhf,
    /// `"localization"`: [`Calculation::Rhf`], writing its products, then
    /// the Pipek-Mezey localization of its occupied orbitals over Mulliken
    /// populations
    /// ([`fb_localization_pipek_mezey::add_pm_localize`]) with
    /// [`Config::localization`]'s options. Writes `c_local.npy`, the
    /// localized orbitals, `nao_sphe × n_occ`, in the order the sweeps
    /// leave them; and `localization.json`, an object of the functional
    /// `L` of the canonical occupied orbitals as `functional_before`, of
    /// the localized ones as `functional_after`, and the `sweeps` it took.
    /// A closed shell alone: a multiplicity above 1 is refused
    /// `Unsupported`, since the C++ localization of an open shell, each
    /// occupied space among itself, is not ported.
    Localization,
}

impl Calculation {
    /// Every calculation.
    pub(crate) const ALL: [Calculation; 4] = [
        Calculation::Overlap,
        Calculation::Guess,
        Calculation::Rhf,
        Calculation::Localization,
    ];

    /// The value as the config spells it.
    pub(crate) fn key(self) -> &'static str {
        match self {
            Calculation::Overlap => "overlap",
            Calculation::Guess => "guess",
            Calculation::Rhf => "rhf",
            Calculation::Localization => "localization",
        }
    }

    /// The optional tables of [`Config::tables`] this calculation reads, by
    /// key, as `(requires, accepts)`: a required table has no default and
    /// its absence is refused; an accepted one is read at its default when
    /// absent. Every other optional table is refused beside it
    /// ([`read_config`](crate::read_config)). A calculation is one row.
    pub(crate) fn tables(self) -> (&'static [&'static str], &'static [&'static str]) {
        match self {
            Calculation::Overlap => (&[], &[]),
            Calculation::Guess => (&[], &["guess"]),
            Calculation::Rhf => (&["eri"], &["guess", "scf"]),
            Calculation::Localization => (&["eri"], &["guess", "scf", "localization"]),
        }
    }

    /// Whether this calculation reads the optional table `table`, required
    /// or accepted.
    pub(crate) fn reads(self, table: &str) -> bool {
        let (requires, accepts) = self.tables();
        requires.contains(&table) || accepts.contains(&table)
    }
}

impl Config {
    /// The optional tables, by key, each with whether the config sets it:
    /// those [`Calculation::tables`] places. `[molecule]` and `[basis_set]`
    /// are required by every calculation and `[memory]` read by every one,
    /// so the type holds them.
    pub(crate) fn tables(&self) -> [(&'static str, bool); 4] {
        [
            ("guess", self.guess.is_some()),
            ("eri", self.eri.is_some()),
            ("scf", self.scf.is_some()),
            ("localization", self.localization.is_some()),
        ]
    }
}

/// `[molecule]`: the geometry, its net charge and its spin multiplicity,
/// the C++ `MoleculeStep`'s keys.
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MoleculeStep {
    /// `xyz`: the XYZ file, coordinates in Angstrom
    /// ([`fb_io_xyz::read_xyz`]). A path, relative to the config file's
    /// directory. Required; no default.
    pub xyz: PathBuf,
    /// `charge`: the net charge, in units of the elementary charge,
    /// positive for a cation, so the electron count is `Σ Z - charge`.
    /// Default 0.
    #[serde(default)]
    pub charge: i32,
    /// `multiplicity`: the spin multiplicity `2S + 1`, at least 1 and of
    /// the electron count's parity plus one; above 1 the SCF is ROHF
    /// ([`fb_pipeline_scf::add_scf`]), its count refused by
    /// [`fb_pipeline_scf::occupied_orbitals`] otherwise. Default 1.
    #[serde(default = "default_multiplicity")]
    pub multiplicity: i32,
}

/// [`MoleculeStep::multiplicity`]'s default: a singlet.
fn default_multiplicity() -> i32 {
    1
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

/// `[memory]`: the caps on each GPU worker's blocks, the keys of the C++
/// tree's memory caps (its `src/pipeline/README.md`), and on the plan's
/// heap block, a key the C++ does not have. Each key's default
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
/// max_host_mem = "32GiB"
/// ```
///
/// [`read_config`]: crate::read_config
/// [`admit_plan`]: crate::admit_plan
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct MemoryCaps {
    /// `max_device_mem_per_gpu`: one GPU worker's device memory, bytes:
    /// its block, the reserve the context and the handles hold beside it,
    /// and the kernels' stack. Bounded by the card: a block is admitted
    /// against the smaller of this and the card's total memory, less the
    /// reserve measured at planning (at least [`DEVICE_RESERVE`]) and the
    /// stack ([`admit_plan`]). Default and floor `"8GiB"`, so a card with
    /// more is used past 8 GiB only when a config says so.
    ///
    /// [`DEVICE_RESERVE`]: crate::DEVICE_RESERVE
    /// [`admit_plan`]: crate::admit_plan
    #[serde(deserialize_with = "byte_size")]
    pub max_device_mem_per_gpu: usize,
    /// `max_pinned_mem_per_gpu`: one GPU worker's pinned host block,
    /// bytes. The workers' caps together must fit the node's
    /// `MemAvailable`: pinned memory is not pageable. Default and floor
    /// `"2GiB"`.
    #[serde(deserialize_with = "byte_size")]
    pub max_pinned_mem_per_gpu: usize,
    /// `max_host_mem`: the plan's heap block, bytes, one for every GPU
    /// worker: the large flat host data its CPU nodes hold, no copy
    /// engine's (`docs/memory.md`, "The three kinds"). Beside it, the
    /// node's pinned blocks and the heap block together must fit the
    /// host's available memory ([`admit_plan`]). Default and floor
    /// `"16GiB"`.
    ///
    /// [`admit_plan`]: crate::admit_plan
    #[serde(deserialize_with = "byte_size")]
    pub max_host_mem: usize,
}

impl Default for MemoryCaps {
    /// Each key at its floor: 8 GiB of device memory, 2 GiB pinned and
    /// 16 GiB on the heap.
    fn default() -> Self {
        MemoryCaps {
            max_device_mem_per_gpu: 8 << 30,
            max_pinned_mem_per_gpu: 2 << 30,
            max_host_mem: 16 << 30,
        }
    }
}

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

fn default_output() -> PathBuf {
    PathBuf::from("fbf_output")
}

#[cfg(test)]
mod tests {
    use super::{Calculation, Config};

    /// The table of tables is whole: every optional table is read by some
    /// calculation, and each row names only optional tables, none both
    /// required and accepted.
    #[test]
    fn every_optional_table_has_a_reader() {
        let config: Config = toml::from_str(
            "calculation = \"overlap\"\n[molecule]\nxyz = \"m.xyz\"\n[basis_set]\nbasis = \
             \"sto-3g\"\n",
        )
        .unwrap();
        let tables = config.tables().map(|(table, _)| table);
        for table in tables {
            assert!(
                Calculation::ALL.iter().any(|c| c.reads(table)),
                "[{table}] is read by no calculation"
            );
        }
        for calculation in Calculation::ALL {
            let (requires, accepts) = calculation.tables();
            for table in requires.iter().chain(accepts) {
                assert!(tables.contains(table), "{calculation:?}: [{table}]");
            }
            assert!(
                requires.iter().all(|t| !accepts.contains(t)),
                "{calculation:?} both requires and accepts a table"
            );
        }
    }
}
