//! The config's schema: [`Config`] and its tables, each key a field whose
//! doc comment is its default and unit, read by serde.

use std::path::PathBuf;

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
    pub(crate) fn key(self) -> &'static str {
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
