//! `src/pipeline/config` and `src/main.cpp`: the `free-breakfast` binary's
//! config, read from TOML, and the plan it asks for. The binary is this
//! crate's `src/main.rs`: `free-breakfast <config.toml>` reads the config
//! ([`read_config`]), plans it ([`plan`]), makes the output directory, runs
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
//! One calculation exists, [`Calculation::Overlap`]: the overlap of
//! [`fb_integrals_one_electron_overlap`], then its write nodes
//! ([`add_write_npy`]), `download overlap.npy` on the stream and
//! `write overlap.npy` on the CPU (`docs/design.md`, "What it writes").
//!
//! **Products are `.npy`.** A matrix is written as a float64 `.npy`,
//! `n × n` in Fortran order ([`fb_io_npy::write_npy`]): the C++ tree's
//! `overlap.npy` is the same file, a plain `np.load` reads it with no
//! reader of ours, and the tests' references (PySCF's, the SAD stamps) are
//! already that format, so a product compares against a fixture with the
//! reader the tree has.
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
//! | `write_overlap_npy`, after a stream sync inside the step | [`add_write_npy`]: a download node and a write node that feeds nothing |
//! | `-o <output_dir>`, default `fbf_output` | [`Config::output`], the same default |
//! | `-i <input_dir>` | nothing: paths are relative to the config file |
//! | `main`, `print_products` | `src/main.rs`: the error logged once, the exit code, `timing.json` |

mod write;

use std::path::{Path, PathBuf};

use fb_core_error_handling::{Error, ErrorKind, Result};
use fb_core_planner::{Dag, Planned};
use fb_integrals_one_electron_overlap::add_overlap;
use serde::Deserialize;

pub use write::add_write_npy;

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
#[derive(Clone, Debug, PartialEq, Eq, Deserialize)]
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
///   wrong type, or a `calculation` that is not one.
pub fn read_config(path: &Path) -> Result<Config> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::io(format!("cannot open '{}': {e}", path.display())))?;
    // Syntax first, so a file that is not TOML is the reader's error and a
    // TOML file the type refuses is the config's.
    toml::from_str::<toml::Table>(&text).map_err(|e| refusal(ErrorKind::Io, path, &text, &e))?;
    let mut config: Config =
        toml::from_str(&text).map_err(|e| refusal(ErrorKind::InvalidConfig, path, &text, &e))?;
    let dir = path.parent().unwrap_or(Path::new(""));
    config.molecule.xyz = dir.join(&config.molecule.xyz);
    if !fb_io_basis::is_builtin_basis(&config.basis_set.basis) {
        config.basis_set.basis = dir.join(&config.basis_set.basis).display().to_string();
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
/// [`Config::output`]. Touches no device; the output directory need not
/// exist until the plan runs.
///
/// For [`Calculation::Overlap`]: [`add_overlap`], then [`add_write_npy`]
/// of S to `overlap.npy`.
///
/// # Errors
/// The readers' refusals, the message led by the key that named the file
/// (`[molecule] xyz: `, `[basis_set] basis: `): an XYZ file that cannot be
/// read or parsed (`Io`), a basis that is neither a compiled-in name nor a
/// readable document (`Io`), an element the basis lacks; and the kernel
/// library's load check ([`add_overlap`]).
pub fn plan(config: &Config) -> Result<Planned<'static>> {
    // Charge and multiplicity do not enter the overlap.
    let molecule =
        fb_io_xyz::read_xyz(&config.molecule.xyz, 0, 1).map_err(|e| keyed(e, "[molecule] xyz"))?;
    let basis = fb_io_basis::resolve_basis(&config.basis_set.basis)
        .and_then(|stamp| stamp.primitive_normalized().place(&molecule))
        .map_err(|e| keyed(e, "[basis_set] basis"))?;
    let mut dag = Dag::new();
    match config.calculation {
        Calculation::Overlap => {
            let s = add_overlap(&mut dag, 0, &basis, &mut None)?;
            let n = basis.nao_sphe() as usize;
            add_write_npy(&mut dag, 0, s, n, config.output.join("overlap.npy"));
        }
    }
    Ok(dag.plan())
}

/// `error` with its message led by `key`, its origin kept.
fn keyed(mut error: Error, key: &str) -> Error {
    error.message = format!("{key}: {}", error.message);
    error
}
