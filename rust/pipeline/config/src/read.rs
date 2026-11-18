//! Reading: the TOML file to a [`Config`], its refusals and its paths
//! joined to the file's directory.

use std::path::Path;

use fb_core_error_handling::{Error, ErrorKind, Result, Status};
use fb_localization_pipek_mezey::LocalizationStep;
use fb_pipeline_eri::EriStep;
use fb_pipeline_guess::GuessStep;
use fb_pipeline_scf::ScfStep;

use crate::admit::gib;
use crate::{Calculation, Config, MemoryCaps};

/// Read the config at `path`, its geometry and any basis path joined to
/// `path`'s directory. Touches nothing the config names.
///
/// # Errors
/// - `Io`, naming the file: it cannot be read, or is not TOML, the message
///   `<file>:<line>: ...`.
/// - `InvalidConfig`, `<file>:<line>: ...` naming the key: a key or table
///   [`Config`] does not have, a required one missing, a value of the
///   wrong type, a `calculation`, `[guess] type` or `[localization] mode`
///   that is not one, or a `[localization] exponent` other than 2 or 4.
/// - `InvalidConfig`, `<file>:<line>: ...`: a `[memory]` value that is not
///   a byte size ([`MemoryCaps`]).
/// - `InvalidConfig`, `<file>: ...` naming the table or the key: a
///   `[guess]`, `[eri]`, `[scf]` or `[localization]` beside a calculation
///   that does not read it, naming those that do (`[eri] is read by
///   calculation = "rhf" and "localization" alone`), or a table its
///   calculation requires missing
///   (`calculation = "rhf" needs the [eri] table: it has no default`), as
///   each [`Calculation`]'s doc says which it reads; then each table's own
///   check: a `[guess]` with both `type` and `from` ([`GuessStep::check`]),
///   an `[eri] aux_basis` that is not a compiled-in fitting set, missing
///   under `mode = "density_fitting"` or given under `"conventional"`, and
///   an `[eri] schwarz_threshold` given under `"conventional"`, or negative
///   or not a number ([`EriStep::check`]), an `[scf]` option out of range
///   ([`ScfStep::check`]), a `[localization]` `max_sweeps` of 0 or a
///   tolerance out of range ([`LocalizationStep::check`]); then a
///   `[memory]` key under its floor, its default, naming the value.
/// - `Unsupported`, `<file>: ...` naming the key: `[localization] mode =
///   "ibo"`, which is not ported ([`LocalizationStep::check`]).
pub fn read_config(path: &Path) -> Result<Config> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::io(format!("cannot open '{}': {e}", path.display())))?;
    // Syntax first, so a file that is not TOML is the reader's error and a
    // TOML file the type refuses is the config's.
    toml::from_str::<toml::Table>(&text).map_err(|e| refusal(ErrorKind::Io, path, &text, &e))?;
    let mut config: Config =
        toml::from_str(&text).map_err(|e| refusal(ErrorKind::InvalidConfig, path, &text, &e))?;
    let file = path.display();
    // The kind and the origin kept: a check's `Unsupported` stays one.
    let at = |mut e: Error| {
        e.message = format!("{file}: {}", e.message);
        e
    };
    check_tables(&config).map_err(at)?;
    let checks = [
        config.guess.as_ref().map(GuessStep::check),
        config.eri.as_ref().map(EriStep::check),
        config.scf.as_ref().map(ScfStep::check),
        config.localization.as_ref().map(LocalizationStep::check),
    ];
    for check in checks.into_iter().flatten() {
        check.map_err(at)?;
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
        (
            "max_host_mem",
            config.memory.max_host_mem,
            floor.max_host_mem,
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
    if let Some(from) = config.guess.as_mut().and_then(|g| g.from.as_mut()) {
        *from = dir.join(&*from);
    }
    Ok(config)
}

/// Hold `config`'s optional tables to its calculation's row of
/// [`Calculation::tables`]: each table set must be one it reads, and each it
/// requires must be set.
///
/// # Errors
/// `InvalidConfig`, naming the first table out of place and, for one set
/// beside a calculation that does not read it, the calculations that do.
fn check_tables(config: &Config) -> Status {
    let calculation = config.calculation;
    for (table, set) in config.tables() {
        if set && !calculation.reads(table) {
            let readers: Vec<String> = Calculation::ALL
                .into_iter()
                .filter(|c| c.reads(table))
                .map(|c| format!("\"{}\"", c.key()))
                .collect();
            let readers = match readers.split_last() {
                Some((last, [])) => last.clone(),
                Some((last, rest)) => format!("{} and {last}", rest.join(", ")),
                None => "none".to_owned(),
            };
            return Err(Error::config(format!(
                "[{table}] is read by calculation = {readers} alone"
            )));
        }
        if !set && calculation.tables().0.contains(&table) {
            return Err(needs(calculation, table));
        }
    }
    Ok(())
}

/// The refusal of a config whose `calculation` requires `table` and does
/// not set it.
pub(crate) fn needs(calculation: Calculation, table: &str) -> Error {
    Error::config(format!(
        "calculation = \"{}\" needs the [{table}] table: it has no default",
        calculation.key()
    ))
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
