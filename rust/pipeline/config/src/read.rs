//! Reading: the TOML file to a [`Config`], its refusals and its paths
//! joined to the file's directory.

use std::path::Path;

use fb_core_error_handling::{Error, ErrorKind, Result};

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
///
/// [`ScfStep::check`]: fb_pipeline_scf::ScfStep::check
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
