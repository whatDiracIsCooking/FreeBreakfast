//! `free-breakfast <config.toml>`: read the config, plan it, write the
//! plan's `plan.txt` and `plan.dot` into its output directory, admit it
//! against the `[memory]` caps, run it once on the run's devices
//! (`num_gpus`, GPU worker `w` on device `w`), and write its
//! products, `timing.json` and `timing.dot` there (`fb_pipeline_config`). `main` is the one place an error is
//! logged, and the process exits with its kind's code
//! (`docs/diagnostics.md`, "The one boundary").

#![forbid(unsafe_code)]

use std::path::Path;
use std::process::ExitCode;
use std::sync::Arc;

use fb_core_error_handling::{Error, Result, Status};
use fb_core_log::{self as log, LogLevel};
use fb_core_thread_pool::default_n_cpu;
use fb_core_wwr::runtime::{Stream, device_count};
use fb_core_wwr::solver::SolverHandle;
use fb_pipeline_config::{Card, admit_plan, plan, read_config};
use fb_recipes_diagonalize::syevd_lwork;

/// Read; check a `num_gpus` above 1 against the devices the runtime sees,
/// a count that makes no stream, for the run's device list, `0..num_gpus`, worker
/// `w` on its device `w`; plan (syevd's work asked of a solver handle on
/// device 0, if the plan asks; each adaptive width in its worker's room,
/// from its card), make the output directory and write the plan into it,
/// `plan.txt` its text and `plan.dot` its graph, so a plan admission
/// refuses can still be read; admit every worker's tables and the heap's against the
/// `[memory]` caps (each device's block by its card; the
/// host blocks against `MemAvailable` within the cgroup's limit),
/// run, and write `timing.json` and `timing.dot`, the graph with the run's
/// times on it. Everything a config can get wrong is refused before a
/// stream is made or the directory made, and a plan too large for its
/// caps or its card before a block is made.
fn run(args: &[String]) -> Status {
    let [_, config] = args else {
        let program = args.first().map_or("free-breakfast", String::as_str);
        return Err(Error::config(format!("usage: {program} <config.toml>")));
    };
    let config = read_config(Path::new(config))?;
    let devices = config.devices(device_count)?;
    // Each worker's card is read once, when planning or admission first
    // asks for it, so the widths and the admission are of one reading.
    let mut cards: Vec<Option<Card>> = vec![None; devices.len()];
    let mut card = |w: usize| -> Result<Card> {
        match cards[w] {
            Some(card) => Ok(card),
            None => Ok(*cards[w].insert(Card::read(devices[w])?)),
        }
    };
    let planned = plan(
        &config,
        |n| syevd_lwork(&SolverHandle::new(Arc::new(Stream::new(0))), n),
        &mut card,
    )?;
    std::fs::create_dir_all(&config.output).map_err(|e| {
        Error::io(format!(
            "cannot make the output directory '{}': {e}",
            config.output.display()
        ))
    })?;
    let graph = planned.graph();
    write(&config.output.join("plan.txt"), &planned.to_string())?;
    write(&config.output.join("plan.dot"), &graph.dot(None))?;
    admit_plan(
        &planned,
        &config.memory,
        &mut card,
        fb_core_arena::host_available()?,
    )?;
    // A device past the plan's workers holds nothing: the instance makes
    // blocks, and its backend streams, for the workers alone.
    let mut instance = planned.instantiate(&devices);
    let mut backend = instance.backend();
    let record = instance.run(&mut backend, default_n_cpu(1));
    // Written whether or not the run failed: the record holds the error,
    // and the times of what ran before it.
    write(&config.output.join("timing.json"), &record.to_json())?;
    write(&config.output.join("timing.dot"), &graph.dot(Some(&record)))?;
    record.result?;
    log::flush();
    println!("wrote:             {}", config.output.display());
    Ok(())
}

/// Write `text` to `path`.
///
/// # Errors
/// `Io`, naming the path, if it cannot be written.
fn write(path: &Path, text: &str) -> Status {
    std::fs::write(path, text)
        .map_err(|e| Error::io(format!("cannot write '{}': {e}", path.display())))
}

fn main() -> ExitCode {
    // The logger and its panic hook: a panic is a critical line before the
    // release abort.
    log::init();
    let args: Vec<String> = std::env::args().collect();
    let status = run(&args);
    if let Err(error) = &status {
        log::log_at(LogLevel::Error, error.origin, format_args!("{error}"));
    }
    // The boundary's line above, and every failed destroy or free the
    // teardown logged without raising it.
    println!(
        "completed: {} errors, {} warnings",
        log::error_count(),
        log::warning_count()
    );
    match status {
        Ok(()) => ExitCode::SUCCESS,
        // The codes are 1 to 8.
        Err(error) => ExitCode::from(error.kind.exit_code() as u8),
    }
}
