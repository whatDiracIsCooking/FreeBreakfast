//! `free-breakfast <config.toml>`: read the config, plan it, write the
//! plan's `plan.txt` and `plan.dot` into its output directory, admit it
//! against the `[memory]` caps, run it once on device 0, and write its
//! products, `timing.json` and `timing.dot` there (`fb_pipeline_config`). `main` is the one place an error is
//! logged, and the process exits with its kind's code
//! (`docs/diagnostics.md`, "The one boundary").

#![forbid(unsafe_code)]

use std::path::Path;
use std::process::ExitCode;
use std::rc::Rc;

use fb_core_error_handling::{Error, Status};
use fb_core_log::{self as log, LogLevel};
use fb_core_thread_pool::{DeviceStream, default_n_cpu};
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::solver::SolverHandle;
use fb_pipeline_config::{KERNEL_STACK_BYTES, admit_plan, plan, read_config};
use fb_recipes_diagonalize::syevd_lwork;

/// The devices the plan's GPU workers run on, worker `w` on `DEVICES[w]`.
const DEVICES: [i32; 1] = [0];

/// Read, plan (syevd's work asked of a solver handle on device 0, if the
/// plan asks), make the output directory and write the plan into it,
/// `plan.txt` its text and `plan.dot` its graph, so a plan admission
/// refuses can still be read; admit every worker's tables and the heap's against the
/// `[memory]` caps (each device's reserve measured with its worker's
/// stream and handles made, and the kernels' stack read after it; the
/// host blocks against `MemAvailable` within the cgroup's limit),
/// run, and write `timing.json` and `timing.dot`, the graph with the run's
/// times on it. Everything a config can get wrong is refused before a
/// device is touched or the directory made, and a plan too large for its
/// caps or its card before a block is made.
fn run(args: &[String]) -> Status {
    let [_, config] = args else {
        let program = args.first().map_or("free-breakfast", String::as_str);
        return Err(Error::config(format!("usage: {program} <config.toml>")));
    };
    let config = read_config(Path::new(config))?;
    let planned = plan(&config, |n| {
        syevd_lwork(&SolverHandle::new(Rc::new(Stream::new(0))), n)
    })?;
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
        // The worker's stream and handles exist for the reading, so the
        // reserve holds them; the run's own are made with its backend.
        |w| {
            let _worker = DeviceStream::new(DEVICES[w]);
            fb_core_arena::device_memory(DEVICES[w])
        },
        |w| fb_core_wwr::runtime::stack_growth(DEVICES[w], KERNEL_STACK_BYTES),
        fb_core_arena::host_available()?,
    )?;
    let mut instance = planned.instantiate(&DEVICES);
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
