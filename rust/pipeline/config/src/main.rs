//! `free-breakfast <config.toml>`: read the config, plan it, admit it
//! against the `[memory]` caps, run it once on device 0, and write its products and `timing.json` into its output
//! directory (`fb_pipeline_config`). `main` is the one place an error is
//! logged, and the process exits with its kind's code
//! (`docs/diagnostics.md`, "The one boundary").

#![forbid(unsafe_code)]

use std::path::Path;
use std::process::ExitCode;
use std::rc::Rc;

use fb_core_error_handling::{Error, Result, Status};
use fb_core_log::{self as log, LogLevel};
use fb_core_thread_pool::{DeviceStream, default_n_cpu};
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::solver::SolverHandle;
use fb_pipeline_config::{KERNEL_STACK_BYTES, admit_plan, plan, read_config};
use fb_recipes_diagonalize::syevd_lwork;

/// The devices the plan's GPU workers run on, worker `w` on `DEVICES[w]`.
const DEVICES: [i32; 1] = [0];

/// The node's `MemAvailable`, in bytes, from `/proc/meminfo`: what the
/// workers' pinned caps together must fit.
fn read_mem_available() -> Result<usize> {
    let meminfo = std::fs::read_to_string("/proc/meminfo")
        .map_err(|e| Error::io(format!("cannot read '/proc/meminfo': {e}")))?;
    let line = meminfo
        .lines()
        .find(|l| l.starts_with("MemAvailable:"))
        .ok_or_else(|| Error::io("no MemAvailable in '/proc/meminfo'"))?;
    match line["MemAvailable:".len()..]
        .split_whitespace()
        .collect::<Vec<_>>()[..]
    {
        [kib, "kB"] => kib
            .parse::<usize>()
            .ok()
            .and_then(|kib| kib.checked_mul(1024))
            .ok_or_else(|| Error::io(format!("cannot parse '{line}' in '/proc/meminfo'"))),
        _ => Err(Error::io(format!(
            "cannot parse '{line}' in '/proc/meminfo'"
        ))),
    }
}

/// Read, plan (syevd's work asked of a solver handle on device 0, if the
/// plan asks), admit every worker's tables against the `[memory]` caps
/// (each device's reserve measured with its worker's stream and handles
/// made, and the kernels' stack read after it),
/// make the output directory, run, and write `timing.json`. Everything a
/// config can get wrong is refused before a device is touched, and a plan
/// too large for its caps or its card before a block is made.
fn run(args: &[String]) -> Status {
    let [_, config] = args else {
        let program = args.first().map_or("free-breakfast", String::as_str);
        return Err(Error::config(format!("usage: {program} <config.toml>")));
    };
    let config = read_config(Path::new(config))?;
    let planned = plan(&config, |n| {
        syevd_lwork(&SolverHandle::new(Rc::new(Stream::new(0))), n)
    })?;
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
        read_mem_available()?,
    )?;
    std::fs::create_dir_all(&config.output).map_err(|e| {
        Error::io(format!(
            "cannot make the output directory '{}': {e}",
            config.output.display()
        ))
    })?;
    let mut instance = planned.instantiate(&DEVICES);
    let mut backend = instance.backend();
    let record = instance.run(&mut backend, default_n_cpu(1));
    // Written whether or not the run failed: the record holds the error,
    // and the times of what ran before it.
    let timing = config.output.join("timing.json");
    std::fs::write(&timing, record.to_json())
        .map_err(|e| Error::io(format!("cannot write '{}': {e}", timing.display())))?;
    record.result?;
    log::flush();
    println!("wrote:             {}", config.output.display());
    Ok(())
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
