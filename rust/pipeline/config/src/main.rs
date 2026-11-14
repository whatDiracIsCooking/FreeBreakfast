//! `free-breakfast <config.toml>`: read the config, plan it, run it once on
//! device 0, and write its products and `timing.json` into its output
//! directory (`fb_pipeline_config`). `main` is the one place an error is
//! logged, and the process exits with its kind's code
//! (`docs/diagnostics.md`, "The one boundary").

#![forbid(unsafe_code)]

use std::path::Path;
use std::process::ExitCode;

use fb_core_error_handling::{Error, Status};
use fb_core_log::{self as log, LogLevel};
use fb_core_thread_pool::{self as thread_pool, default_n_cpu};
use fb_pipeline_config::{plan, read_config};

/// Read, plan, make the output directory, run, and write `timing.json`.
/// Everything a config can get wrong is refused before a device is touched.
fn run(args: &[String]) -> Status {
    let [_, config] = args else {
        let program = args.first().map_or("free-breakfast", String::as_str);
        return Err(Error::config(format!("usage: {program} <config.toml>")));
    };
    let config = read_config(Path::new(config))?;
    let planned = plan(&config)?;
    std::fs::create_dir_all(&config.output).map_err(|e| {
        Error::io(format!(
            "cannot make the output directory '{}': {e}",
            config.output.display()
        ))
    })?;
    let mut instance = planned.instantiate(&[0]);
    let mut backend = thread_pool::Device::new(&[0], instance.plan());
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
    log::init();
    let args: Vec<String> = std::env::args().collect();
    let status = run(&args);
    if let Err(error) = &status {
        log::log_at(LogLevel::Error, error.origin, format_args!("{error}"));
    }
    println!(
        "completed: {} errors, {} warnings",
        u8::from(status.is_err()),
        log::warning_count()
    );
    match status {
        Ok(()) => ExitCode::SUCCESS,
        // The codes are 1 to 8.
        Err(error) => ExitCode::from(error.kind.exit_code() as u8),
    }
}
