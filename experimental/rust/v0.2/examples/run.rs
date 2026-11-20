//! Run a config through the prototype: setup, budget mode, execute mode.
//!
//! ```sh
//! cargo run --release --example run -- examples/h2o.toml
//! ```
//!
//! `--budget-only` stops after budget mode, with no device touched: the
//! table of what the plan would allocate (less the library workspace, whose
//! size only a live handle knows).

use std::path::Path;
use std::process::ExitCode;

use fb_core_error_handling::Result;
use fb_v02::budget::Backing;
use fb_v02::bus::{Config, MainBus};
use fb_v02::flow::Bus;
use fb_v02::steps::scf::ScfStep;
use rustflow::Executor;

struct Printer;
impl log::Log for Printer {
    fn enabled(&self, m: &log::Metadata<'_>) -> bool {
        m.level() <= log::Level::Info
    }
    fn log(&self, record: &log::Record<'_>) {
        println!("{}", record.args());
    }
    fn flush(&self) {}
}

fn run(path: &Path, budget_only: bool) -> Result<()> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| fb_core_error_handling::Error::io(format!("{}: {e}", path.display())))?;
    let config = Config::parse(&text, path.parent().unwrap_or(Path::new(".")))?;
    if budget_only {
        let executor = Executor::new(config.resources.threads);
        let plan = fb_v02::plan(MainBus::setup(config, Backing::Dry)?)?;
        print!("{}", plan.dot());
        plan.budget(&executor)?;
        return Ok(());
    }
    let has_scf = config.scf.is_some();
    let run = fb_v02::run(config)?;
    println!(
        "setup {:?}, budget mode {:?}, execute mode {:?}",
        run.setup, run.budget, run.execute
    );
    for step in &run.steps {
        println!(
            "  {:<18} trips {:>3}  began {:>10.3?}  ended {:>10.3?}",
            step.name, step.trips, step.began, step.ended
        );
    }
    println!("dims: {:?}", run.plan.bus().shared().dims);
    if has_scf {
        let scf = run.plan.output(ScfStep);
        println!(
            "E = {:.12} hartree, {} iterations, |e| = {:.3e}",
            scf.energy(),
            scf.iterations(),
            scf.residual()
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    let _ = log::set_logger(&Printer);
    log::set_max_level(log::LevelFilter::Info);
    let args: Vec<String> = std::env::args().skip(1).collect();
    let budget_only = args.iter().any(|a| a == "--budget-only");
    let Some(path) = args.iter().find(|a| !a.starts_with("--")) else {
        eprintln!("usage: run [--budget-only] <config.toml>");
        return ExitCode::from(2);
    };
    match run(Path::new(path), budget_only) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
