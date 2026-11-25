//! Run a config through the prototype: setup, budget mode, execute mode.
//!
//! ```sh
//! cargo run --release --example run -- examples/h2o.toml
//! ```
//!
//! `--budget-only` stops after budget mode, with no device touched: the
//! table of what the plan would allocate, exact, since every length is a
//! count. With no card to read, a defaulted device cap is none, so a step's
//! default limit holds one batch per class.

use std::path::Path;
use std::process::ExitCode;

use fb_core_error_handling::Result;
use fb_v03::budget::Backing;
use fb_v03::bus::{Config, MainBus};
use fb_v03::flow::Bus;
use fb_v03::steps::eri::EriStep;
use fb_v03::steps::jk::JkStep;
use rustflow::Executor;

fn run(path: &Path, budget_only: bool) -> Result<()> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| fb_core_error_handling::Error::io(format!("{}: {e}", path.display())))?;
    let config = Config::parse(&text, path.parent().unwrap_or(Path::new(".")))?;
    if budget_only {
        let executor = Executor::new(config.resources.threads);
        let plan = fb_v03::plan(MainBus::setup(config, Backing::Dry)?)?;
        for (step, reads) in plan.edges() {
            println!("{step} reads [{}]", reads.join(", "));
        }
        plan.budget(&executor)?;
        println!("{}", plan.bus().ledger().table());
        return Ok(());
    }
    let (has_eri, has_jk) = (config.eri.is_some(), config.jk.is_some());
    let run = fb_v03::run(config)?;
    println!("{}", run.table);
    println!(
        "setup {:.3?}, budget mode {:.3?}, execute mode {:.3?}",
        run.setup, run.budget, run.execute
    );
    for step in &run.steps {
        println!(
            "  {:<8} began {:>10.3?}  ended {:>10.3?}  run {:>10.3?}  wrapper sync {:>10.3?}",
            step.name, step.began, step.ended, step.run, step.sync
        );
    }
    for held in run.plan.bus().shared().held() {
        println!(
            "  lock: {:<4} card {} from {:>10.3?} to {:>10.3?}",
            held.step, held.card, held.locked, held.released
        );
    }
    if has_eri {
        let eri = run.plan.output(EriStep);
        println!(
            "eri: B is {} aux x {} pair columns on the host, built in {} batches of at most {} \
             Cartesian columns",
            eri.n_aux(),
            eri.n_col(),
            eri.batches(),
            eri.width()
        );
    }
    if has_jk {
        let jk = run.plan.output(JkStep);
        let trace = |m: &[f64]| {
            let n = m.len().isqrt();
            (0..n).map(|i| m[i * (n + 1)]).sum::<f64>()
        };
        println!(
            "jk: {} batches of at most {} aux functions; tr J = {:.12}, tr K = {:.12}",
            jk.batches(),
            jk.width(),
            trace(jk.j_host()),
            trace(jk.k_host())
        );
    }
    Ok(())
}

fn main() -> ExitCode {
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
