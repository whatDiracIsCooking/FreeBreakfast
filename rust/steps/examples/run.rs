//! Run a config through the steps: setup, budget mode, execute mode.
//!
//! ```sh
//! cargo run -p fb-steps --example run -- steps/examples/gly10.toml
//! cargo run -p fb-steps --example run -- steps/examples/gly10.toml --budget-only
//! ```
//!
//! `--budget-only` stops after budget mode with no device touched: the
//! dimensions setup counted and the table of what the plan would allocate.
//! Every length is a count, so it is exact but for each solver call's work
//! buffer, which needs a live handle to query and is reported as zero.

use std::path::Path;
use std::process::ExitCode;

use fb_core_budget::Backing;
use fb_core_error_handling::Result;
use fb_core_flow::{Bus, Executor};
use fb_steps::{Config, MainBus, plan};

fn run(path: &Path, budget_only: bool) -> Result<()> {
    let config = Config::read(path)?;
    let executor = Executor::new(config.resources.threads);
    let backing = if budget_only {
        Backing::Dry
    } else {
        Backing::Real
    };
    let plan = plan(MainBus::setup(config, backing)?)?;
    let dims = &plan.bus().shared().dims;
    println!(
        "{} AOs, {} shells, {} shell pairs, {} pair columns; {} aux functions, {} aux shells, \
         {} aux primitives",
        dims.n_ao,
        dims.n_shell,
        dims.n_shell_pair,
        dims.n_col,
        dims.n_aux,
        dims.n_aux_shell,
        dims.n_aux_prim
    );
    for (step, reads) in plan.edges() {
        println!("{step} reads [{}]", reads.join(", "));
    }
    plan.budget(&executor)?;
    println!("{}", plan.bus().ledger().table());
    if budget_only {
        return Ok(());
    }
    for step in plan.execute(&executor)? {
        println!(
            "  {:<17} began {:>10.3?}  ended {:>10.3?}  run {:>10.3?}  wrapper sync {:>10.3?}",
            step.name, step.began, step.ended, step.run, step.sync
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let budget_only = args.iter().any(|a| a == "--budget-only");
    let paths: Vec<&String> = args.iter().filter(|a| *a != "--budget-only").collect();
    let [path] = paths[..] else {
        eprintln!("usage: run <config.toml> [--budget-only]");
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
