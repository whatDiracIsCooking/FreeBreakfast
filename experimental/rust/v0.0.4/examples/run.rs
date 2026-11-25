//! Run a config through the prototype: setup, budget mode, execute mode.
//!
//! ```sh
//! cargo run --release --example run -- examples/h2o.toml
//! ```
//!
//! `--budget-only` stops after budget mode, with no device touched: the
//! table of what the plan would allocate. Every length is a count, so it
//! is exact but for each solver call's work buffer, which needs a live
//! handle to query and is reported as zero.
//!
//! `--devices 0,1` overrides `[resources] devices`, and `--mem-per-gpu-mib
//! N` the step limit of `[eri]`.

use std::path::Path;
use std::process::ExitCode;

use fb_core_error_handling::{Error, Result};
use fb_v04::budget::Backing;
use fb_v04::bus::{Config, MainBus};
use fb_v04::flow::Bus;
use fb_v04::steps::eri::EriStep;
use fb_v04::steps::scf::ScfStep;
use rustflow::Executor;

struct Args {
    path: String,
    budget_only: bool,
    devices: Option<Vec<i32>>,
    mem_per_gpu_mib: Option<f64>,
    threads: Option<usize>,
}

fn args() -> Option<Args> {
    let mut out = Args {
        path: String::new(),
        budget_only: false,
        devices: None,
        mem_per_gpu_mib: None,
        threads: None,
    };
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--budget-only" => out.budget_only = true,
            "--devices" => {
                out.devices = Some(
                    it.next()?
                        .split(',')
                        .map(|d| d.trim().parse().ok())
                        .collect::<Option<_>>()?,
                );
            }
            "--mem-per-gpu-mib" => out.mem_per_gpu_mib = Some(it.next()?.parse().ok()?),
            "--threads" => out.threads = Some(it.next()?.parse().ok()?),
            _ if out.path.is_empty() => out.path = arg,
            _ => return None,
        }
    }
    (!out.path.is_empty()).then_some(out)
}

fn run(args: &Args) -> Result<()> {
    let path = Path::new(&args.path);
    let text =
        std::fs::read_to_string(path).map_err(|e| Error::io(format!("{}: {e}", path.display())))?;
    let mut config = Config::parse(&text, path.parent().unwrap_or(Path::new(".")))?;
    if let Some(devices) = &args.devices {
        config.resources.devices.clone_from(devices);
    }
    if let Some(threads) = args.threads {
        config.resources.threads = threads;
    }
    if let (Some(mib), Some(eri)) = (args.mem_per_gpu_mib, config.eri.as_mut()) {
        eri.mem_per_gpu_mib = Some(mib);
    }
    if args.budget_only {
        let executor = Executor::new(config.resources.threads);
        let plan = fb_v04::plan(MainBus::setup(config, Backing::Dry)?)?;
        for (step, reads) in plan.edges() {
            println!("{step} reads [{}]", reads.join(", "));
        }
        plan.budget(&executor)?;
        println!("{}", plan.bus().ledger().table());
        return Ok(());
    }
    let (has_eri, has_scf) = (config.eri.is_some(), config.scf.is_some());
    let run = fb_v04::run(config)?;
    println!("{}", run.table);
    let dims = &run.plan.bus().shared().dims;
    println!(
        "{} AOs, {} aux functions, {} pair columns, {} GPUs",
        dims.n_ao,
        dims.n_aux,
        dims.n_col,
        run.plan.bus().shared().num_gpus()
    );
    println!(
        "setup {:.3?}, budget mode {:.3?}, execute mode {:.3?}",
        run.setup, run.budget, run.execute
    );
    for step in &run.steps {
        println!(
            "  {:<17} began {:>10.3?}  ended {:>10.3?}  run {:>10.3?}  wrapper sync {:>10.3?}",
            step.name, step.began, step.ended, step.run, step.sync
        );
    }
    for a in run.plan.bus().shared().acquisitions() {
        println!(
            "  lock: {:<15} {:<14} asked {:>10.3?}  held {:>10.3?} to {:>10.3?}{}",
            a.step,
            format!("{:?}", a.lock),
            a.asked,
            a.locked,
            a.released,
            if a.waited { "  (waited)" } else { "" }
        );
    }
    if has_eri {
        let eri = run.plan.output(EriStep);
        println!(
            "eri: B is {} aux x {} pair columns on the host, built in {} pair batches of at \
             most {} columns; J/K passes in aux batches of {}",
            eri.n_aux(),
            eri.n_col(),
            eri.batches(),
            eri.width(),
            eri.jk_width()
        );
    }
    if has_scf {
        let scf = run.plan.output(ScfStep);
        println!(
            "scf: E = {:.12} Eh in {} iterations, |e| = {:.3e}",
            scf.energy(),
            scf.iterations(),
            scf.residual()
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    let Some(args) = args() else {
        eprintln!(
            "usage: run <config.toml> [--budget-only] [--devices 0,1] [--mem-per-gpu-mib N] \
             [--threads N]"
        );
        return ExitCode::from(2);
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}
