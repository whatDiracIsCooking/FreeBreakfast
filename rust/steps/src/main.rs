//! `free-breakfast <config.toml>`: read the config, run it through the
//! steps ([`fb_steps::run`]: setup, budget mode, execute mode), and write
//! the run's record and products ([`fb_steps::Run::write`]). `calculation`
//! picks the steps and the products ([`fb_steps::Calculation`]). `main` is
//! the one place an error is logged, and the process exits with its kind's
//! code (`docs/diagnostics.md`, "The one boundary").
//!
//! ```sh
//! cargo run -p fb-steps --release --bin free-breakfast -- examples/rhf.toml
//! cargo run -p fb-steps --release --bin free-breakfast -- steps/examples/gly10.toml --budget-only
//! ```
//!
//! `--output DIR` (default `fbf_output`, relative to the working
//! directory) is where the run's record, `budget.txt` (the table printed
//! here) and `timing.json`, and, unless `--budget-only`, its products go.
//!
//! `--budget-only` stops after budget mode with no device touched: the
//! dimensions setup counted and the table of what the plan would allocate.
//! Every length is a count, so it is exact but for each solver call's work
//! buffer, which needs a live handle to query and is reported as zero.
//!
//! `--devices 0,1` overrides `[resources] devices` (a device named twice is
//! two logical GPUs on it), `--threads N` `[resources] threads`, and
//! `--mem-per-gpu-mib N` `[eri] mem_per_gpu`, converted to bytes.
//! `--stage-sync` sets `[scf] stage_sync`: a sync of gpu0's stream where an
//! SCF iteration would end a step, were it three.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use fb_core_budget::Backing;
use fb_core_error_handling::{Error, Status};
use fb_core_flow::Bus;
use fb_core_log::{self as log, LogLevel};
use fb_steps::{Config, EriStep, LocalizationStep, ScfStep, run};

/// The command line.
#[derive(Default)]
struct Args {
    path: String,
    output: Option<PathBuf>,
    budget_only: bool,
    devices: Option<Vec<i32>>,
    mem_per_gpu_mib: Option<f64>,
    threads: Option<usize>,
    stage_sync: bool,
}

fn args() -> Option<Args> {
    let mut out = Args::default();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--budget-only" => out.budget_only = true,
            "--stage-sync" => out.stage_sync = true,
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
            "--output" => out.output = Some(it.next()?.into()),
            _ if out.path.is_empty() => out.path = arg,
            _ => return None,
        }
    }
    (!out.path.is_empty()).then_some(out)
}

/// The usage line, an `InvalidConfig` refusal.
const USAGE: &str = "usage: free-breakfast <config.toml> [--output DIR] [--budget-only] \
                     [--devices 0,1] [--threads N] [--mem-per-gpu-mib N] [--stage-sync]";

/// The run of the command line's config, its record and products written.
fn go() -> Status {
    let args = args().ok_or_else(|| Error::config(USAGE))?;
    let mut config = Config::read(Path::new(&args.path))?;
    if let Some(devices) = &args.devices {
        config.resources.devices.clone_from(devices);
    }
    if let Some(threads) = args.threads {
        config.resources.threads = threads;
    }
    if let (Some(mib), Some(eri)) = (args.mem_per_gpu_mib, config.eri.as_mut()) {
        // Whole bytes: a fraction of a MiB names the least limit exactly.
        eri.mem_per_gpu = Some((mib * f64::from(1 << 20)).round() as u64);
    }
    if let (true, Some(scf)) = (args.stage_sync, config.scf.as_mut()) {
        scf.stage_sync = true;
    }
    let backing = if args.budget_only {
        Backing::Dry
    } else {
        Backing::Real
    };
    let (has_eri, has_scf) = (config.eri.is_some(), config.scf.is_some());
    let has_localization = config.localization.is_some();
    let run = run(config, backing)?;
    let output = args.output.unwrap_or_else(|| PathBuf::from("fbf_output"));
    run.write(&output)?;
    println!("wrote {}", output.display());
    let shared = run.plan.bus().shared();
    let dims = &shared.dims;
    println!(
        "{} AOs, {} shells, {} shell pairs, {} pair columns; {} aux functions, {} aux shells, \
         {} aux primitives; {} GPUs",
        dims.n_ao,
        dims.n_shell,
        dims.n_shell_pair,
        dims.n_col,
        dims.n_aux,
        dims.n_aux_shell,
        dims.n_aux_prim,
        shared.num_gpus()
    );
    for (step, reads) in run.plan.edges() {
        println!("{step} reads [{}]", reads.join(", "));
    }
    println!("{}", run.table);
    if args.budget_only {
        return Ok(());
    }
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
            "scf: E = {:.12} Eh in {} iterations, |e| = {:.3e}; the guess's factor of rank {} \
             of {} AOs",
            scf.energy(),
            scf.iterations(),
            scf.residual(),
            scf.guess_rank(),
            run.plan.bus().shared().dims.n_ao
        );
        let meter = scf.meter();
        let builds = u32::try_from(scf.iterations().max(1)).unwrap_or(u32::MAX);
        println!(
            "scf: the loop took {:.3?}, {:.3?} per iteration; {} stream syncs, {:.3?} blocked \
             in them",
            meter.wall,
            meter.wall / builds,
            meter.syncs,
            meter.blocked
        );
    }
    if has_localization {
        let local = run.plan.output(LocalizationStep);
        println!(
            "localization: L = {:.12} from {:.12} in {} sweeps",
            local.functional_after(),
            local.functional_before(),
            local.sweeps()
        );
    }
    Ok(())
}

fn main() -> ExitCode {
    // The logger and its panic hook: a panic is a critical line before the
    // release abort.
    log::init();
    let status = go();
    if let Err(error) = &status {
        log::log_at(LogLevel::Error, error.origin, format_args!("{error}"));
    }
    log::flush();
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
