//! `steps`: the v0.0.4 layer's MainBus and every step's `mem_budget` and
//! `run`, in one crate so the bus names every `Output` with no cycle
//! (`docs/programming_model.md`, "The MainBus and its steps"). Method code
//! stays in the per-module crates the steps call.
//!
//! Ported from the v0.0.4 prototype's `src/bus.rs` and its first five
//! steps (`../experimental/rust/v0.0.4/README.md`, "Steps and the MainBus",
//! "Lengths: counted in setup, built by the step").
//!
//! - [`Config`]: the config file, one table per step family.
//! - [`MainBus`]: setup ([`MainBus::setup`]), the unbudgeted host phase:
//!   the geometry and the bases read and placed, every length counted into
//!   [`Dims`] by a walk that builds no array ([`for_each_pair`]), each
//!   GPU's `GpuResources` made, the caps read and clamped. Then the slots.
//! - [`plan`]: the DAG a config asks for; [`run`]: setup, budget mode and
//!   execute mode over the run's one executor, and [`Run::replay`], execute
//!   mode again in the same buffers.
//!
//! ```text
//! overlap ──> orthogonalizer ─┐
//! core_hamiltonian ───────────┤
//! metric ──> eri ─────────────┼──> scf    (eri with [eri], scf with [scf])
//! guess ──────────────────────┘
//! ```
//!
//! `scf` reads `overlap` too.
//!
//! | Step | Runs on | Locks | `Output` | Scratch |
//! |---|---|---|---|---|
//! | [`OverlapStep`] | host, then gpu0 | none | S on gpu0 | the pinned staging |
//! | [`CoreHamiltonianStep`] | host, then gpu0 | none | H = T + V on gpu0 | the pinned staging |
//! | [`MetricStep`] | host, then every GPU | none | (P\|Q) on every GPU | the pinned staging, uploaded to every GPU under one token |
//! | [`GuessStep`] | host, then gpu0 | none | the SAD density on gpu0 | the pinned staging |
//! | [`OrthogonalizerStep`] | gpu0 | gpu0 | X = S^(-1/2) on gpu0 | the eigenvectors, a scaled copy, syevd's work and flag |
//! | [`EriStep`] | every GPU, two ring streams each | the [`EriManager`]'s, every GPU's | the [`EriManager`]: L on every GPU, B on the host heap, each GPU's ERI buffer and staging | the shell and pair tables, the aux records, each class's records, the Cholesky tile and flag |
//! | [`ScfStep`] | every GPU, its loop on gpu0 | the [`EriManager`]'s, every GPU's | C, the orbital energies, D and the energy, on gpu0 and the host | on gpu0: J, K, F, the DIIS ring, syevd's work; the turn's pinned reads |
//!
//! The host steps build through the host integral crates inside
//! [`Shared::host_pool`], the one pool beside the executor, started in setup
//! with `[resources] host_threads` threads (by default the host's
//! parallelism less the executor's `threads`, floored at one) and joined
//! when the bus drops; each writes its triangle into the step's pinned
//! staging, so no `run` allocates one of its own (`docs/programming_model.md`,
//! "CPU workers"; `docs/memory.md`, "The budget").
//!
//! A plan with no device behind it is budgeted exactly, every length a
//! count, syevd's work buffer as zero:
//!
//! ```
//! use fb_core_budget::Backing;
//! use fb_core_flow::{Bus, Executor};
//! use fb_steps::{Config, MainBus, plan};
//!
//! let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data"));
//! let text = "[molecule]\nxyz = \"h2o.xyz\"\n[basis]\nname = \"cc-pvdz\"\n";
//! let config = Config::parse(text, dir).unwrap();
//! let plan = plan(MainBus::setup(config, Backing::Dry).unwrap()).unwrap();
//! plan.budget(&Executor::new(2)).unwrap();
//! let n = plan.bus().shared().dims.n_ao as u64;
//! assert_eq!(n, 24);
//! // S, H, the guess and X on gpu0, each n × n doubles, and their scratch.
//! assert!(plan.bus().ledger().device_total(0) > 4 * n * n * 8);
//! ```

#![forbid(unsafe_code)]

mod bus;
mod steps;

use std::sync::Arc;
use std::time::{Duration, Instant};

use fb_core_budget::Backing;
use fb_core_error_handling::Result;
use fb_core_flow::{Bus, Dag, Executor, Plan, StepTime};

pub use bus::{
    BasisTable, ClassDim, Config, Dims, EriTable, MainBus, MoleculeTable, ResourcesTable, ScfTable,
    Shared, for_each_pair,
};
pub use steps::core_hamiltonian::{CoreHamiltonianOutput, CoreHamiltonianStep};
pub use steps::eri::{DEFAULT_MEM_PER_GPU, Density, EriManager, EriStep};
pub use steps::guess::{GuessOutput, GuessStep};
pub use steps::metric::{MetricOutput, MetricStep};
pub use steps::orthogonalizer::{EIGENVALUE_FLOOR, OrthogonalizerOutput, OrthogonalizerStep};
pub use steps::overlap::{OverlapOutput, OverlapStep};
pub use steps::scf::{LoopMeter, ScfOutput, ScfStep};

/// The DAG `bus`'s config asks for: the one-electron steps, the guess and
/// the orthogonalizer with `[basis]`, the metric and the ERI with `[eri]`,
/// the SCF with `[scf]`. A step that reads one the config leaves out is
/// refused, naming both.
///
/// # Errors
/// [`Dag::build`]'s refusals.
pub fn plan(bus: Arc<MainBus>) -> Result<Plan<MainBus>> {
    let mut dag = Dag::new();
    dag.add(OverlapStep)
        .add(CoreHamiltonianStep)
        .add(GuessStep)
        .add(OrthogonalizerStep);
    if bus.shared().config.eri.is_some() {
        dag.add(MetricStep).add(EriStep);
    }
    if bus.shared().config.scf.is_some() {
        dag.add(ScfStep);
    }
    dag.build(bus)
}

/// What a whole run leaves: the plan (and through it the bus and every
/// step's `Output`), the executor, the budget table, and the times.
pub struct Run {
    /// The plan; `plan.output(ScfStep)` reads the energy.
    pub plan: Plan<MainBus>,
    /// The memory budget's table.
    pub table: String,
    /// Setup's wall time.
    pub setup: Duration,
    /// Budget mode's wall time.
    pub budget: Duration,
    /// Execute mode's wall time; zero for a dry run.
    pub execute: Duration,
    /// Each step's time in the last execute mode; empty for a dry run.
    pub steps: Vec<StepTime>,
    executor: Executor,
}

impl Run {
    /// Execute mode again, at the same dimensions, in the same buffers:
    /// v0.0.2's reset mode, each step's `Output` carrying over what it
    /// keeps (the SCF's orbitals) and nothing in scratch.
    ///
    /// # Errors
    /// The first failing step's error; a dry run's refusal.
    pub fn replay(&mut self) -> Result<()> {
        let start = Instant::now();
        self.steps = self.plan.execute(&self.executor)?;
        self.execute = start.elapsed();
        Ok(())
    }
}

/// The run of `config`: setup on `backing`, budget mode, and, unless the
/// backing is [`Backing::Dry`], execute mode, over the run's one executor
/// of `[resources] threads` workers, made here and nowhere else in the
/// layer (`docs/programming_model.md`, "CPU workers").
///
/// A dry run is budgeted and touches no device:
///
/// ```
/// use fb_core_budget::Backing;
/// use fb_steps::{Config, run};
///
/// let dir = std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data"));
/// let text = "[molecule]\nxyz = \"h2o.xyz\"\n[basis]\nname = \"cc-pvdz\"\n\
///             [eri]\naux_basis = \"cc-pvdz-jkfit\"\n[scf]\n";
/// let run = run(Config::parse(text, dir).unwrap(), Backing::Dry).unwrap();
/// assert!(run.table.contains("scf"));
/// assert!(run.steps.is_empty());
/// ```
///
/// # Errors
/// Setup's refusals, the plan's, the budget's refusal with its table, or
/// the first failing step's error.
pub fn run(config: Config, backing: Backing) -> Result<Run> {
    let executor = Executor::new(config.resources.threads);
    let start = Instant::now();
    let plan = plan(MainBus::setup(config, backing)?)?;
    let setup = start.elapsed();
    let start = Instant::now();
    plan.budget(&executor)?;
    let budget = start.elapsed();
    let table = plan.bus().ledger().table();
    let mut run = Run {
        plan,
        table,
        setup,
        budget,
        execute: Duration::ZERO,
        steps: Vec::new(),
        executor,
    };
    if backing != Backing::Dry {
        run.replay()?;
    }
    Ok(run)
}
