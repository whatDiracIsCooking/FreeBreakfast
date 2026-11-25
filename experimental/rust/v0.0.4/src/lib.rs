//! The fourth prototype of the rustflow redesign (`README.md`, `v0.0.4.html`):
//! the DF-RHF chain as steps on `num_gpus` GPUs, batched, written from
//! scratch over the C ABI. `README.md`, "The v0.0.4 prototype", has what was built and what it
//! showed.
//!
//! - [`pack`]: the closure, first fit largest-first, the lower bound.
//!   v0.0.3's, as it is.
//! - [`flow`]: steps, slots, the declared view, the two-mode DAG. v0.0.3's,
//!   with the guards handed to `run`.
//! - [`budget`]: the ledger, the `Budget` handle, the buffers, the lease,
//!   and the lock ranks.
//! - [`device`]: the device layer over the C ABI: streams, events, copies,
//!   a GPU's guard with its BLAS and solver calls, the recipes, and the
//!   kernel launches.
//! - [`host`]: the host's available memory.
//! - [`bus`]: the config, setup, the `MainBus`, the wrapper's acquire and
//!   release.
//! - [`steps`]: the seven steps of the chain.
//!
//! [`plan`] makes the DAG a config asks for; [`run`] is the whole run.

pub mod budget;
pub mod bus;
pub mod device;
pub mod flow;
pub mod host;
pub mod pack;
pub mod steps;

use std::sync::Arc;
use std::time::{Duration, Instant};

use fb_core_error_handling::Result;
use rustflow::Executor;

use crate::budget::Backing;
use crate::bus::{Config, MainBus};
use crate::flow::{Bus, Dag, Plan, StepTime};
use crate::steps::core_hamiltonian::CoreHamiltonianStep;
use crate::steps::eri::EriStep;
use crate::steps::guess::GuessStep;
use crate::steps::metric::MetricStep;
use crate::steps::orthogonalizer::OrthogonalizerStep;
use crate::steps::overlap::OverlapStep;
use crate::steps::scf::ScfStep;

/// The DAG `bus`'s config asks for. The one-electron steps, the guess and
/// the orthogonalizer come with `[basis]`; `metric` and `eri` with
/// `[eri]`; `scf` with `[scf]`. A step the config does not ask for is not
/// in the DAG, and a step that reads one is refused here, naming both.
///
/// # Errors
/// [`Dag::build`]'s refusals.
pub fn plan(bus: Arc<MainBus>) -> Result<Plan<MainBus>> {
    let config = &bus.shared().config;
    let mut dag = Dag::new();
    dag.add(OverlapStep);
    dag.add(CoreHamiltonianStep);
    dag.add(GuessStep);
    dag.add(OrthogonalizerStep);
    if config.eri.is_some() {
        dag.add(MetricStep);
        dag.add(EriStep);
    }
    if config.scf.is_some() {
        dag.add(ScfStep);
    }
    dag.build(bus)
}

/// What a whole run leaves: the plan (and through it the bus and every
/// step's `Output`), the budget table, and the times.
pub struct Run {
    /// The plan; `plan.output(ScfStep)` reads the energy.
    pub plan: Plan<MainBus>,
    /// The executor, for a replay.
    pub executor: Executor,
    /// The memory budget's table, as the final node logged it.
    pub table: String,
    /// Setup's wall time.
    pub setup: Duration,
    /// Budget mode's wall time.
    pub budget: Duration,
    /// Execute mode's wall time.
    pub execute: Duration,
    /// Each step's time in execute mode.
    pub steps: Vec<StepTime>,
}

impl Run {
    /// Execute mode again, at the same dimensions, in the same buffers.
    ///
    /// # Errors
    /// The first failing step's error.
    pub fn replay(&mut self) -> Result<()> {
        let start = Instant::now();
        self.steps = self.plan.execute(&self.executor)?;
        self.execute = start.elapsed();
        Ok(())
    }
}

/// Setup, budget mode, execute mode: the run of `config` over an executor
/// of `[resources] threads` workers.
///
/// # Errors
/// Setup's refusals, the plan's, the budget's refusal with its table, or
/// the first failing step's error.
pub fn run(config: Config) -> Result<Run> {
    let executor = Executor::new(config.resources.threads);
    let start = Instant::now();
    let bus = MainBus::setup(config, Backing::Real)?;
    let plan = plan(bus)?;
    let setup = start.elapsed();
    let start = Instant::now();
    plan.budget(&executor)?;
    let budget = start.elapsed();
    let table = plan.bus().ledger().table();
    let start = Instant::now();
    let steps = plan.execute(&executor)?;
    let execute = start.elapsed();
    Ok(Run {
        plan,
        executor,
        table,
        setup,
        budget,
        execute,
        steps,
    })
}
