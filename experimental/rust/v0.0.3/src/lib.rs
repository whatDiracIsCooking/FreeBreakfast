//! The second prototype of the rustflow redesign (`experimental/rust/v0.0.4/v0.0.4.html`),
//! covering items 1 and 2 of its order of work: the device-free core, with
//! step scratch declared per step and packed by the DAG's closure, and
//! batched steps on `num_gpus` cards under one lock per card. `README.md`
//! has what it showed.
//!
//! - [`pack`]: the closure, first fit largest-first, the lower bound.
//! - [`budget`]: the ledger, the `Budget` handle, the buffers, the lease.
//! - [`flow`]: steps, slots, the declared view, the two-mode DAG. No device.
//! - [`device`]: a step's stream per card, a card's lock, the bind.
//! - [`bus`]: the config, setup, and the `MainBus`.
//! - [`steps`]: `records`, `guess`, `eri` and `jk`.
//!
//! [`plan`] makes the DAG a config asks for; [`run`] is the whole run.

pub mod budget;
pub mod bus;
pub mod device;
pub mod flow;
pub mod pack;
pub mod steps;

use std::sync::Arc;
use std::time::{Duration, Instant};

use fb_core_error_handling::Result;
use rustflow::Executor;

use crate::budget::Backing;
use crate::bus::{Config, MainBus};
use crate::flow::{Bus, Dag, Plan, StepTime};
use crate::steps::eri::EriStep;
use crate::steps::guess::GuessStep;
use crate::steps::jk::JkStep;
use crate::steps::records::RecordsStep;

/// The DAG `bus`'s config asks for. `records` comes with `[basis]`; `eri`,
/// `guess` and `jk` each with its table, so a step the config does not ask
/// for is not in the DAG, and a step that reads one is refused here,
/// naming both.
///
/// # Errors
/// [`Dag::build`]'s refusals.
pub fn plan(bus: Arc<MainBus>) -> Result<Plan<MainBus>> {
    let config = &bus.shared().config;
    let mut dag = Dag::new();
    dag.add(RecordsStep);
    if config.guess.is_some() {
        dag.add(GuessStep);
    }
    if config.eri.is_some() {
        dag.add(EriStep);
    }
    if config.jk.is_some() {
        dag.add(JkStep);
    }
    dag.build(bus)
}

/// What a whole run leaves: the plan (and through it the bus and every
/// step's `Output`), the budget table, and the times.
pub struct Run {
    /// The plan; `plan.output(JkStep)` reads J and K.
    pub plan: Plan<MainBus>,
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
        table,
        setup,
        budget,
        execute,
        steps,
    })
}
