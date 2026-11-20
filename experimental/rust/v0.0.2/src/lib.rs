//! A prototype of the rustflow redesign (`.temp/rust-redesign`): a
//! calculation as a DAG of steps over one shared `MainBus`, each step one
//! function that counts and allocates (`mem_budget`) and one that runs, the
//! same DAG run in two modes over one rustflow `Executor`. It carries the
//! DF-RHF of a closed-shell molecule on one card, every step in its crudest
//! form. `README.md` has what it showed.
//!
//! - [`budget`]: the ledger, the `Budget` handle, the buffers it hands back.
//! - [`flow`]: steps, slots, the declared view, the two-mode DAG. No device.
//! - [`device`]: the step's `ForkStream`, the device's `DeviceResources`.
//! - [`bus`]: the config, setup, and the `MainBus`.
//! - [`steps`]: the eight steps.
//!
//! [`plan`] makes the DAG a config asks for; [`run`] is the whole run.

pub mod budget;
pub mod bus;
pub mod device;
pub mod flow;
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
use crate::steps::harmonics::HarmonicsStep;
use crate::steps::orthogonalizer::OrthogonalizerStep;
use crate::steps::overlap::OverlapStep;
use crate::steps::records::RecordsStep;
use crate::steps::scf::ScfStep;

/// The DAG `bus`'s config asks for. The one-electron steps come with
/// `[basis]`; `eri`, `guess` and `scf` each with its table, so a step the
/// config does not ask for is not in the DAG, and a step that reads one is
/// refused here, naming both.
///
/// # Errors
/// [`Dag::build`]'s refusals.
pub fn plan(bus: Arc<MainBus>) -> Result<Plan<MainBus>> {
    let config = &bus.shared().config;
    let mut dag = Dag::new();
    dag.add(HarmonicsStep)
        .add(RecordsStep)
        .add(OverlapStep)
        .add(CoreHamiltonianStep)
        .add(OrthogonalizerStep);
    if config.eri.is_some() {
        dag.add(EriStep);
    }
    if config.guess.is_some() {
        dag.add(GuessStep);
    }
    if config.scf.is_some() {
        dag.add(ScfStep);
    }
    dag.build(bus)
}

/// What a whole run leaves: the plan (and through it the bus and every
/// step's `Output`), the budget table, and the times.
pub struct Run {
    /// The plan; `plan.output(ScfStep)` reads the SCF's products.
    pub plan: Plan<MainBus>,
    /// The memory budget's table, as the final node logged it.
    pub table: String,
    /// Setup's wall time: the parse and the handles.
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
