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
//! - [`plan`]: the DAG a config asks for.
//!
//! ```text
//! overlap ──> orthogonalizer
//! core_hamiltonian
//! metric            (with [eri])
//! guess
//! ```
//!
//! | Step | Runs on | Locks | `Output` | Scratch |
//! |---|---|---|---|---|
//! | [`OverlapStep`] | host, then gpu0 | none | S on gpu0 | the pinned staging |
//! | [`CoreHamiltonianStep`] | host, then gpu0 | none | H = T + V on gpu0 | the pinned staging |
//! | [`MetricStep`] | host, then every GPU | none | (P\|Q) on every GPU | a pinned staging per GPU |
//! | [`GuessStep`] | host, then gpu0 | none | the SAD density on gpu0 | the pinned staging |
//! | [`OrthogonalizerStep`] | gpu0 | gpu0 | X = S^(-1/2) on gpu0 | the eigenvectors, a scaled copy, syevd's work and flag |
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

use fb_core_error_handling::Result;
use fb_core_flow::{Bus, Dag, Plan};

pub use bus::{
    BasisTable, ClassDim, Config, Dims, EriTable, MainBus, MoleculeTable, ResourcesTable, Shared,
    for_each_pair,
};
pub use steps::core_hamiltonian::{CoreHamiltonianOutput, CoreHamiltonianStep};
pub use steps::guess::{GuessOutput, GuessStep};
pub use steps::metric::{MetricOutput, MetricStep};
pub use steps::orthogonalizer::{EIGENVALUE_FLOOR, OrthogonalizerOutput, OrthogonalizerStep};
pub use steps::overlap::{OverlapOutput, OverlapStep};

/// The DAG `bus`'s config asks for: the one-electron steps, the guess and
/// the orthogonalizer with `[basis]`, the metric with `[eri]`.
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
        dag.add(MetricStep);
    }
    dag.build(bus)
}
