//! A task: the two functions a node of a [`Dag`](super::Dag) is; and a
//! loop's decision, the node that ends each iteration.

use std::ops::ControlFlow;

use fb_core_arena::{Bufs, Device, HostKind, Layout, Pinned};
use fb_core_error_handling::{Result, Status};
use fb_core_thread_pool::{DeviceStream, Lane};

/// A unit of work with its own rows and no internal host sync
/// (`docs/programming_model.md`, "A task declares"): what it holds, as
/// [`Task::layout`], and what it computes, as [`Task::run`], which reaches
/// memory only through the rows its layout declared.
///
/// The task's value is its dimensions. The docs write `layout(dims)`; here
/// `&self` is `dims`, with the [`RowId`](fb_core_arena::RowId)s of the
/// outputs it reads among its fields, handed to it by whoever built the
/// [`Dag`](super::Dag) from the rows an earlier [`Dag::add`](super::Dag::add)
/// returned. An output's bytes are declared once, by its producer
/// (`docs/memory.md`, "The table").
///
/// `H` is the host kind its host rows live in and `S` the stream a GPU task
/// enqueues on. The defaults are the production ones, so a real task is
/// `impl Task for Fock`; the device-free tests use a heap kind and
/// [`HostOnly`](fb_core_thread_pool::HostOnly)'s `()` stream.
///
/// A task allocates nothing, since it is handed no allocator; a row it did
/// not declare panics when asked for ([`Bufs`]). A GPU task enqueues on its
/// lane's stream and returns, and never touches a pinned row from the host
/// ([`Bufs`], "A stream node does not touch a pinned row from the host").
pub trait Task<H: HostKind = Pinned, S = DeviceStream>: Send + Sync {
    /// The rows it uses, as `layout` returned them: a struct of `RowId`s,
    /// usually `#[derive(Clone, Copy)]`. [`Dag::add`](super::Dag::add)
    /// hands a copy back to the caller, and passes one to every `run`.
    type Rows: Clone + Send + Sync;

    /// Whether it runs on its worker's stream (true) or the CPU pool
    /// (false). A CPU task holds no device row.
    const GPU: bool;

    /// Declare every row: its scratch and outputs, on the device and on the
    /// host, and each earlier output it reads. Called once, at planning.
    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, H>) -> Self::Rows;

    /// Its dimensions by name, such as `[("n_ao", 24)]`: what its entry in
    /// the timing record carries beside its name, and the nightly keys its
    /// times by ([`TaskTime::dims`](fb_core_thread_pool::TaskTime::dims)).
    /// Called once, at planning; none by default.
    fn dims(&self) -> Vec<(&'static str, usize)> {
        Vec::new()
    }

    /// Compute, through the rows `layout` returned. Called once per run.
    ///
    /// # Errors
    /// Whatever the work fails with; the orchestrator adds `name@worker`
    /// and skips every descendant.
    fn run(
        &self,
        rows: &Self::Rows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, H>,
        lane: Lane<'_, S>,
    ) -> Status;
}

/// A loop's decision (`docs/programming_model.md`, "Plan once, run
/// many"): the CPU node [`Dag::add_loop`](super::Dag::add_loop) adds after
/// the loop's body, run once per iteration when the whole body has
/// finished and its GPU work has completed, whose answer replays the body
/// or ends the loop. Like a [`Task`], it declares its rows, here host rows
/// alone, and reaches only them: what the body copied to the host for it
/// (an energy, a residual's norm) as reads, and what it keeps from one
/// iteration to the next (the last energy) as a row the loop carries
/// ([`Layout::write`]), since nothing outside its rows survives a run.
///
/// It is a CPU node, so its layout is handed no device table: a decision
/// that would read a device row does not build.
///
/// ```compile_fail,E0053
/// use std::ops::ControlFlow;
/// use fb_core_arena::{Bufs, Device, Layout, Pinned};
/// use fb_core_error_handling::Result;
/// use fb_core_planner::Decision;
///
/// struct ReadsTheDevice;
///
/// impl Decision for ReadsTheDevice {
///     type Rows = ();
///     fn layout(&self, _: &mut Layout<'_, Device>) {}
///     fn decide(&self, _: &(), _: &Bufs<'_, Pinned>, _: usize) -> Result<ControlFlow<()>> {
///         Ok(ControlFlow::Break(()))
///     }
/// }
/// ```
pub trait Decision<H: HostKind = Pinned>: Send + Sync {
    /// The rows it uses, as `layout` returned them; a copy goes back to
    /// the caller of [`Dag::add_loop`](super::Dag::add_loop).
    type Rows: Clone + Send + Sync;

    /// Declare every host row it holds or reads. Called once, at planning.
    fn layout(&self, host: &mut Layout<'_, H>) -> Self::Rows;

    /// Whether to run the body again (`Continue`) or end the loop
    /// (`Break`), after iteration `iteration`, from 0. Called once per
    /// iteration.
    ///
    /// # Errors
    /// Whatever the decision fails with, such as a diverged iterate; the
    /// orchestrator adds `name@cpu`, ends the loop and skips what follows
    /// it. A `Continue` from the loop's last allowed iteration fails the
    /// run with
    /// [`MaxIterationsExceeded`](fb_core_error_handling::ErrorKind::MaxIterationsExceeded).
    fn decide(
        &self,
        rows: &Self::Rows,
        host: &Bufs<'_, H>,
        iteration: usize,
    ) -> Result<ControlFlow<()>>;
}
