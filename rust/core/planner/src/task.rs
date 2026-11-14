//! A task, of either sort: the two functions a node of a [`Dag`](super::Dag)
//! is, on a stream or the CPU pool; and a loop's decision, the node that
//! ends each iteration.

use std::ops::ControlFlow;

use fb_core_arena::{Bufs, CopyOnly, Device, HostKind, Layout, Pinned};
use fb_core_error_handling::{Result, Status};
use fb_core_thread_pool::DeviceStream;

/// A GPU task: a unit of work with its own rows and no internal host sync
/// (`docs/programming_model.md`, "A task declares"), run on its worker's
/// stream. What it holds is [`GpuTask::layout`], what it computes
/// [`GpuTask::run`], which reaches memory only through the rows its layout
/// declared.
///
/// The task's value is its dimensions. The docs write `layout(dims)`; here
/// `&self` is `dims`, with the [`RowId`](fb_core_arena::RowId)s of the
/// outputs it reads among its fields, handed to it by whoever built the
/// [`Dag`](super::Dag) from the rows an earlier
/// [`Dag::add_gpu`](super::Dag::add_gpu) or
/// [`Dag::add_cpu`](super::Dag::add_cpu) returned. An output's bytes are
/// declared once, by its producer (`docs/memory.md`, "The table").
///
/// `H` is the host kind its pinned rows live in and `S` the stream it
/// enqueues on. The defaults are the production ones, so a real task is
/// `impl GpuTask for Fock`; the device-free tests use the heap kind and
/// [`HostOnly`](fb_core_thread_pool::HostOnly)'s `()` stream.
///
/// A task allocates nothing, since it is handed no allocator; a row it did
/// not declare panics when asked for ([`Bufs`]). It enqueues on its stream
/// and returns, and its pinned rows come as a [`CopyOnly`], which only the
/// copies it enqueues can take: a GPU task does not touch a pinned row from
/// the host (`docs/memory.md`, "The Rust shape").
///
/// ```compile_fail,E0599
/// use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId};
/// use fb_core_error_handling::Status;
/// use fb_core_planner::GpuTask;
/// use fb_core_thread_pool::DeviceStream;
///
/// struct ReadsPinned(RowId<Pinned, f64>);
///
/// impl GpuTask for ReadsPinned {
///     type Rows = ();
///     fn layout(&self, _: &mut Layout<'_, Device>, pinned: &mut Layout<'_, Pinned>) {
///         pinned.read(self.0);
///     }
///     fn run(&self, _: &(), _: &Bufs<'_, Device>, pinned: &CopyOnly<'_, Pinned>, _: &DeviceStream) -> Status {
///         let _ = pinned.get(self.0)[0];
///         Ok(())
///     }
/// }
/// ```
pub trait GpuTask<H: HostKind = Pinned, S = DeviceStream>: Send + Sync {
    /// The rows it uses, as `layout` returned them: a struct of `RowId`s,
    /// usually `#[derive(Clone, Copy)]`. [`Dag::add_gpu`](super::Dag::add_gpu)
    /// hands a copy back to the caller, and passes one to every `run`.
    type Rows: Clone + Send + Sync;

    /// Declare every row: its scratch and outputs, on the device and
    /// pinned, and each earlier output it reads. Called once, at planning.
    fn layout(&self, device: &mut Layout<'_, Device>, pinned: &mut Layout<'_, H>) -> Self::Rows;

    /// Its dimensions by name, such as `[("n_ao", 24)]`: what its entry in
    /// the timing record carries beside its name, and the nightly keys its
    /// times by ([`TaskTime::dims`](fb_core_thread_pool::TaskTime::dims)).
    /// Called once, at planning; none by default.
    fn dims(&self) -> Vec<(&'static str, usize)> {
        Vec::new()
    }

    /// Enqueue its work on `stream`, through the rows `layout` returned,
    /// and return. Called once per run.
    ///
    /// # Errors
    /// Whatever the enqueue fails with; the orchestrator adds
    /// `name@worker` and skips every descendant.
    fn run(
        &self,
        rows: &Self::Rows,
        device: &Bufs<'_, Device>,
        pinned: &CopyOnly<'_, H>,
        stream: &S,
    ) -> Status;
}

/// A CPU task: a unit of work run on the CPU pool, over host rows alone
/// (`docs/programming_model.md`, "A task declares"). Its rows are pinned
/// rows of the GPU worker it is added on, whose copies they stage; the
/// executor starts it only after the events of the stream nodes it runs
/// after, so it reads and writes them as slices.
///
/// As for a [`GpuTask`], `&self` is its dimensions; `P` is the host kind,
/// pinned in production and the heap kind in the device-free tests.
///
/// A CPU task holds no device row: its layout is handed no device table,
/// so one that would declare or read a device row does not build.
///
/// ```compile_fail,E0053
/// use fb_core_arena::{Bufs, Device, Layout, Pinned};
/// use fb_core_error_handling::Status;
/// use fb_core_planner::CpuTask;
///
/// struct HoldsADeviceRow;
///
/// impl CpuTask for HoldsADeviceRow {
///     type Rows = ();
///     fn layout(&self, _: &mut Layout<'_, Device>) {}
///     fn run(&self, _: &(), _: &Bufs<'_, Pinned>) -> Status {
///         Ok(())
///     }
/// }
/// ```
pub trait CpuTask<P: HostKind = Pinned>: Send + Sync {
    /// The rows it uses, as `layout` returned them;
    /// [`Dag::add_cpu`](super::Dag::add_cpu) hands a copy back to the
    /// caller, and passes one to every `run`.
    type Rows: Clone + Send + Sync;

    /// Declare every host row: its scratch and outputs, and each earlier
    /// output it reads. Called once, at planning.
    fn layout(&self, pinned: &mut Layout<'_, P>) -> Self::Rows;

    /// Its dimensions by name, as [`GpuTask::dims`]; none by default.
    fn dims(&self) -> Vec<(&'static str, usize)> {
        Vec::new()
    }

    /// Compute, through the rows `layout` returned. Called once per run.
    ///
    /// # Errors
    /// Whatever the work fails with; the orchestrator adds `name@cpu` and
    /// skips every descendant.
    fn run(&self, rows: &Self::Rows, pinned: &Bufs<'_, P>) -> Status;
}

/// A loop's decision (`docs/programming_model.md`, "Plan once, run
/// many"): the CPU node [`Dag::add_loop`](super::Dag::add_loop) adds after
/// the loop's body, run once per iteration when the whole body has
/// finished and its GPU work has completed, whose answer replays the body
/// or ends the loop. Like a [`CpuTask`], it declares its rows, here host rows
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
