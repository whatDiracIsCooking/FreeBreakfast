//! A task: the two functions a node of a [`Dag`](super::Dag) is.

use fb_core_arena::{Bufs, Device, HostKind, Layout, Pinned};
use fb_core_error_handling::Status;
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
