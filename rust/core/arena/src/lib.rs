//! The arena: every buffer a plan holds, laid out once at planning
//! (`docs/memory.md`, `docs/programming_model.md`, "Memory: layouts and
//! arenas").
//!
//! - [`TableBuilder`] declares the rows node by node: a node's scratch, its
//!   outputs and who reads them, and the worker's own rows. Lifetimes are
//!   derived from the users and never written. [`TableBuilder::node`] is
//!   one node's [`Layout`], what a task declares through.
//! - [`TableBuilder::build`] lays them out with [`assign`], the pure offset
//!   assigner, into a [`Table`]: an offset per row and the block's total,
//!   which is the plan's peak for that worker and kind. Its text is the
//!   committed memory golden.
//! - [`Table::handoffs`] is what reusing bytes costs the schedule: the
//!   orderings the plan must have so that a row's previous owner is done
//!   before the next one starts. [`Table::reads`] is what the data costs
//!   it: each reader after its producer.
//! - [`admit`] checks a table against its worker's cap, and refuses naming
//!   the rows live at the peak.
//! - [`Block::new`] makes the table's one allocation, on the device or
//!   pinned, filled with `0xFF`; it is freed when the block drops.
//! - [`Block::partition`] is what a node is handed when it runs: a
//!   [`Bufs`] that gives it its own rows and no others, as `&[T]` /
//!   `&mut [T]` on the host and [`DeviceRef`] / [`DeviceMut`] pointers on
//!   the device.
//!
//! Everything but the block is pure and needs no device. The kind of
//! memory, [`Device`] or [`Pinned`], is in every type, so a row of one kind
//! cannot reach a table or block of the other, and a row's element type is
//! an [`Element`]. This module is the only one that allocates: the raw calls
//! are on a trait nothing outside it can name.
//!
//! # Why a view's checks are always on
//!
//! Two nodes running at once both need rows of one block, so
//! [`Block::partition`] takes `&self`, and the orchestrator that calls it
//! has no `unsafe`. A safe `&self` API that hands out `&mut [T]` cannot rest
//! on a debug-only contract, so each row has an atomic borrow flag: a
//! writer is alone, readers share, and a conflict panics naming the row.
//! In a plan that honours [`Table::handoffs`] none ever fires, and each
//! costs one uncontended atomic per row a node asks for. For the same
//! reason a pinned block is filled in release builds too: a slice must not
//! be uninitialised memory.
//!
//! # What it replaces
//!
//! | C++ | Here |
//! |---|---|
//! | `Footprint`'s `then` / `drop` and the plan fold's peak (`src/core/footprint`, `src/pipeline/footprint`) | [`Table::total`]: the peak is not estimated, it is the layout |
//! | The footprint tests against the measured pool peak | Nothing to measure: the table is what the block will hold |
//! | `kOwnPoolBytes`, subtracted from the room by hand | A worker row ([`TableBuilder::worker`]), laid out like any other |
//! | `DeviceResources`' workspace, carved by one routine at a time | Each node's scratch is a row of its own, aliased with every other node's dead scratch |
//! | The 64 MiB fragmentation headroom | None: one allocation has nothing to fragment |
//! | `check_plan`'s refusal naming the step at the peak | [`admit`]'s, naming the node and its live rows |
//! | `kWorkspaceAlignment` | [`DEVICE_ALIGNMENT`], honoured by every row |
//!
//! # Not yet
//!
//! One stream worker per table: lifetimes are positions in the plan's rank
//! order, and reuse across streams, transport rows and nested regions wait
//! for placement. Declaring rows from tasks, and turning the table's
//! [`reads`](Table::reads) and [`handoffs`](Table::handoffs) into a plan's
//! dependencies, is `fb_core_planner`'s. A block is
//! allocated on the thread that makes it, not its worker's: a block lives
//! for the plan, and the run's threads only for one run. Adaptive
//! widths, the measured reserve and the node-wide pinned sum are
//! admission's and are not here. Library handles do not take their
//! workspace from a worker row yet: WarpWraps' C ABI has no
//! `wwr_blas_set_workspace`.

#![deny(
    clippy::undocumented_unsafe_blocks,
    clippy::multiple_unsafe_ops_per_block
)]

mod admit;
mod assign;
mod block;
mod bufs;
mod kind;
mod table;
#[cfg(test)]
mod tests;

pub use admit::admit;
pub use assign::{Assignment, Life, Row, assign};
pub use block::Block;
pub use bufs::{Bufs, DeviceMut, DeviceRef};
#[cfg(any(test, feature = "test-support"))]
pub use kind::Heap;
pub use kind::{Device, Element, HostKind, Kind, Pinned};
pub use table::{Access, Layout, Peak, RowId, Table, TableBuilder};

/// The alignment of every row's offset, in bytes, and the least the block
/// itself has: the C++ `kWorkspaceAlignment`, and the pool's allocation
/// granularity every launcher has assumed so far
/// (`docs/gpu_layer.md`, "Scratch: the workspace query").
pub const DEVICE_ALIGNMENT: usize = 256;
