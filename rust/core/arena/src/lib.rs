//! The arena: every buffer a plan holds, laid out once at planning
//! (`docs/memory.md`, `docs/programming_model.md`, "Memory: layouts and
//! arenas").
//!
//! - [`TableBuilder`] declares the rows node by node: a node's scratch, its
//!   outputs and who reads them, and the worker's own rows. Lifetimes are
//!   derived from the users and never written. [`TableBuilder::node`] is
//!   one node's [`Layout`], what a task declares through.
//!   [`TableBuilder::add_loop`] marks a loop's positions, whose body's
//!   rows are a table of their own laid out in a region the rest of the
//!   table holds for the loop; [`TableBuilder::add_counted`] a counted
//!   loop's, whose region holds that table once per iteration in flight,
//!   and which may lie in a decision loop's body, its region a row of
//!   that body's table;
//!   [`TableBuilder::write`] is a row the loop carries from one iteration
//!   to the next (`docs/memory.md`, "Nested plans").
//! - [`TableBuilder::build`] lays them out with [`assign`], the pure offset
//!   assigner, into a [`Table`]: an offset per row and the block's total,
//!   which is the plan's peak for that worker and kind. Its text is the
//!   committed memory golden.
//! - [`Table::handoffs`] is what reusing bytes costs the schedule: the
//!   orderings the plan must have so that a row's previous owner is done
//!   before the next one starts. [`Table::reads`] is what the data costs
//!   it: each reader after its producer.
//! - [`admit`] checks a table against its worker's cap, and refuses naming
//!   every row larger than the cap on its own and the rows live at the
//!   peak; [`admit_host`] a node's pinned blocks and the plan's heap block
//!   together against [`host_available`], the node's available memory
//!   within its cgroups' limits. With `device`, `device_memory` reads a card's free and total
//!   memory at planning: the card bounds a device cap, and the difference
//!   is the device's measured reserve.
//! - [`Block::new`] makes the table's one allocation, on the device,
//!   pinned or on the heap, filled with `0xFF`; it is freed when the block
//!   drops.
//!   [`allocations`] counts them, for a test that a run allocates nothing.
//! - [`Block::partition`] is what a node is handed when it runs, in one
//!   iteration: a [`Bufs`] that gives it its own rows and no others, a
//!   counted loop's body rows in its iteration's copy, as `&[T]` /
//!   `&mut [T]` on the host and [`DeviceRef`] / [`DeviceMut`] pointers on
//!   the device. A stream node is handed its host rows as a [`CopyOnly`],
//!   which only its copies can take. With `device`, `Block::hold` holds a worker row outside
//!   any view, as a `HeldRow` that keeps the block alive: memory a library
//!   keeps beyond a node, a BLAS handle's workspace.
//! - [`memcpy_h2p`], [`memcpy_p2h`] and, with `device`, `memcpy_p2d`,
//!   `memcpy_d2p` and `memcpy_d2d` move a node's data between the host and
//!   its rows, with no `unsafe` at the call (the `memcpy` module has why
//!   each is sound).
//!
//! Everything but the block is pure and needs no device. The kind of
//! memory, [`Device`], [`Pinned`] or [`Heap`], is in every type, so a row
//! of one kind cannot reach a table or block of another, and a row's element type is
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
//! | `check_plan`'s refusal naming the step at the peak | [`admit`]'s, naming the node and its live rows, and every row over the cap alone |
//! | `kWorkspaceAlignment` | [`DEVICE_ALIGNMENT`], honoured by every row |
//!
//! # Not yet
//!
//! Transport rows, between two workers' tables, wait for placement. A table
//! is one GPU worker's, shared by its lanes: lifetimes are positions in the
//! plan's rank order whatever lane a node is on, and the assigner packs to
//! the fewest bytes, so a reuse between two lanes is a handoff the planner
//! adds as an edge, an event on the device that serializes the two
//! (`docs/memory.md`, "Lifetimes"). A loop's region holds one body, a
//! counted loop's one per copy, and a counted loop's region may be a row
//! of a decision loop's body; the max over several inner shapes, the
//! semi-numerical Hessian's, a decision loop in a decision loop's body
//! and a loop in a counted loop's are not built. Declaring rows from tasks, and turning the table's
//! [`reads`](Table::reads) and [`handoffs`](Table::handoffs) into a plan's
//! dependencies, is `fb_core_planner`'s. A block is
//! allocated on the thread that makes it, not its worker's: a block lives
//! for the plan, and the run's threads only for one run. Adaptive
//! widths are not built; the reserve, the default caps and which tables go
//! into the node's host sum are admission's caller's
//! (`fb_pipeline_config::admit_plan`).

#![deny(
    clippy::undocumented_unsafe_blocks,
    clippy::multiple_unsafe_ops_per_block
)]

mod admit;
mod assign;
mod block;
mod bufs;
mod kind;
mod memcpy;
mod table;
#[cfg(test)]
mod tests;

#[cfg(feature = "device")]
pub use admit::device_memory;
pub use admit::{admit, admit_host, host_available};
pub use assign::{Assignment, Life, Row, assign};
pub use block::{Block, allocations};
#[cfg(feature = "device")]
pub use bufs::HeldRow;
pub use bufs::{Bufs, CopyOnly, DeviceMut, DeviceRef};
#[cfg(any(test, feature = "test-support"))]
pub use kind::PinnedStandIn;
pub use kind::{Device, Element, Heap, HostKind, Kind, Pinned, StagingKind};
#[cfg(feature = "device")]
pub use memcpy::{memcpy_d2d, memcpy_d2p, memcpy_p2d};
pub use memcpy::{memcpy_h2p, memcpy_p2h};
pub use table::{Access, Layout, Peak, RowId, Table, TableBuilder};

/// The alignment of every row's offset, in bytes, and the least the block
/// itself has: the C++ `kWorkspaceAlignment`, and the pool's allocation
/// granularity every launcher has assumed so far
/// (`docs/gpu_layer.md`, "Scratch: the workspace query").
pub const DEVICE_ALIGNMENT: usize = 256;
