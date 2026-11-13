//! The arena: every buffer a plan holds, laid out once at planning
//! (`docs/memory.md`, `docs/programming_model.md`, "Memory: layouts and
//! arenas").
//!
//! - [`TableBuilder`] declares the rows node by node: a node's scratch, its
//!   outputs and who reads them, and the worker's own rows. Lifetimes are
//!   derived from the users and never written.
//! - [`TableBuilder::build`] lays them out with [`assign`], the pure offset
//!   assigner, into a [`Table`]: an offset per row and the block's total,
//!   which is the plan's peak for that worker and kind. Its text is the
//!   committed memory golden.
//! - [`Table::handoffs`] is what reusing bytes costs the schedule: the
//!   orderings the plan must have so that a row's previous owner is done
//!   before the next one starts.
//! - [`admit`] checks a table against its worker's cap, and refuses naming
//!   the rows live at the peak.
//!
//! Everything here is pure and needs no device. The kind of memory,
//! [`Device`] or [`Pinned`], is in every type, so a row of one kind cannot
//! reach a table of the other, and a row's element type is an [`Element`].
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
//! for placement. The blocks the tables are allocated as, and the views a
//! node is handed, come next. Adaptive widths, the measured reserve and the
//! node-wide pinned sum are admission's and are not here.

#![deny(
    clippy::undocumented_unsafe_blocks,
    clippy::multiple_unsafe_ops_per_block
)]

mod admit;
mod assign;
mod kind;
mod table;
#[cfg(test)]
mod tests;

pub use admit::admit;
pub use assign::{Assignment, Life, Row, assign};
pub use kind::{Device, Element, Kind, Pinned};
pub use table::{Access, Peak, RowId, Table, TableBuilder};

/// The alignment of every row's offset, in bytes, and the least the block
/// itself has: the C++ `kWorkspaceAlignment`, and the pool's allocation
/// granularity every launcher has assumed so far
/// (`docs/gpu_layer.md`, "Scratch: the workspace query").
pub const DEVICE_ALIGNMENT: usize = 256;
