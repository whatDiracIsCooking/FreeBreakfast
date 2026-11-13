//! The McMurchie–Davidson primitive-pair records, as tasks of the model
//! (`docs/programming_model.md`, "A task declares"): a [`BasisSet`] turned
//! into the per-class arrays the generated one-electron kernels read, with
//! each record's Schwarz bounds, undressed and dressed. The C++
//! `src/integrals/mmd` is the reference design, and its README the
//! description of every field and of the order.
//!
//! [`add_records`] adds two tasks to a [`Dag`](fb_core_planner::Dag):
//!
//! - **`mmd tables`**, on the CPU: the shell table (each shell's
//!   primitives, centre, first Cartesian AO and rank) and every class's
//!   shell pairs with their record prefixes, formed on the host and written
//!   into pinned rows, its outputs. A basis of N primitives has O(N²)
//!   records but only O(shells²) shell pairs, so that is all the host forms.
//! - **`mmd records`**, on the worker's stream: the tables copied into
//!   device rows, its scratch, then each class `(la, lb)`, `la <= lb`, in
//!   ascending order: its order (`fb_mmd_order`: the key `-ζ` and a stable
//!   sort, in scratch sized to the largest class), its records
//!   (`fb_mmd_records`, every field written once, already in descending ζ),
//!   its bound (`fb_mmd_bound`) and its two dressed bounds
//!   (`fb_ordered_pair_bound`, as stored and swapped). The record fields
//!   are its outputs, [`ClassRows`], one per class.
//!
//! Every size is fixed at planning from the basis: a class's `n_rec` is
//! the sum of its shell pairs' primitive products, nothing is screened, and
//! the sort's workspace is `fb_mmd_order_workspace_bytes` of the largest
//! class. So the layout calls into the kernel library at planning (a pure
//! host function), and this crate always links it.
//!
//! | C++ (`src/integrals/mmd`) | Here |
//! |---|---|
//! | `primitive_pair_classes(res, basis)` | [`add_records`]`(dag, worker, basis)`: two tasks, run whenever the plan runs |
//! | `PrimitivePairClass`, a device buffer per field on its own stream | [`ClassRows`]: a device output row per field, on the worker's one stream |
//! | `basis_sweep`, `shell_table`, `pair_table`, `shell_ranks` | the `mmd tables` task's host tables |
//! | `upload`'s carve of `res`'s pinned workspace, and the device buffers for the call | the tables' pinned rows (outputs of `mmd tables`) and device rows (scratch of `mmd records`) |
//! | `form_classes`' carve of `res`'s device workspace, two keys and two permutations | one key, one permutation and `fb_mmd_order`'s workspace, scratch of `mmd records`, sized to the largest class |
//! | each class forked from the one before, so the sorts share the carve | every class on the worker's one stream |
//! | `primitive_pair_classes_footprint` | the plan's tables ([`Planned`](fb_core_planner::Planned)): the footprint is the layout |
//! | `mixed_primitive_pair_classes` | not ported yet |
//!
//! [`BasisSet`]: fb_basis::BasisSet

mod tables;
mod tasks;
#[cfg(test)]
mod tests;

pub use tasks::{ClassRows, add_records};
