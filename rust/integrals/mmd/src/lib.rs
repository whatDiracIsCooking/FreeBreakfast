//! The McMurchie–Davidson primitive-pair records, as tasks of the model
//! (`docs/programming_model.md`, "A task declares"): a [`BasisSet`] turned
//! into the per-class arrays the generated one-electron kernels read, with
//! each record's Schwarz bounds, undressed and dressed. The C++
//! `src/integrals/mmd` is the reference design, and its README the
//! description of every field and of the order.
//!
//! This crate forms the sweep on the host; the device path is
//! `fb-integrals-sys`'s. [`add_records`] forms the shell table (each
//! shell's primitives, centre, first Cartesian AO and rank) and every
//! class's shell pairs with their record prefixes, a
//! [`RecordTables`](fb_integrals_sys::RecordTables), and hands it to
//! [`fb_integrals_sys::add_record_tasks`], which adds two tasks to a
//! [`Dag`](fb_core_planner::Dag):
//!
//! - **`mmd tables`**, on the CPU: the tables written into pinned rows.
//!   A basis of N primitives has O(N²) records but only O(shells²) shell
//!   pairs, so that is all the host forms.
//! - **`mmd records`**, on the worker's stream: the tables copied into
//!   device rows, then each class `(la, lb)`, `la <= lb`, in ascending
//!   order: its order (`fb_mmd_order`), its records (`fb_mmd_records`,
//!   already in descending ζ), its bound (`fb_mmd_bound`) and its two
//!   dressed bounds (`fb_ordered_pair_bound`, as stored and swapped). The
//!   record fields are its outputs,
//!   [`ClassRows`](fb_integrals_sys::ClassRows), one per class, a type only
//!   those tasks make, so the class launchers can trust the AO offsets it
//!   holds on the device.
//!
//! Every size is fixed at planning from the basis: a class's `n_rec` is
//! the sum of its shell pairs' primitive products, nothing is screened, and
//! the sort's workspace is `fb_mmd_order_workspace_bytes` of the largest
//! class.
//!
//! | C++ (`src/integrals/mmd`) | Here |
//! |---|---|
//! | `primitive_pair_classes(res, basis)` | [`add_records`]`(dag, worker, basis)`: two tasks, run whenever the plan runs |
//! | `PrimitivePairClass`, a device buffer per field on its own stream | [`ClassRows`](fb_integrals_sys::ClassRows): a device output row per field, on the worker's one stream |
//! | `basis_sweep`, `shell_table`, `pair_table`, `shell_ranks` | the host tables, formed here and checked by [`RecordTables::new`](fb_integrals_sys::RecordTables::new) |
//! | `upload`'s carve of `res`'s pinned workspace, and the device buffers for the call | the tables' pinned rows (outputs of `mmd tables`) and device rows (scratch of `mmd records`) |
//! | `form_classes`' carve of `res`'s device workspace, two keys and two permutations | one key, one permutation and `fb_mmd_order`'s workspace, scratch of `mmd records`, sized to the largest class |
//! | each class forked from the one before, so the sorts share the carve | every class on the worker's one stream |
//! | `primitive_pair_classes_footprint` | the plan's tables ([`Planned`](fb_core_planner::Planned)): the footprint is the layout |
//! | `mixed_primitive_pair_classes` | not ported yet |
//!
//! [`BasisSet`]: fb_basis::BasisSet

#![forbid(unsafe_code)]

mod tables;
#[cfg(test)]
mod tests;

use fb_core_error_handling::{Error, Result};
use fb_core_planner::Dag;
use fb_integrals_sys::{ClassRows, add_record_tasks};

/// Add the records of `basis` to `dag` on GPU worker `worker`: the sweep
/// formed on the host and handed to [`add_record_tasks`], a CPU task,
/// `mmd tables`, and a stream task, `mmd records`. Returns each class's
/// rows, ascending in `(la, lb)`; a basis with no shells adds nothing and
/// returns none.
///
/// Every size is the basis's, fixed here at planning: a class's `n_rec` is
/// the sum of its shell pairs' primitive products. Aborts past `i32::MAX`
/// primitives, or records in one class, as the C++ does.
///
/// First, before anything is added, the kernel library's load check
/// ([`fb_integrals_sys::check_library`]): planning is a run's first call into
/// the library, and every launch of the records, or of a task reading them,
/// comes after it.
///
/// # Errors
/// A `GpuRuntime` error with the check's message, if the loaded
/// `libfb_integrals_kernels.so` is not the one the bindings describe.
///
/// # Panics
/// As [`RecordTables::new`](fb_integrals_sys::RecordTables::new), which a
/// basis's own sweep never trips.
#[track_caller]
pub fn add_records(
    dag: &mut Dag<'_>,
    worker: usize,
    basis: &fb_basis::BasisSet,
) -> Result<Vec<ClassRows>> {
    // Not `map_err(Error::gpu)`: through a function pointer the error's
    // origin would be core's, not this call's caller.
    if let Err(e) = fb_integrals_sys::check_library() {
        return Err(Error::gpu(e));
    }
    if basis.nshells() == 0 {
        return Ok(Vec::new());
    }
    Ok(add_record_tasks(dag, worker, tables::tables(basis)))
}
