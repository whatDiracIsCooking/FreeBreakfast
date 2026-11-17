//! The McMurchie–Davidson primitive-pair records, as tasks of the model
//! (`docs/programming_model.md`, "A task declares"): a [`BasisSet`] turned
//! into the per-class arrays the generated one-electron kernels read, with
//! each record's Schwarz bounds, undressed and dressed. The C++
//! `src/integrals/mmd` is the reference design, and its README the
//! description of every field; the order is the one thing this fork
//! changes (below).
//!
//! This crate forms the sweep on the host; the device path is
//! `fb-integrals-sys`'s. [`record_tables`] forms the shell table (each
//! shell's primitives, centre, first Cartesian AO and rank) and every
//! class's shell pairs with their record prefixes, a
//! [`RecordTables`](fb_integrals_sys::RecordTables), and [`add_records`]
//! hands it to
//! [`fb_integrals_sys::add_record_tasks`], which adds to a [`Dag`]:
//!
//! - **`mmd tables`**, on the CPU: the tables written into pinned rows.
//!   A basis of N primitives has O(N²) records but only O(shells²) shell
//!   pairs, so that is all the host forms.
//! - **`upload <table>`**, one per table, on the worker's lane 0: the
//!   tables copied into device rows ([`Dag::upload`]).
//! - **`mmd records (la,lb)`**, one per class `(la, lb)`, `la <= lb`, in
//!   ascending order, the `c`-th on the worker's lane `c`: its records
//!   (`fb_mmd_records`, in enumeration order), its bound (`fb_mmd_bound`)
//!   and its two dressed bounds (`fb_ordered_pair_bound`, as stored and
//!   swapped). The record fields
//!   are its outputs, the class's [`ClassRows`], a type only those tasks
//!   make, so the class launchers can trust the AO offsets it holds on the
//!   device. A reader on another lane, as every reader on lane 0 is but
//!   the first class's, waits for the class by an event on the device, no
//!   host wait.
//!
//! Every size is fixed at planning from the basis: a class's `n_rec` is
//! the sum of its shell pairs' primitive products, nothing is screened, and
//! a class's task holds no scratch.
//!
//! **The records are not sorted.** The C++ sorts each class's records into
//! descending ζ (`mmd_order`: a key, a merge sort and its scratch) for J's
//! screened bra loop, whose load it balances, and to spread a shell pair's
//! records across warps for J's and K's atomics. No reader here needs it:
//! the one-electron kernels and the bounds are one thread per record and
//! independent of the order, and the three-center integrals regrouped the
//! sorted records by shell pair on the host, undoing it. So the records
//! stay in enumeration order (shell pair, then bra primitive, then ket
//! primitive), each shell pair's records one contiguous run of its class,
//! its range the pair's record prefix in the [`record_tables`], which the
//! three-center integrals take at planning instead of regrouping. A J or K
//! built here sorts, or bins, its own bras.
//!
//! | C++ (`src/integrals/mmd`) | Here |
//! |---|---|
//! | `primitive_pair_classes(res, basis)` | [`add_records`]`(dag, worker, basis)`: the tasks above, run whenever the plan runs |
//! | `PrimitivePairClass`, a device buffer per field on its own stream | [`ClassRows`]: a device output row per field, written on the class's own lane |
//! | `basis_sweep`, `shell_table`, `pair_table`, `shell_ranks` | the host tables, formed here and checked by [`RecordTables::new`](fb_integrals_sys::RecordTables::new) |
//! | `upload`'s carve of `res`'s pinned workspace, and the device buffers for the call | the tables' pinned rows (outputs of `mmd tables`) and device rows (outputs of the uploads, read by every class) |
//! | `mmd_order`, each class sorted into descending ζ, and `form_classes`' carve of `res`'s device workspace for it, two keys and two permutations | nothing: the records stay in enumeration order, each shell pair's one run (above), and a class's task holds no scratch |
//! | each class forked from the one before, so the sorts share the carve | each class on its own lane, with no scratch to share, so nothing chains them |
//! | `three_center` copying the sorted records' offsets back and regrouping them by shell pair | [`record_tables`], formed again at planning by a reader: each pair's record range is its prefix, its offsets its shells' |
//! | `primitive_pair_classes_footprint` | the plan's tables ([`Plan`](fb_core_planner::Plan)): the footprint is the layout |
//! | `mixed_primitive_pair_classes(res, rows, cols)`, `mixed_sweep` | [`add_mixed_records`]`(dag, worker, rows, cols)`: the same tasks over both bases' table, returning [`MixedClassRows`], which no one-basis launcher takes; [`mixed_record_tables`], the tables, for a reader at planning |
//! | `mixed_primitive_pair_classes_footprint` | the plan's tables, as for one basis |
//!
//! [`BasisSet`]: fb_basis::BasisSet

#![forbid(unsafe_code)]

mod tables;
#[cfg(test)]
mod tests;

use fb_core_error_handling::{Error, Result};
use fb_core_planner::Dag;
use fb_integrals_sys::{ClassRows, MixedClassRows, add_mixed_record_tasks, add_record_tasks};

/// Add the records of `basis` to `dag` on GPU worker `worker`: the sweep
/// formed on the host and handed to [`add_record_tasks`], a CPU task,
/// `mmd tables`, an upload of each table on the worker's lane 0, and a
/// stream task per class, `mmd records (la,lb)`, the `c`-th on the
/// worker's lane `c`. Returns each class's rows, ascending in `(la, lb)`;
/// a basis with no shells adds nothing and returns none.
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
    Ok(add_record_tasks(dag, worker, record_tables(basis)))
}

/// The tables [`add_records`] forms the records of `basis` from: every
/// shell pair `i <= j`, swapped when `i` has the higher momentum, in its
/// class `(la, lb)`, ascending, with its first record, and the shell
/// table. A host function, which a reader of the records calls at planning
/// for what their order puts where: the records of the `c`-th class's
/// `p`-th pair are `[rec_begin[p], rec_begin[p + 1])` of that class's rows,
/// and their AO offsets the pair's shells'. Aborts past `i32::MAX`
/// primitives, or records in one class, as [`add_records`] does.
///
/// # Panics
/// As [`RecordTables::new`](fb_integrals_sys::RecordTables::new), which a
/// basis's own sweep never trips.
#[track_caller]
pub fn record_tables(basis: &fb_basis::BasisSet) -> fb_integrals_sys::RecordTables {
    tables::tables(basis)
}

/// The tables [`add_mixed_records`] forms the records between `rows` and
/// `cols` from: every ordered shell pair, `i` of `rows` outer and `j` of
/// `cols` inner, in its class `(l_i, l_j)`, ascending, with its first
/// record, and both bases' shell table, `rows`' shells first. A host
/// function, as [`record_tables`] is: what a reader of the records calls
/// at planning for what their order puts where.
///
/// # Panics
/// As [`RecordTables::mixed`](fb_integrals_sys::RecordTables::mixed),
/// which two bases' own sweep never trips.
#[track_caller]
pub fn mixed_record_tables(
    rows: &fb_basis::BasisSet,
    cols: &fb_basis::BasisSet,
) -> fb_integrals_sys::RecordTables {
    tables::mixed_tables(rows, cols)
}

/// Add the records between two bases, `rows` and `cols`, to `dag` on GPU
/// worker `worker`, as [`add_records`] adds one basis's: the sweep formed
/// on the host, then the same tasks on the same lanes over both bases'
/// shells. Returns each class's rows, ascending in `(la, lb)`; either
/// basis with no shells adds nothing and returns none.
///
/// Every **ordered** shell pair (`i` of `rows`, `j` of `cols`) gives one
/// record per primitive pair, `i` always the bra, in class `(l_i, l_j)`:
/// nothing is swapped, so `la > lb` classes appear. `off_a` is a Cartesian
/// AO of `rows` and `off_b` one of `cols`, two index sets, so equal
/// offsets mean nothing; each rank is among its own basis's shells. The
/// fields, their order (enumeration order: `rows`' shell, then `cols`',
/// then the two primitives) and the bounds are
/// [`add_records`]'. The bases may differ in set, in geometry, in both or
/// in neither, and may be the same.
///
/// First the kernel library's load check, as [`add_records`] makes it.
///
/// # Errors
/// As [`add_records`].
///
/// # Panics
/// As [`RecordTables::mixed`](fb_integrals_sys::RecordTables::mixed),
/// which two bases' own sweep never trips.
#[track_caller]
pub fn add_mixed_records(
    dag: &mut Dag<'_>,
    worker: usize,
    rows: &fb_basis::BasisSet,
    cols: &fb_basis::BasisSet,
) -> Result<Vec<MixedClassRows>> {
    // Not `map_err(Error::gpu)`, as in add_records.
    if let Err(e) = fb_integrals_sys::check_library() {
        return Err(Error::gpu(e));
    }
    if rows.nshells() == 0 || cols.nshells() == 0 {
        return Ok(Vec::new());
    }
    Ok(add_mixed_record_tasks(
        dag,
        worker,
        tables::mixed_tables(rows, cols),
    ))
}
