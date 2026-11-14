//! The primitive-pair records' device path: the tables the kernels read
//! ([`RecordTables`], validated), the two tasks that turn them into every
//! class's records ([`add_record_tasks`]), and the records themselves
//! ([`ClassRows`]), which only those tasks produce.
//!
//! The tasks, added in this order:
//!
//! - **`mmd tables`**, on the CPU: the shell table and every class's shell
//!   pairs with their record prefixes, written into pinned rows, its
//!   outputs.
//! - **`mmd records`**, on the worker's stream: the tables copied into
//!   device rows, its scratch, then each class `(la, lb)` in the tables'
//!   order: its order ([`fb_mmd_order`]: the key `-ζ` and a stable sort, in
//!   scratch sized to the largest class), its records ([`fb_mmd_records`],
//!   every field written once, already in descending ζ), its bound
//!   ([`fb_mmd_bound`]) and its two dressed bounds
//!   ([`fb_ordered_pair_bound`], as stored and swapped). The record fields
//!   are its outputs, one [`ClassRows`] per class.
//!
//! Forming the tables from a basis, the sweep, is `fb-integrals-mmd`'s;
//! this module trusts none of it, and checks every index the kernels
//! follow when the tables are made ([`RecordTables::new`]).

use std::sync::Arc;

use fb_core_arena::{Bufs, Device, Element, Kind, Layout, Pinned, RowId, memcpy_h2p, memcpy_p2d};
use fb_core_error_handling::{Result, Status};
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::handle::DeviceHandleStream;
use fb_core_wwr::memset;

use crate::{
    FB_PAIR_BOUND_MAX_MOMENTUM, FB_RECORD_BOUND_MAX_MOMENTUM, fb_mmd_bound, fb_mmd_order,
    fb_mmd_order_workspace_bytes, fb_mmd_records, fb_ordered_pair_bound,
    fb_primitive_pair_class_view, fb_primitive_pair_records, fb_primitive_pair_source, launched,
};

/// A basis's shells and primitives, a field per array, as the device reads
/// them ([`fb_primitive_pair_source`]). Plain data: [`RecordTables::new`]
/// checks it.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ShellTable {
    /// Each shell's first primitive, `n_shells + 1` long.
    pub prim_begin: Vec<i32>,
    /// Every primitive's exponent, shell by shell.
    pub exponent: Vec<f64>,
    /// Every primitive's coefficient, as the basis gives it.
    pub coefficient: Vec<f64>,
    /// Each shell's centre: x.
    pub c_x: Vec<f64>,
    /// y.
    pub c_y: Vec<f64>,
    /// z.
    pub c_z: Vec<f64>,
    /// Each shell's first Cartesian AO.
    pub ao_offset: Vec<i32>,
    /// Each shell's index among the basis's shells of its momentum, in
    /// basis order.
    pub rank: Vec<i32>,
}

/// One class's shell pairs, each pair's bra of momentum `la` and ket of
/// momentum `lb`, and each one's first record. Plain data:
/// [`RecordTables::new`] checks it.
#[derive(Clone, Debug, PartialEq)]
pub struct ClassPairs {
    /// The bra shells' momentum.
    pub la: i32,
    /// The ket shells'.
    pub lb: i32,
    /// Each pair's bra shell.
    pub bra: Vec<u32>,
    /// Each pair's ket shell.
    pub ket: Vec<u32>,
    /// The prefix of the pairs' primitive products, `n_pairs + 1` long: its
    /// last entry is the class's record count.
    pub rec_begin: Vec<i32>,
}

impl ClassPairs {
    /// The class's records: the last entry of its prefix.
    ///
    /// # Panics
    /// If the prefix is empty.
    pub fn n_rec(&self) -> i32 {
        *self.rec_begin.last().expect("a prefix starts at 0")
    }
}

/// The shell table and every class of a basis whose `n_cart` Cartesian AOs
/// the records' offsets index, or of two bases, a bra basis's shells and
/// then a ket basis's, each indexing its own: what [`add_record_tasks`]
/// (one basis) or [`add_mixed_record_tasks`] (two) forms the records from.
/// Made only by [`RecordTables::new`] and [`RecordTables::mixed`], which
/// check every index the kernels follow, so the tables are fixed once
/// checked.
#[derive(Debug, PartialEq)]
pub struct RecordTables {
    shells: ShellTable,
    classes: Vec<ClassPairs>,
    /// The Cartesian AOs the bra offsets index, and the ket offsets.
    n_cart: [usize; 2],
    /// For two bases, the bra basis's shells, which come first.
    n_bra_shells: Option<usize>,
}

/// A shell of momentum `l`'s Cartesian functions, `(l + 1)(l + 2) / 2`.
fn n_cart_of(l: i32) -> usize {
    let l = l as usize;
    (l + 1) * (l + 2) / 2
}

impl RecordTables {
    /// The tables of a basis of `n_cart` Cartesian AOs, checked.
    ///
    /// Every check is one a kernel's memory access rests on, so each is
    /// always on: the order and records kernels find a record's pair by
    /// binary search over its class's prefix, then its primitives through
    /// `prim_begin`, and read every shell array at the pair's two shells;
    /// the class kernels write a class's matrix block at the AO offsets the
    /// records copy from `ao_offset`, `n_cart_of(la)` × `n_cart_of(lb)` of
    /// it, into a matrix they index in 32 bits.
    ///
    /// # Panics
    /// If `n_cart²` does not fit an `i32`; a shell array is not one entry
    /// per shell (`prim_begin` one more); `prim_begin` does not start at 0,
    /// decreases, or does not end at the primitive count, which the
    /// exponents and the coefficients do not share; a class has a negative
    /// momentum, bra and ket lists of different lengths, or a prefix that
    /// is not one longer, does not start at 0, or does not step by each
    /// pair's primitive product; a pair names a shell past the table; or a
    /// pair's shell block, its AO offset and the class's momentum, does not
    /// lie inside `[0, n_cart)`.
    #[track_caller]
    pub fn new(shells: ShellTable, classes: Vec<ClassPairs>, n_cart: usize) -> RecordTables {
        RecordTables::checked(shells, classes, [n_cart; 2], None)
    }

    /// The tables of two bases, checked as [`RecordTables::new`] checks one:
    /// the first `n_bra_shells` shells of `shells` are the bra basis's,
    /// whose `n_cart[0]` Cartesian AOs their offsets index, and the rest
    /// the ket basis's, indexing its own `n_cart[1]`. Every pair's bra is a
    /// bra-basis shell and its ket a ket-basis shell, in a class of any
    /// order: what a two-basis build reads into an `n_cart[0] × n_cart[1]`
    /// matrix.
    ///
    /// # Panics
    /// As [`RecordTables::new`], with `n_cart[0] · n_cart[1]` for `n_cart²`
    /// and each shell's block inside its own basis's AOs; and if
    /// `n_bra_shells` is past the table, or a pair's bra is not one of the
    /// first `n_bra_shells` shells or its ket is one of them.
    #[track_caller]
    pub fn mixed(
        shells: ShellTable,
        classes: Vec<ClassPairs>,
        n_bra_shells: usize,
        n_cart: [usize; 2],
    ) -> RecordTables {
        RecordTables::checked(shells, classes, n_cart, Some(n_bra_shells))
    }

    /// [`RecordTables::new`]'s and [`RecordTables::mixed`]'s checks.
    #[track_caller]
    fn checked(
        shells: ShellTable,
        classes: Vec<ClassPairs>,
        n_cart: [usize; 2],
        n_bra_shells: Option<usize>,
    ) -> RecordTables {
        let [n_bra, n_ket] = n_cart;
        assert!(
            n_bra
                .checked_mul(n_ket)
                .is_some_and(|n2| i32::try_from(n2).is_ok()),
            "RecordTables: {} Cartesian AOs do not fit the kernels' 32-bit index",
            if n_bra_shells.is_some() {
                format!("{n_bra} × {n_ket}")
            } else {
                format!("{n_bra}²")
            }
        );
        let s = &shells;
        let n_shells = s.ao_offset.len();
        assert!(
            s.prim_begin.len() == n_shells + 1
                && [s.c_x.len(), s.c_y.len(), s.c_z.len(), s.rank.len()]
                    .iter()
                    .all(|&len| len == n_shells),
            "RecordTables: the shell table's arrays are not one per shell"
        );
        let n_prim = s.exponent.len();
        assert!(
            s.prim_begin[0] == 0
                && s.prim_begin.is_sorted()
                && s.prim_begin[n_shells] as usize == n_prim
                && s.coefficient.len() == n_prim,
            "RecordTables: prim_begin is not a prefix of the {n_prim} primitives"
        );
        if let Some(n) = n_bra_shells {
            assert!(
                n <= n_shells,
                "RecordTables: {n} bra shells, but the table has {n_shells}"
            );
        }
        // Non-negative and non-decreasing, checked above.
        let n_prims_of = |shell: usize| i64::from(s.prim_begin[shell + 1] - s.prim_begin[shell]);
        for c in &classes {
            let (la, lb) = (c.la, c.lb);
            let n_pairs = c.bra.len();
            assert!(
                la >= 0
                    && lb >= 0
                    && c.ket.len() == n_pairs
                    && c.rec_begin.len() == n_pairs + 1
                    && c.rec_begin[0] == 0,
                "RecordTables: class ({la}, {lb}) is not {n_pairs} pairs with a prefix from 0"
            );
            for p in 0..n_pairs {
                let (a, b) = (c.bra[p] as usize, c.ket[p] as usize);
                assert!(
                    a < n_shells && b < n_shells,
                    "RecordTables: class ({la}, {lb}) pair {p} names shells {a} and {b} of {n_shells}"
                );
                assert!(
                    i64::from(c.rec_begin[p + 1]) - i64::from(c.rec_begin[p])
                        == n_prims_of(a) * n_prims_of(b),
                    "RecordTables: class ({la}, {lb}) pair {p}'s records are not its primitive product"
                );
                if let Some(n) = n_bra_shells {
                    assert!(
                        a < n && b >= n,
                        "RecordTables: class ({la}, {lb}) pair {p} pairs shells {a} and {b}, \
                         not a bra shell below {n} with a ket shell from it"
                    );
                }
                for (shell, l, n_cart) in [(a, la, n_bra), (b, lb, n_ket)] {
                    let offset = s.ao_offset[shell];
                    assert!(
                        offset >= 0 && offset as usize + n_cart_of(l) <= n_cart,
                        "RecordTables: shell {shell} at AO {offset} with momentum {l} does not \
                         fit {n_cart} Cartesian AOs"
                    );
                }
            }
        }
        RecordTables {
            shells,
            classes,
            n_cart,
            n_bra_shells,
        }
    }

    /// The shell table.
    pub fn shells(&self) -> &ShellTable {
        &self.shells
    }

    /// Every class, in the order the records are formed in.
    pub fn classes(&self) -> &[ClassPairs] {
        &self.classes
    }

    /// The Cartesian AOs the records' bra offsets index: the basis's, or
    /// the bra basis's of two.
    pub fn n_cart(&self) -> usize {
        self.n_cart[0]
    }

    /// Every class's shell pairs.
    pub fn n_pairs(&self) -> usize {
        self.classes.iter().map(|c| c.bra.len()).sum()
    }

    /// The largest class's records.
    pub fn n_max(&self) -> i32 {
        self.classes
            .iter()
            .map(ClassPairs::n_rec)
            .max()
            .unwrap_or(0)
    }
}

/// A set of a class's record fields: the ones a family's kernels read. A
/// reader's layout declares exactly these ([`ClassRows::read`]), so the
/// rows its kernels never touch are not edges into it, and a launcher
/// hands its kernel a view with these pointers and null for the rest.
///
/// Each family's set is every field its forked kernels dereference
/// through the view, and no other: a test scans
/// `integrals/kernels/one_electron/<family>/` and holds each set to it,
/// which is what makes the null pointers sound.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Fields(u32);

impl Fields {
    /// The overlap's: ζ, P − A, P − B, `s` and both AO offsets.
    pub const OVERLAP: Self = Self(
        1 << F_ZETA | 0b111 << F_PA_X | 0b111 << F_PB_X | 1 << F_S | 1 << I_OFF_A | 1 << I_OFF_B,
    );
    /// The kinetic energy's: the overlap's, and β.
    pub const KINETIC: Self = Self(Self::OVERLAP.0 | 1 << F_EB);
    /// The nuclear attraction's: the overlap's, and P.
    pub const POTENTIAL_EN: Self = Self(Self::OVERLAP.0 | 0b111 << F_P_X);

    /// The fields of either set: a task that launches two families over
    /// one class, as the core Hamiltonian does, reads both.
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether the field at `bit` is in the set.
    const fn has(self, bit: usize) -> bool {
        self.0 & 1 << bit != 0
    }
}

// Each field's bit in a [`Fields`]: the double fields in
// [`ClassRows::f64_rows`]'s order, then the integer ones in
// [`ClassRows::i32_rows`]'s.
const F_ZETA: usize = 0;
const F_P_X: usize = 1;
const F_PA_X: usize = 4;
const F_PB_X: usize = 7;
const F_S: usize = 10;
const F_EB: usize = 12;
const I_OFF_A: usize = 16;
const I_OFF_B: usize = 17;

/// One class's records on the device: its momenta, its record count, the
/// Cartesian AO count of its basis, and a row per field, each `n_rec`
/// long, all outputs of the records task. A reader declares that it reads
/// the fields its family uses with [`ClassRows::read`], and hands the
/// class to a launcher:
/// [`overlap_class`](crate::overlap_class),
/// [`kinetic_class`](crate::kinetic_class) or
/// [`potential_en_class`](crate::potential_en_class).
///
/// The fields are the C++ `PrimitivePairClass`'s (its README's design has
/// each one), in descending ζ order, equal ζ in enumeration order (shell
/// pair, then bra primitive, then ket primitive).
///
/// **Why a launcher may trust it.** A class kernel adds into
/// `m[off_a + i][off_b + j]` at the AO offsets the records hold, and those
/// are device data, which no host check can read. So the type carries
/// them: its fields are private, and the one place that makes a
/// `ClassRows` is [`add_record_tasks`], which adds, with it, the two tasks
/// that write its rows.
///
/// - **What the rows hold.** The records task launches the order, the
///   records and the bounds into every row on its stream, from tables
///   [`RecordTables::new`] checked, so every offset is a shell's
///   `ao_offset` with its block inside `[0, n_cart)`, and the records
///   carry that `n_cart`.
/// - **Who writes them.** Each row is an output of the records task, so no
///   other task can write it: [`Bufs::take_mut`] refuses a row the task's
///   layout did not declare as its own output. Copying a `ClassRows`
///   copies the names of rows, not their contents.
/// - **When a reader sees them.** A launcher fetches each row through the
///   reading task's [`Bufs::get`], which panics unless that task's layout
///   declared the read. A declared read is an edge: the executor runs the
///   reader only after the records task succeeded (a failure skips every
///   descendant), orders the reader's stream work after the records' (the
///   same stream, or an event on it), and keeps the rows' bytes from any
///   other row until the reader is done.
/// - **Which plan.** A row names its table, so a `ClassRows` of another
///   plan or another worker panics in [`Dag::add`] or at [`Bufs::get`]
///   rather than reading some other row as records.
///
/// What the launchers check, in every build, is the rest: the output is
/// `n_cart²` long and `n_cart²` fits the kernels' 32-bit index. The one
/// thing assumed rather than checked is the one the arena's own
/// `DeviceSlice` rows assume: the launch goes on the stream of the task
/// that reads the rows, the stream the executor ordered.
#[derive(Clone, Copy, Debug)]
pub struct ClassRows {
    la: i32,
    lb: i32,
    n_rec: i32,
    n_cart: usize,
    zeta: RowId<Device, f64>,
    p_x: RowId<Device, f64>,
    p_y: RowId<Device, f64>,
    p_z: RowId<Device, f64>,
    pa_x: RowId<Device, f64>,
    pa_y: RowId<Device, f64>,
    pa_z: RowId<Device, f64>,
    pb_x: RowId<Device, f64>,
    pb_y: RowId<Device, f64>,
    pb_z: RowId<Device, f64>,
    s: RowId<Device, f64>,
    ea: RowId<Device, f64>,
    eb: RowId<Device, f64>,
    bound: RowId<Device, f64>,
    bound_ab: RowId<Device, f64>,
    bound_ba: RowId<Device, f64>,
    off_a: RowId<Device, i32>,
    off_b: RowId<Device, i32>,
    rank_a: RowId<Device, i32>,
    rank_b: RowId<Device, i32>,
}

impl ClassRows {
    /// The bra shell's momentum, at most `lb`.
    pub fn la(&self) -> i32 {
        self.la
    }

    /// The ket shell's momentum.
    pub fn lb(&self) -> i32 {
        self.lb
    }

    /// The number of records: every primitive product of every shell pair
    /// of the class, none screened.
    pub fn n_rec(&self) -> i32 {
        self.n_rec
    }

    /// The Cartesian AOs of the basis, which the records' offsets index: a
    /// launcher's matrix is `n_cart²`.
    pub fn n_cart(&self) -> usize {
        self.n_cart
    }

    /// The double fields' rows, in this order: ζ = α + β; P (x, y, z);
    /// P − A (x, y, z); P − B (x, y, z); `s`, c_a c_b (π/ζ)^{3/2}; α, the
    /// bra exponent; β, the ket exponent; the undressed Schwarz bound,
    /// `pair_bound(ζ, la + lb)`; the dressed bound as stored; the dressed
    /// bound swapped.
    pub fn f64_rows(&self) -> [RowId<Device, f64>; 16] {
        [
            self.zeta,
            self.p_x,
            self.p_y,
            self.p_z,
            self.pa_x,
            self.pa_y,
            self.pa_z,
            self.pb_x,
            self.pb_y,
            self.pb_z,
            self.s,
            self.ea,
            self.eb,
            self.bound,
            self.bound_ab,
            self.bound_ba,
        ]
    }

    /// The integer fields' rows, in this order: the bra shell's first
    /// Cartesian AO, the ket shell's, the bra shell's rank among the
    /// basis's shells of momentum `la`, the ket shell's among those of
    /// `lb`.
    pub fn i32_rows(&self) -> [RowId<Device, i32>; 4] {
        [self.off_a, self.off_b, self.rank_a, self.rank_b]
    }

    /// Declare, in a reader's layout, that it reads `fields`, its
    /// family's ([`Fields`]): what a launcher of that family then fetches.
    pub fn read(&self, device: &mut Layout<'_, Device>, fields: Fields) {
        for (bit, row) in self.f64_rows().into_iter().enumerate() {
            if fields.has(bit) {
                device.read(row);
            }
        }
        for (bit, row) in (16..).zip(self.i32_rows()) {
            if fields.has(bit) {
                device.read(row);
            }
        }
    }

    /// The class as the C ABI takes it for a kernel that reads `fields`:
    /// each of those a row of `device`, every other pointer null.
    ///
    /// # Panics
    /// If a field of `fields` was not declared by the reading task's
    /// layout ([`Bufs::get`]).
    pub(crate) fn view(
        &self,
        device: &Bufs<'_, Device>,
        fields: Fields,
    ) -> fb_primitive_pair_class_view {
        let n_rec = self.n_rec as usize;
        let rows = self.f64_rows();
        let f = |bit: usize| {
            if !fields.has(bit) {
                return std::ptr::null();
            }
            let row = device.get(rows[bit]);
            // INVARIANT: the records task declared every field n_rec long.
            debug_assert_eq!(row.len(), n_rec);
            row.as_ptr()
        };
        let rows = self.i32_rows();
        let i = |bit: usize| {
            if !fields.has(bit) {
                return std::ptr::null();
            }
            let row = device.get(rows[bit - 16]);
            // INVARIANT: as above.
            debug_assert_eq!(row.len(), n_rec);
            row.as_ptr()
        };
        fb_primitive_pair_class_view {
            la: self.la,
            lb: self.lb,
            n_rec: self.n_rec,
            zeta: f(0),
            p_x: f(1),
            p_y: f(2),
            p_z: f(3),
            pa_x: f(4),
            pa_y: f(5),
            pa_z: f(6),
            pb_x: f(7),
            pb_y: f(8),
            pb_z: f(9),
            s: f(10),
            ea: f(11),
            eb: f(12),
            bound: f(13),
            bound_ab: f(14),
            bound_ba: f(15),
            off_a: i(16),
            off_b: i(17),
            rank_a: i(18),
            rank_b: i(19),
        }
    }
}

/// One class of two bases' records ([`add_mixed_record_tasks`]): every
/// ordered shell pair, a bra-basis shell with a ket-basis shell, never
/// swapped, so `la > lb` classes appear; `off_a` indexes the bra basis's
/// Cartesian AOs and `off_b` the ket basis's, and each rank is among its
/// own basis's shells. The fields, their order and their bounds are a
/// [`ClassRows`]', which this is not: a one-basis build reads a basis's
/// `i <= j` pairs and mirrors each value, so a two-basis class handed to
/// one would build the wrong matrix, and no one-basis launcher takes this
/// type. What makes its offsets trustworthy is [`ClassRows`]' argument,
/// with each offset's block inside its own basis's AOs
/// ([`RecordTables::mixed`]).
#[derive(Clone, Copy, Debug)]
pub struct MixedClassRows {
    rows: ClassRows,
    n_cart_ket: usize,
}

impl MixedClassRows {
    /// The bra shell's momentum.
    pub fn la(&self) -> i32 {
        self.rows.la
    }

    /// The ket shell's momentum, below `la` or not.
    pub fn lb(&self) -> i32 {
        self.rows.lb
    }

    /// The number of records: every primitive product of every ordered
    /// shell pair of the class.
    pub fn n_rec(&self) -> i32 {
        self.rows.n_rec
    }

    /// The Cartesian AOs the bra offsets index, and the ket offsets: a
    /// build's matrix is `n_cart[0] × n_cart[1]`.
    pub fn n_cart(&self) -> [usize; 2] {
        [self.rows.n_cart, self.n_cart_ket]
    }

    /// The double fields' rows, as [`ClassRows::f64_rows`] orders them.
    pub fn f64_rows(&self) -> [RowId<Device, f64>; 16] {
        self.rows.f64_rows()
    }

    /// The integer fields' rows, as [`ClassRows::i32_rows`] orders them.
    pub fn i32_rows(&self) -> [RowId<Device, i32>; 4] {
        self.rows.i32_rows()
    }
}

/// Add the records of `tables` to `dag` on GPU worker `worker`: a CPU task,
/// `mmd tables`, that writes the tables into pinned rows, and a stream
/// task, `mmd records`, that copies them to device scratch and forms every
/// class from them. Returns each class's rows, in the tables' order;
/// tables with no class add nothing and return none.
///
/// Every size is the tables', fixed here at planning: a class's rows are
/// its `n_rec` long, and the sort's workspace is
/// [`fb_mmd_order_workspace_bytes`] of the largest class, a host query.
///
/// # Panics
/// If `tables` are two bases' ([`RecordTables::mixed`]): their ket offsets
/// index another basis's AOs than the `n_cart` a [`ClassRows`] promises
/// its launchers. [`add_mixed_record_tasks`] takes them.
#[track_caller]
pub fn add_record_tasks(dag: &mut Dag<'_>, worker: usize, tables: RecordTables) -> Vec<ClassRows> {
    assert!(
        tables.n_bra_shells.is_none(),
        "add_record_tasks: the tables are two bases'; add_mixed_record_tasks takes them"
    );
    record_tasks(dag, worker, tables)
}

/// Add the records of two bases' `tables` ([`RecordTables::mixed`]) to
/// `dag` on GPU worker `worker`, as [`add_record_tasks`] adds one basis's:
/// the same two tasks, every field formed the same way, in a class of any
/// order. Returns each class's rows as [`MixedClassRows`], which no
/// one-basis launcher takes.
///
/// # Panics
/// If `tables` are one basis's ([`RecordTables::new`]).
#[track_caller]
pub fn add_mixed_record_tasks(
    dag: &mut Dag<'_>,
    worker: usize,
    tables: RecordTables,
) -> Vec<MixedClassRows> {
    assert!(
        tables.n_bra_shells.is_some(),
        "add_mixed_record_tasks: the tables are one basis's; add_record_tasks takes them"
    );
    let n_cart_ket = tables.n_cart[1];
    record_tasks(dag, worker, tables)
        .into_iter()
        .map(|rows| MixedClassRows { rows, n_cart_ket })
        .collect()
}

/// [`add_record_tasks`]' and [`add_mixed_record_tasks`]' two tasks, each
/// class's rows carrying the bra offsets' `n_cart`.
#[track_caller]
fn record_tasks(dag: &mut Dag<'_>, worker: usize, tables: RecordTables) -> Vec<ClassRows> {
    if tables.classes.is_empty() {
        return Vec::new();
    }
    let tables = Arc::new(tables);
    let staged = dag.add(
        "mmd tables",
        worker,
        Stage {
            tables: Arc::clone(&tables),
        },
    );
    dag.add("mmd records", worker, Records { tables, staged })
        .classes
}

/// The eleven tables, one row each, of one kind: pinned as the CPU task's
/// outputs, device as the stream task's scratch.
struct TableRows<K> {
    prim_begin: RowId<K, i32>,
    exponent: RowId<K, f64>,
    coefficient: RowId<K, f64>,
    c_x: RowId<K, f64>,
    c_y: RowId<K, f64>,
    c_z: RowId<K, f64>,
    ao_offset: RowId<K, i32>,
    rank: RowId<K, i32>,
    bra: RowId<K, u32>,
    ket: RowId<K, u32>,
    /// Every class's prefix, one after the other: class `c`'s
    /// `n_pairs + 1` entries start at its first pair's index plus `c`.
    rec_begin: RowId<K, i32>,
}

impl<K> Clone for TableRows<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K> Copy for TableRows<K> {}

impl<K: Kind> TableRows<K> {
    /// Declare the eleven rows, as outputs or as scratch.
    fn declare(l: &mut Layout<'_, K>, t: &RecordTables, output: bool) -> Self {
        fn row<K: Kind, T: Element>(
            l: &mut Layout<'_, K>,
            name: &str,
            len: usize,
            output: bool,
        ) -> RowId<K, T> {
            if output {
                l.output(name, len)
            } else {
                l.scratch(name, len)
            }
        }
        let s = &t.shells;
        let n_pairs = t.n_pairs();
        TableRows {
            prim_begin: row(l, "prim_begin", s.prim_begin.len(), output),
            exponent: row(l, "exponent", s.exponent.len(), output),
            coefficient: row(l, "coefficient", s.coefficient.len(), output),
            c_x: row(l, "c_x", s.c_x.len(), output),
            c_y: row(l, "c_y", s.c_y.len(), output),
            c_z: row(l, "c_z", s.c_z.len(), output),
            ao_offset: row(l, "ao_offset", s.ao_offset.len(), output),
            rank: row(l, "rank", s.rank.len(), output),
            bra: row(l, "bra", n_pairs, output),
            ket: row(l, "ket", n_pairs, output),
            rec_begin: row(l, "rec_begin", n_pairs + t.classes.len(), output),
        }
    }
}

/// CPU: the host tables written into pinned rows.
struct Stage {
    tables: Arc<RecordTables>,
}

impl Task for Stage {
    type Rows = TableRows<Pinned>;
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        TableRows::declare(host, &self.tables, true)
    }

    fn run(
        &self,
        rows: &Self::Rows,
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        let s = &self.tables.shells;
        memcpy_h2p(host, rows.prim_begin, &s.prim_begin);
        memcpy_h2p(host, rows.exponent, &s.exponent);
        memcpy_h2p(host, rows.coefficient, &s.coefficient);
        memcpy_h2p(host, rows.c_x, &s.c_x);
        memcpy_h2p(host, rows.c_y, &s.c_y);
        memcpy_h2p(host, rows.c_z, &s.c_z);
        memcpy_h2p(host, rows.ao_offset, &s.ao_offset);
        memcpy_h2p(host, rows.rank, &s.rank);
        let bra = host.take_mut(rows.bra);
        let ket = host.take_mut(rows.ket);
        let rec_begin = host.take_mut(rows.rec_begin);
        let (mut p, mut r) = (0, 0);
        for c in &self.tables.classes {
            let n = c.bra.len();
            bra[p..p + n].copy_from_slice(&c.bra);
            ket[p..p + n].copy_from_slice(&c.ket);
            rec_begin[r..=r + n].copy_from_slice(&c.rec_begin);
            p += n;
            r += n + 1;
        }
        Ok(())
    }
}

/// GPU: the tables copied to device scratch, then each class, in order, on
/// the one stream: its order (key and stable sort, in scratch sized to the
/// largest class), its records, its bound and its two dressed bounds.
struct Records {
    tables: Arc<RecordTables>,
    staged: TableRows<Pinned>,
}

#[derive(Clone)]
struct RecordsRows {
    tables: TableRows<Device>,
    key: RowId<Device, f64>,
    perm: RowId<Device, u32>,
    workspace: RowId<Device, u8>,
    classes: Vec<ClassRows>,
}

impl Task for Records {
    type Rows = RecordsRows;
    const GPU: bool = true;

    fn layout(
        &self,
        device: &mut Layout<'_, Device>,
        host: &mut Layout<'_, Pinned>,
    ) -> RecordsRows {
        let s = &self.staged;
        host.read(s.prim_begin);
        host.read(s.exponent);
        host.read(s.coefficient);
        host.read(s.c_x);
        host.read(s.c_y);
        host.read(s.c_z);
        host.read(s.ao_offset);
        host.read(s.rank);
        host.read(s.bra);
        host.read(s.ket);
        host.read(s.rec_begin);
        let t = &self.tables;
        let tables = TableRows::declare(device, t, false);
        let n_max = t.n_max() as usize;
        let key = device.scratch("order key", n_max);
        let perm = device.scratch("order perm", n_max);
        // SAFETY: a host function of its argument alone.
        let bytes = unsafe { fb_mmd_order_workspace_bytes(t.n_max()) };
        let workspace = device.scratch("order workspace", bytes);
        let classes = t
            .classes
            .iter()
            .map(|c| {
                let (la, lb, n) = (c.la, c.lb, c.n_rec() as usize);
                let mut f64_row =
                    |field: &str| device.output::<f64>(&format!("{field} ({la},{lb})"), n);
                let (zeta, p_x, p_y, p_z) = (
                    f64_row("zeta"),
                    f64_row("p_x"),
                    f64_row("p_y"),
                    f64_row("p_z"),
                );
                let (pa_x, pa_y, pa_z) = (f64_row("pa_x"), f64_row("pa_y"), f64_row("pa_z"));
                let (pb_x, pb_y, pb_z) = (f64_row("pb_x"), f64_row("pb_y"), f64_row("pb_z"));
                let (s, ea, eb) = (f64_row("s"), f64_row("ea"), f64_row("eb"));
                let (bound, bound_ab, bound_ba) =
                    (f64_row("bound"), f64_row("bound_ab"), f64_row("bound_ba"));
                let mut i32_row =
                    |field: &str| device.output::<i32>(&format!("{field} ({la},{lb})"), n);
                let (off_a, off_b, rank_a, rank_b) = (
                    i32_row("off_a"),
                    i32_row("off_b"),
                    i32_row("rank_a"),
                    i32_row("rank_b"),
                );
                ClassRows {
                    la,
                    lb,
                    n_rec: c.n_rec(),
                    n_cart: t.n_cart[0],
                    zeta,
                    p_x,
                    p_y,
                    p_z,
                    pa_x,
                    pa_y,
                    pa_z,
                    pb_x,
                    pb_y,
                    pb_z,
                    s,
                    ea,
                    eb,
                    bound,
                    bound_ab,
                    bound_ba,
                    off_a,
                    off_b,
                    rank_a,
                    rank_b,
                }
            })
            .collect();
        RecordsRows {
            tables,
            key,
            perm,
            workspace,
            classes,
        }
    }

    fn run(
        &self,
        rows: &RecordsRows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream: worker, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let stream = worker.stream();
        let (d, s) = (&rows.tables, &self.staged);
        let prim_begin = upload(worker, device, d.prim_begin, host, s.prim_begin)?;
        let exponent = upload(worker, device, d.exponent, host, s.exponent)?;
        let coefficient = upload(worker, device, d.coefficient, host, s.coefficient)?;
        let c_x = upload(worker, device, d.c_x, host, s.c_x)?;
        let c_y = upload(worker, device, d.c_y, host, s.c_y)?;
        let c_z = upload(worker, device, d.c_z, host, s.c_z)?;
        let ao_offset = upload(worker, device, d.ao_offset, host, s.ao_offset)?;
        let rank = upload(worker, device, d.rank, host, s.rank)?;
        let bra = upload(worker, device, d.bra, host, s.bra)?;
        let ket = upload(worker, device, d.ket, host, s.ket)?;
        let rec_begin = upload(worker, device, d.rec_begin, host, s.rec_begin)?;

        let key = device.take_mut(rows.key);
        let perm = device.take_mut(rows.perm);
        let workspace = device.take_mut(rows.workspace);
        let mut first = 0;
        for (c, (pairs, cls)) in self.tables.classes.iter().zip(&rows.classes).enumerate() {
            let n_pairs = pairs.bra.len();
            let class = [cls.la, cls.lb];
            let src = fb_primitive_pair_source {
                n_pairs: n_pairs as i32,
                // In bounds: the class's pairs, and its prefix, lie inside
                // the concatenated rows (TableRows::rec_begin).
                bra: bra.wrapping_add(first),
                ket: ket.wrapping_add(first),
                rec_begin: rec_begin.wrapping_add(first + c),
                prim_begin,
                exponent,
                coefficient,
                c_x,
                c_y,
                c_z,
                ao_offset,
                rank,
            };
            first += n_pairs;
            let n = cls.n_rec;
            // SAFETY: key and perm are n_max >= n long, the workspace is the
            // largest class's query, every array of src is device memory
            // the uploads above wrote on this stream from tables
            // RecordTables::new checked (each index the kernels follow is
            // inside its array), and none overlap: each is a row of its own.
            launched("fb_mmd_order", &class, unsafe {
                fb_mmd_order(
                    stream,
                    n,
                    &src,
                    key.as_mut_ptr(),
                    perm.as_mut_ptr(),
                    workspace.as_mut_ptr().cast(),
                    workspace.len(),
                )
            })?;
            let f = |row| device.take_mut(row);
            let (zeta, p_x, p_y, p_z) = (f(cls.zeta), f(cls.p_x), f(cls.p_y), f(cls.p_z));
            let (pa_x, pa_y, pa_z) = (f(cls.pa_x), f(cls.pa_y), f(cls.pa_z));
            let (pb_x, pb_y, pb_z) = (f(cls.pb_x), f(cls.pb_y), f(cls.pb_z));
            let (sv, ea, eb) = (f(cls.s), f(cls.ea), f(cls.eb));
            let (mut bound, mut bound_ab, mut bound_ba) =
                (f(cls.bound), f(cls.bound_ab), f(cls.bound_ba));
            let g = |row| device.take_mut(row);
            let (off_a, off_b, rank_a, rank_b) =
                (g(cls.off_a), g(cls.off_b), g(cls.rank_a), g(cls.rank_b));
            let out = fb_primitive_pair_records {
                zeta: zeta.as_mut_ptr(),
                p_x: p_x.as_mut_ptr(),
                p_y: p_y.as_mut_ptr(),
                p_z: p_z.as_mut_ptr(),
                pa_x: pa_x.as_mut_ptr(),
                pa_y: pa_y.as_mut_ptr(),
                pa_z: pa_z.as_mut_ptr(),
                pb_x: pb_x.as_mut_ptr(),
                pb_y: pb_y.as_mut_ptr(),
                pb_z: pb_z.as_mut_ptr(),
                s: sv.as_mut_ptr(),
                ea: ea.as_mut_ptr(),
                eb: eb.as_mut_ptr(),
                off_a: off_a.as_mut_ptr(),
                off_b: off_b.as_mut_ptr(),
                rank_a: rank_a.as_mut_ptr(),
                rank_b: rank_b.as_mut_ptr(),
            };
            // SAFETY: perm is the order just written, a permutation of the
            // class's n host records; src is as above; and every array of
            // out is an n-long row of its own.
            launched("fb_mmd_records", &class, unsafe {
                fb_mmd_records(stream, n, &src, perm.as_ptr(), &out)
            })?;

            // A class above a family's table keeps that family's bounds
            // zero, as the C++ does; every build refuses such a class. The
            // pair bound depends on la + lb alone, so a two-basis class
            // with la > lb takes (lb, la)'s kernel, the one emitted.
            let (lo, hi) = (cls.la.min(cls.lb), cls.la.max(cls.lb));
            if hi <= FB_PAIR_BOUND_MAX_MOMENTUM {
                // SAFETY: zeta and bound are n-long rows of their own.
                launched("fb_mmd_bound", &class, unsafe {
                    fb_mmd_bound(stream, lo, hi, n, zeta.as_ptr(), bound.as_mut_ptr())
                })?;
            } else {
                memset::zero(&mut bound, worker)?;
            }
            if hi <= FB_RECORD_BOUND_MAX_MOMENTUM {
                let view = fb_primitive_pair_class_view {
                    la: cls.la,
                    lb: cls.lb,
                    n_rec: n,
                    zeta: zeta.as_ptr(),
                    p_x: p_x.as_ptr(),
                    p_y: p_y.as_ptr(),
                    p_z: p_z.as_ptr(),
                    pa_x: pa_x.as_ptr(),
                    pa_y: pa_y.as_ptr(),
                    pa_z: pa_z.as_ptr(),
                    pb_x: pb_x.as_ptr(),
                    pb_y: pb_y.as_ptr(),
                    pb_z: pb_z.as_ptr(),
                    s: sv.as_ptr(),
                    ea: ea.as_ptr(),
                    eb: eb.as_ptr(),
                    bound: bound.as_ptr(),
                    bound_ab: bound_ab.as_ptr(),
                    bound_ba: bound_ba.as_ptr(),
                    off_a: off_a.as_ptr(),
                    off_b: off_b.as_ptr(),
                    rank_a: rank_a.as_ptr(),
                    rank_b: rank_b.as_ptr(),
                };
                // The swap: la and lb, and P - A and P - B, exchanged.
                let swapped = fb_primitive_pair_class_view {
                    la: cls.lb,
                    lb: cls.la,
                    pa_x: view.pb_x,
                    pa_y: view.pb_y,
                    pa_z: view.pb_z,
                    pb_x: view.pa_x,
                    pb_y: view.pa_y,
                    pb_z: view.pa_z,
                    ..view
                };
                // SAFETY: the view's arrays are the class's n-long rows,
                // written above on this stream, and each bound is a row of
                // its own that the kernel only writes.
                launched("fb_ordered_pair_bound", &class, unsafe {
                    fb_ordered_pair_bound(stream, &view, bound_ab.as_mut_ptr())
                })?;
                // SAFETY: as above, the view swapped.
                launched("fb_ordered_pair_bound (swapped)", &class, unsafe {
                    fb_ordered_pair_bound(stream, &swapped, bound_ba.as_mut_ptr())
                })?;
            } else {
                memset::zero(&mut bound_ab, worker)?;
                memset::zero(&mut bound_ba, worker)?;
            }
        }
        Ok(())
    }
}

/// The pinned row `src` copied on `worker`'s stream to the device row
/// `dst` of the same length; returns `dst`'s pointer, for the launches to
/// read.
fn upload<T: Element>(
    worker: &DeviceStream,
    device: &Bufs<'_, Device>,
    dst: RowId<Device, T>,
    host: &Bufs<'_, Pinned>,
    src: RowId<Pinned, T>,
) -> Result<*const T> {
    let dst = device.take_mut(dst);
    memcpy_p2d(&dst, host, src, worker)?;
    Ok(dst.as_ptr())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The view's pointer fields, each at its bit in a [`Fields`].
    const VIEW_FIELDS: [&str; 20] = [
        "zeta", "p_x", "p_y", "p_z", "pa_x", "pa_y", "pa_z", "pb_x", "pb_y", "pb_z", "s", "ea",
        "eb", "bound", "bound_ab", "bound_ba", "off_a", "off_b", "rank_a", "rank_b",
    ];

    /// The view's pointer fields that the forked sources under
    /// `integrals/kernels/one_electron/<family>/` dereference: `name->field`
    /// or `name.field` for every `name` declared there as a
    /// `fb_primitive_pair_class_view` pointer or reference.
    fn dereferenced(family: &str) -> std::collections::BTreeSet<String> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../kernels/one_electron")
            .join(family);
        let mut sources = Vec::new();
        let mut todo = vec![dir.clone()];
        while let Some(d) = todo.pop() {
            for entry in std::fs::read_dir(&d).unwrap_or_else(|e| panic!("{d:?}: {e}")) {
                let path = entry.unwrap().path();
                if path.is_dir() {
                    todo.push(path);
                } else if path
                    .extension()
                    .is_some_and(|x| ["cu", "h", "cuh"].iter().any(|e| x == *e))
                {
                    sources.push(std::fs::read_to_string(&path).unwrap());
                }
            }
        }
        let text = sources.join("\n");
        // Every name a view is bound to: `fb_primitive_pair_class_view *cls`,
        // `... &cls`, `const fb_primitive_pair_class_view *const cls`.
        let mut names = std::collections::BTreeSet::new();
        for (at, _) in text.match_indices("fb_primitive_pair_class_view") {
            let rest = &text[at + "fb_primitive_pair_class_view".len()..];
            let rest = rest.trim_start_matches(|c: char| c.is_whitespace() || "*&".contains(c));
            let rest = rest.strip_prefix("const").unwrap_or(rest).trim_start();
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !name.is_empty() {
                names.insert(name);
            }
        }
        assert!(
            !names.is_empty(),
            "{family}: no fb_primitive_pair_class_view in {dir:?}"
        );
        let mut used = std::collections::BTreeSet::new();
        for name in &names {
            for sep in ["->", "."] {
                let pat = format!("{name}{sep}");
                for (at, _) in text.match_indices(&pat) {
                    let before = text[..at].chars().next_back();
                    if before.is_some_and(|c| c.is_alphanumeric() || c == '_') {
                        continue;
                    }
                    let field: String = text[at + pat.len()..]
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    if VIEW_FIELDS.contains(&field.as_str()) {
                        used.insert(field);
                    }
                }
            }
        }
        used
    }

    /// Each family's [`Fields`] is exactly the view fields its forked
    /// kernels dereference. Every other pointer a launcher hands that
    /// family is null, so a kernel that starts reading another field fails
    /// here before it can read a null one on the device.
    #[test]
    fn each_family_reads_exactly_its_fields() {
        for (family, fields) in [
            ("overlap", Fields::OVERLAP),
            ("kinetic", Fields::KINETIC),
            ("potential_en", Fields::POTENTIAL_EN),
        ] {
            let set: std::collections::BTreeSet<String> = VIEW_FIELDS
                .iter()
                .enumerate()
                .filter(|&(bit, _)| fields.has(bit))
                .map(|(_, f)| f.to_string())
                .collect();
            assert_eq!(dereferenced(family), set, "{family}");
        }
    }

    /// The bits name the view's fields in [`ClassRows::f64_rows`]'s and
    /// [`ClassRows::i32_rows`]'s order.
    #[test]
    fn the_field_bits_follow_the_rows() {
        let bit = |name: &str| VIEW_FIELDS.iter().position(|f| *f == name).unwrap();
        assert_eq!(
            [F_ZETA, F_P_X, F_PA_X, F_PB_X, F_S, F_EB, I_OFF_A, I_OFF_B],
            ["zeta", "p_x", "pa_x", "pb_x", "s", "eb", "off_a", "off_b"].map(bit)
        );
    }

    /// Two shells, an s of two primitives at AO 0 and a p of one at AOs
    /// 1..4, and their three pairs in classes (0,0), (0,1) and (1,1).
    fn parts() -> (ShellTable, Vec<ClassPairs>) {
        let shells = ShellTable {
            prim_begin: vec![0, 2, 3],
            exponent: vec![3.0, 0.5, 1.0],
            coefficient: vec![0.4, 0.6, 1.0],
            c_x: vec![0.0; 2],
            c_y: vec![0.0; 2],
            c_z: vec![0.0, 1.0],
            ao_offset: vec![0, 1],
            rank: vec![0, 0],
        };
        let class = |la, lb, bra, ket, k| ClassPairs {
            la,
            lb,
            bra: vec![bra],
            ket: vec![ket],
            rec_begin: vec![0, k],
        };
        let classes = vec![
            class(0, 0, 0, 0, 4),
            class(0, 1, 0, 1, 2),
            class(1, 1, 1, 1, 1),
        ];
        (shells, classes)
    }

    /// Consistent tables are accepted as they are.
    #[test]
    fn consistent_tables_are_kept() {
        let (shells, classes) = parts();
        let t = RecordTables::new(shells.clone(), classes.clone(), 4);
        assert_eq!(
            (t.shells(), t.classes(), t.n_cart()),
            (&shells, &classes[..], 4)
        );
        assert_eq!((t.n_pairs(), t.n_max()), (3, 4));
    }

    /// A p shell whose block runs past the matrix: its offset is device data
    /// a class kernel would write at.
    #[test]
    #[should_panic(expected = "shell 1 at AO 1 with momentum 1 does not fit 3 Cartesian AOs")]
    fn a_block_past_the_matrix_panics() {
        let (shells, classes) = parts();
        RecordTables::new(shells, classes, 3);
    }

    /// The tables of [`parts`] as two bases: shells 0 and 1 the bra
    /// basis's (4 Cartesians), and a copy of the s as the ket basis's
    /// (1 Cartesian, at its own AO 0), one class (0, 0) and one (1, 0).
    fn mixed_parts() -> (ShellTable, Vec<ClassPairs>) {
        let (mut shells, _) = parts();
        shells.prim_begin.push(5);
        shells.exponent.extend([3.0, 0.5]);
        shells.coefficient.extend([0.4, 0.6]);
        shells.c_x.push(0.0);
        shells.c_y.push(0.0);
        shells.c_z.push(2.0);
        shells.ao_offset.push(0);
        shells.rank.push(0);
        let class = |la, lb, bra, ket, k| ClassPairs {
            la,
            lb,
            bra: vec![bra],
            ket: vec![ket],
            rec_begin: vec![0, k],
        };
        (shells, vec![class(0, 0, 0, 2, 4), class(1, 0, 1, 2, 2)])
    }

    /// Two bases' consistent tables are accepted, each offset checked in
    /// its own basis's AOs: the ket's AO 0 of 1 is inside.
    #[test]
    fn consistent_mixed_tables_are_kept() {
        let (shells, classes) = mixed_parts();
        let t = RecordTables::mixed(shells, classes, 2, [4, 1]);
        assert_eq!((t.n_cart(), t.n_pairs(), t.n_max()), (4, 2, 4));
    }

    /// A ket shell whose block runs past the ket basis's AOs, though it
    /// would fit the bra basis's.
    #[test]
    #[should_panic(expected = "shell 2 at AO 0 with momentum 1 does not fit 1 Cartesian AOs")]
    fn a_ket_block_past_its_own_basis_panics() {
        let (mut shells, mut classes) = mixed_parts();
        shells.ao_offset[2] = 0;
        classes[1] = ClassPairs {
            la: 0,
            lb: 1,
            bra: vec![0],
            ket: vec![2],
            rec_begin: vec![0, 4],
        };
        RecordTables::mixed(shells, classes, 2, [4, 1]);
    }

    /// A pair whose ket is a bra-basis shell: its offset indexes the other
    /// basis's AOs.
    #[test]
    #[should_panic(expected = "pair 0 pairs shells 0 and 1, not a bra shell below 2")]
    fn a_pair_within_the_bra_basis_panics() {
        let (shells, mut classes) = mixed_parts();
        classes[0] = ClassPairs {
            la: 0,
            lb: 1,
            bra: vec![0],
            ket: vec![1],
            rec_begin: vec![0, 2],
        };
        RecordTables::mixed(shells, classes, 2, [4, 1]);
    }

    /// Two bases' tables are not one basis's records: a `ClassRows`
    /// promises its launchers one `n_cart` for both offsets.
    #[test]
    #[should_panic(expected = "add_record_tasks: the tables are two bases'")]
    fn one_basis_records_refuse_two_bases_tables() {
        let (shells, classes) = mixed_parts();
        let mut dag = Dag::new();
        add_record_tasks(&mut dag, 0, RecordTables::mixed(shells, classes, 2, [4, 1]));
    }

    /// And one basis's tables are not two bases'.
    #[test]
    #[should_panic(expected = "add_mixed_record_tasks: the tables are one basis's")]
    fn two_bases_records_refuse_one_basis_tables() {
        let (shells, classes) = parts();
        let mut dag = Dag::new();
        add_mixed_record_tasks(&mut dag, 0, RecordTables::new(shells, classes, 4));
    }

    /// A pair naming a shell past the table.
    #[test]
    #[should_panic(expected = "class (0, 1) pair 0 names shells 0 and 2 of 2")]
    fn a_shell_past_the_table_panics() {
        let (shells, mut classes) = parts();
        classes[1].ket[0] = 2;
        RecordTables::new(shells, classes, 4);
    }

    /// A prefix that does not step by the pair's primitive product: the
    /// kernels would read primitives past the pair's shells.
    #[test]
    #[should_panic(expected = "class (0, 0) pair 0's records are not its primitive product")]
    fn a_prefix_past_the_primitives_panics() {
        let (shells, mut classes) = parts();
        classes[0].rec_begin[1] = 5;
        RecordTables::new(shells, classes, 4);
    }

    /// `prim_begin` ending short of the exponents.
    #[test]
    #[should_panic(expected = "prim_begin is not a prefix of the 3 primitives")]
    fn a_short_primitive_prefix_panics() {
        let (mut shells, classes) = parts();
        shells.prim_begin[2] = 2;
        RecordTables::new(shells, classes, 4);
    }

    /// A matrix whose square passes the kernels' 32-bit index.
    #[test]
    #[should_panic(expected = "46341² Cartesian AOs do not fit the kernels' 32-bit index")]
    fn a_matrix_past_the_index_panics() {
        let (shells, classes) = parts();
        RecordTables::new(shells, classes, 46_341);
    }
}
