//! Adaptive widths' tests (tier 3): the bisection against a linear scan,
//! and plans whose widths are the largest admission accepts.

use std::sync::{Mutex, MutexGuard, OnceLock, PoisonError};

use proptest::prelude::*;

use super::tests::{Decl, Declares};
use super::*;
use fb_core_arena::{PinnedStandIn, admit};
use fb_test_support::golden;

/// A part of `width` columns of `column` bytes each, held `copies` times,
/// up to `max` columns.
struct Columns {
    column: usize,
    copies: usize,
    max: usize,
    floor: usize,
    pow2: bool,
}

impl Columns {
    fn new(column: usize, max: usize) -> Self {
        Columns {
            column,
            copies: 1,
            max,
            floor: 2,
            pow2: false,
        }
    }
}

impl Adaptive for Columns {
    fn width_max(&self) -> usize {
        self.max
    }

    fn bytes(&self, width: usize) -> usize {
        self.copies * self.column * width
    }

    fn warn_below(&self) -> usize {
        self.floor
    }

    fn power_of_two(&self) -> bool {
        self.pow2
    }
}

/// A part whose bytes at width `w` are `table[w - 1]`.
struct Tabled {
    table: Vec<usize>,
    pow2: bool,
}

impl Adaptive for Tabled {
    fn width_max(&self) -> usize {
        self.table.len()
    }

    fn bytes(&self, width: usize) -> usize {
        self.table[width - 1]
    }

    fn power_of_two(&self) -> bool {
        self.pow2
    }
}

/// The width a linear scan finds: the largest whose bytes fit, 1 if none,
/// rounded as [`widest`] rounds.
fn scan(part: &impl Adaptive, room: usize) -> usize {
    let width = (1..=part.width_max())
        .filter(|&w| part.bytes(w) <= room)
        .max()
        .unwrap_or(1);
    if part.power_of_two() {
        1 << width.ilog2()
    } else {
        width
    }
}

proptest! {
    /// Random monotone bytes, from random steps: the bisected width is the
    /// linear scan's, in rooms below, among and above the bytes.
    #[test]
    fn bisection_equals_a_linear_scan(
        base in 0usize..1 << 20,
        steps in prop::collection::vec(0usize..1 << 12, 1..200),
        room in 0usize..1 << 21,
        pow2 in any::<bool>(),
    ) {
        let table: Vec<usize> = steps
            .iter()
            .scan(base, |at, &s| {
                *at += s;
                Some(*at)
            })
            .collect();
        let part = Tabled { table, pow2 };
        prop_assert_eq!(widest(&part, room), scan(&part, room));
        for &bytes in &part.table {
            prop_assert_eq!(widest(&part, bytes), scan(&part, bytes));
        }
    }
}

/// A non-monotone `bytes` is caught: for some room the bisection and the
/// scan disagree, which is what a part's fuzz fails on.
#[test]
fn a_non_monotone_bytes_is_caught() {
    let part = Tabled {
        table: vec![10, 50, 20, 60, 70, 80, 90, 100],
        pow2: false,
    };
    assert_eq!(scan(&part, 30), 3);
    assert_eq!(widest(&part, 30), 1);
}

/// Every line logged at `warn` and `debug`, in order.
static LOGGED: Mutex<Vec<(log::Level, String)>> = Mutex::new(Vec::new());

struct Keeps;

impl log::Log for Keeps {
    fn enabled(&self, metadata: &log::Metadata<'_>) -> bool {
        metadata.level() <= log::Level::Debug
    }

    fn log(&self, record: &log::Record<'_>) {
        lock(&LOGGED).push((record.level(), record.args().to_string()));
    }

    fn flush(&self) {}
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// The lines logged naming the part `name`, after installing [`Keeps`] once.
fn logged(name: &str) -> Vec<(log::Level, String)> {
    let named = format!("width {name}:");
    lock(&LOGGED)
        .iter()
        .filter(|(_, line)| line.contains(&named))
        .cloned()
        .collect()
}

fn keep_logs() {
    static INSTALLED: OnceLock<()> = OnceLock::new();
    INSTALLED.get_or_init(|| {
        log::set_logger(&Keeps).expect("no other logger in this binary");
        log::set_max_level(log::LevelFilter::Debug);
    });
}

/// Doubles in 512 bytes, so a row of whole columns is padded to nothing.
const COLUMN: usize = 64;
/// A fixed row: 4096 bytes.
const N: usize = 512;

/// The DAG the tests declare, and what a width is asked of: [`Dag::width`]
/// in a plan, or a fixed number.
type Test = Dag<'static, PinnedStandIn, ()>;

/// One adaptive task between fixed layouts: `a` makes X, the part reads X
/// and holds `width` columns of scratch beside its output Y, `c` reads Y.
/// Returns the width.
fn between(dag: &mut Test, width: impl FnOnce(&mut Test) -> usize) -> usize {
    use Decl::*;
    let x = dag.add_gpu("a", 0, Declares(vec![Out("X", N)]))[0];
    let width = width(dag);
    let y = dag.add_gpu(
        "part",
        0,
        Declares(vec![Read(x), Scratch("batch", COLUMN * width), Out("Y", N)]),
    )[1];
    dag.add_gpu("c", 0, Declares(vec![Read(y), Out("Z", N)]));
    width
}

/// `between` planned at a fixed width, admitted against `room` or not.
fn admitted(width: usize, room: usize) -> bool {
    let mut dag = Dag::new();
    between(&mut dag, |_| width);
    admit(dag.plan().device_table(0), room, "gpu0").is_ok()
}

/// The width is the largest admission accepts and the next is refused;
/// it is a line of the plan's text, and narrowed below its floor it is
/// logged at `warn`.
#[test]
fn the_width_is_the_largest_admission_accepts() {
    keep_logs();
    let part = Columns {
        floor: 32,
        ..Columns::new(8 * COLUMN, 64)
    };
    let room = 2 * 8 * N + 20 * 8 * COLUMN + 100;
    let (planned, width) = Dag::plan_within(&[room], |dag| {
        between(dag, |dag| dag.width("between", 0, &part))
    });
    assert_eq!(width, 20);
    assert!(admit(planned.device_table(0), room, "gpu0").is_ok());
    assert!(admitted(width, room));
    assert!(!admitted(width + 1, room));
    golden(
        env!("CARGO_MANIFEST_DIR"),
        "planner_adaptive.txt",
        &planned.to_string(),
    );
    let lines = logged("between");
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert_eq!(lines[0].0, log::Level::Warn);
    assert!(lines[0].1.contains("20 of 64"), "{}", lines[0].1);
    assert!(lines[0].1.contains("below 32"), "{}", lines[0].1);
}

/// Planned with no room, a part is its natural width, logged at `debug`.
#[test]
fn no_room_is_the_natural_width() {
    keep_logs();
    let part = Columns::new(8 * COLUMN, 64);
    let mut dag = Test::new();
    let width = between(&mut dag, |dag| dag.width("natural", 0, &part));
    let planned = dag.plan();
    assert_eq!(width, 64);
    assert!(planned.to_string().contains("width natural: 64 of 64"));
    let lines = logged("natural");
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert_eq!(lines[0].0, log::Level::Debug);
    // A worker past the rooms given has none either.
    let (_, width) = Test::plan_within(&[], |dag| {
        between(dag, |dag| dag.width("past the rooms", 0, &part))
    });
    assert_eq!(width, 64);
}

/// The total a counted loop over batches cuts into `width`-column batches.
const TOTAL: usize = 1000;

/// A counted loop over `TOTAL` columns in batches of `width`, two in
/// flight, each batch staged and folded into an accumulator made before
/// the loop and read after it.
fn batched(dag: &mut Test, width: impl FnOnce(&mut Test) -> usize) -> usize {
    use Decl::*;
    let acc = dag.add_gpu("zero", 0, Declares(vec![Out("acc", N)]))[0];
    let width = width(dag);
    let body = dag.len();
    let batch = dag.add_gpu("stage", 0, Declares(vec![Out("batch", COLUMN * width)]))[0];
    dag.add_gpu("fold", 0, Declares(vec![Read(batch), Write(acc)]));
    let trips = TOTAL.div_ceil(width);
    dag.add_counted(body, trips, trips.min(2));
    dag.add_gpu("read", 0, Declares(vec![Read(acc), Out("out", N)]));
    width
}

/// A counted loop's trip count follows its width, rounded down to a power
/// of two, which still fits; below its floor it is logged at `warn`.
#[test]
fn a_counted_loop_takes_its_trips_from_the_width() {
    keep_logs();
    let part = Columns {
        copies: 2,
        floor: 32,
        pow2: true,
        ..Columns::new(8 * COLUMN, 64)
    };
    let room = 8 * N + 2 * 20 * 8 * COLUMN;
    let (planned, width) = Dag::plan_within(&[room], |dag| {
        batched(dag, |dag| dag.width("batched", 0, &part))
    });
    assert_eq!(width, 16);
    let text = planned.to_string();
    assert!(text.contains("63 trips, depth 2"), "{text}");
    assert!(admit(planned.device_table(0), room, "gpu0").is_ok());
    let mut wider = Dag::new();
    batched(&mut wider, |_| 32);
    assert!(admit(wider.plan().device_table(0), room, "gpu0").is_err());
    let lines = logged("batched");
    assert_eq!(lines.len(), 1, "{lines:?}");
    assert_eq!(lines[0].0, log::Level::Warn);
}

/// Two parts in schedule order: `p0` makes W0, `width0` columns read at
/// the end, and `p1` holds `width1` columns of scratch beside Y while W0
/// is live.
fn two(dag: &mut Test, mut width: impl FnMut(&mut Test, usize) -> usize) -> (usize, usize) {
    use Decl::*;
    let x = dag.add_gpu("a", 0, Declares(vec![Out("X", N)]))[0];
    let width0 = width(dag, 0);
    let w0 = dag.add_gpu("p0", 0, Declares(vec![Read(x), Out("W0", COLUMN * width0)]))[0];
    let width1 = width(dag, 1);
    let y = dag.add_gpu(
        "p1",
        0,
        Declares(vec![Scratch("S1", COLUMN * width1), Out("Y", N)]),
    )[1];
    dag.add_gpu("c", 0, Declares(vec![Read(w0), Read(y), Out("Z", N)]));
    (width0, width1)
}

/// The second part takes the remainder the first leaves at its position:
/// each width is the largest admission accepts with the other's.
#[test]
fn a_later_part_takes_the_remainder() {
    keep_logs();
    let (first, second) = (Columns::new(8 * COLUMN, 8), Columns::new(8 * COLUMN, 64));
    let room = 8 * N + 8 * 8 * COLUMN + 12 * 8 * COLUMN;
    let (planned, widths) = Dag::plan_within(&[room], |dag| {
        two(dag, |dag, i| match i {
            0 => dag.width("first", 0, &first),
            _ => dag.width("second", 0, &second),
        })
    });
    assert_eq!(widths, (8, 12));
    assert!(admit(planned.device_table(0), room, "gpu0").is_ok());
    let fits = |width0, width1| {
        let mut dag = Dag::new();
        two(&mut dag, |_, i| [width0, width1][i]);
        admit(dag.plan().device_table(0), room, "gpu0").is_ok()
    };
    assert!(!fits(8, 13));
    assert!(!fits(9, 12));
}
