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
    /// Pinned copies of each column, a staging window.
    pinned: usize,
    max: usize,
    floor: usize,
    pow2: bool,
}

impl Columns {
    fn new(column: usize, max: usize) -> Self {
        Columns {
            column,
            copies: 1,
            pinned: 0,
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

    fn pinned_bytes(&self, width: usize) -> usize {
        self.pinned * self.column * width
    }

    fn warn_below(&self) -> usize {
        self.floor
    }

    fn power_of_two(&self) -> bool {
        self.pow2
    }
}

/// A part whose device bytes at width `w` are `table[w - 1]`, and its
/// pinned bytes `pinned[w - 1]`, or none if `pinned` is empty.
struct Tabled {
    table: Vec<usize>,
    pinned: Vec<usize>,
    pow2: bool,
}

impl Adaptive for Tabled {
    fn width_max(&self) -> usize {
        self.table.len()
    }

    fn bytes(&self, width: usize) -> usize {
        self.table[width - 1]
    }

    fn pinned_bytes(&self, width: usize) -> usize {
        self.pinned.get(width - 1).copied().unwrap_or(0)
    }

    fn power_of_two(&self) -> bool {
        self.pow2
    }
}

/// The width a linear scan finds: the largest whose bytes of both kinds
/// fit, 1 if none, rounded as [`widest`] rounds.
fn scan(part: &impl Adaptive, room: Room) -> usize {
    let width = (1..=part.width_max())
        .filter(|&w| part.bytes(w) <= room.device && part.pinned_bytes(w) <= room.pinned)
        .max()
        .unwrap_or(1);
    if part.power_of_two() {
        1 << width.ilog2()
    } else {
        width
    }
}

proptest! {
    /// Random monotone bytes of both kinds, from random steps: the
    /// bisected width is the linear scan's, in rooms below, among and
    /// above the bytes, with either kind the one that binds.
    #[test]
    fn bisection_equals_a_linear_scan(
        base in (0usize..1 << 20, 0usize..1 << 20),
        steps in prop::collection::vec((0usize..1 << 12, 0usize..1 << 12), 1..200),
        room in (0usize..1 << 21, 0usize..1 << 21),
        pow2 in any::<bool>(),
    ) {
        let sums = |base: usize, step: fn(&(usize, usize)) -> usize| -> Vec<usize> {
            steps
                .iter()
                .scan(base, |at, s| {
                    *at += step(s);
                    Some(*at)
                })
                .collect()
        };
        let part = Tabled { table: sums(base.0, |s| s.0), pinned: sums(base.1, |s| s.1), pow2 };
        let room = Room { device: room.0, pinned: room.1 };
        prop_assert_eq!(widest(&part, room), scan(&part, room));
        for (&device, &pinned) in part.table.iter().zip(&part.pinned) {
            for room in [Room { device, ..room }, Room { pinned, ..room }] {
                prop_assert_eq!(widest(&part, room), scan(&part, room));
            }
        }
    }
}

/// A non-monotone `bytes` is caught: for some room the bisection and the
/// scan disagree, which is what a part's fuzz fails on.
#[test]
fn a_non_monotone_bytes_is_caught() {
    let part = Tabled {
        table: vec![10, 50, 20, 60, 70, 80, 90, 100],
        pinned: Vec::new(),
        pow2: false,
    };
    let room = Room {
        device: 30,
        pinned: 0,
    };
    assert_eq!(scan(&part, room), 3);
    assert_eq!(widest(&part, room), 1);
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

/// A room of `bytes` on the device, and no pinned bytes, for every worker.
fn device(bytes: usize) -> impl FnMut(usize) -> fb_core_error_handling::Result<Room> {
    move |_| {
        Ok(Room {
            device: bytes,
            pinned: 0,
        })
    }
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
    let (planned, width) = Dag::plan_within(device(room), |dag| {
        Ok(between(dag, |dag| dag.width("between", 0, &part)))
    })
    .unwrap();
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
    let (planned, width) = Dag::plan_within(device(room), |dag| {
        Ok(batched(dag, |dag| dag.width("batched", 0, &part)))
    })
    .unwrap();
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

/// A counted loop's part whose bytes bound its region from above, more
/// loosely at small widths (a floor of ten columns' ring): measured
/// against its loop's region as laid out, not its bytes at 1, its width
/// is the largest admission accepts, where the bytes at 1 would have
/// handed it the floor's slack as room.
#[test]
fn a_loose_bound_is_measured_against_the_region_laid_out() {
    keep_logs();
    struct Floored(Columns);
    impl Adaptive for Floored {
        fn width_max(&self) -> usize {
            self.0.width_max()
        }
        fn bytes(&self, width: usize) -> usize {
            self.0.bytes(width).max(self.0.bytes(10))
        }
    }
    let part = Floored(Columns {
        copies: 2,
        ..Columns::new(8 * COLUMN, 64)
    });
    let room = 8 * N + 2 * 20 * 8 * COLUMN;
    let (planned, width) = Dag::plan_within(device(room), |dag| {
        Ok(batched(dag, |dag| dag.width("loose", 0, &part)))
    })
    .unwrap();
    assert_eq!(width, 20);
    assert!(admit(planned.device_table(0), room, "gpu0").is_ok());
    let mut wider = Dag::new();
    batched(&mut wider, |_| 21);
    assert!(admit(wider.plan().device_table(0), room, "gpu0").is_err());
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
    let (planned, widths) = Dag::plan_within(device(room), |dag| {
        Ok(two(dag, |dag, i| match i {
            0 => dag.width("first", 0, &first),
            _ => dag.width("second", 0, &second),
        }))
    })
    .unwrap();
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

/// One adaptive task staging `width` columns through a pinned window of
/// two copies, beside a fixed pinned worker row of `N` doubles, and holding
/// them on the device. Returns the width.
fn staged(dag: &mut Test, width: impl FnOnce(&mut Test) -> usize) -> usize {
    use Decl::*;
    let _ = dag.host_worker_row::<f64>(0, "fixed", N);
    let width = width(dag);
    dag.add_gpu(
        "part",
        0,
        Declares(vec![
            Staged("window", 2 * COLUMN * width),
            Scratch("batch", COLUMN * width),
            Out("Y", N),
        ]),
    );
    width
}

/// The pinned room binds a part whose device room does not: the width is
/// the largest whose window the pinned cap admits, and the next is
/// refused by it, the device block admitted either way.
#[test]
fn a_pinned_window_takes_the_pinned_room() {
    let part = Columns {
        pinned: 2,
        ..Columns::new(8 * COLUMN, 64)
    };
    let room = Room {
        device: 1 << 20,
        pinned: 8 * N + 2 * 12 * 8 * COLUMN + 100,
    };
    let (planned, width) = Dag::plan_within(
        |_| Ok(room),
        |dag| Ok(staged(dag, |dag| dag.width("staged", 0, &part))),
    )
    .unwrap();
    assert_eq!(width, 12);
    let line = format!(
        "width staged: 12 of 64, worker 0 at node 0, {} bytes and {} pinned in a room of ",
        12 * 8 * COLUMN,
        2 * 12 * 8 * COLUMN,
    );
    let text = planned.to_string();
    assert!(text.contains(&line), "{text}");
    assert!(
        text.contains(&format!(" and {} pinned\n", room.pinned - 8 * N)),
        "{text}"
    );
    let fits = |width| {
        let mut dag = Dag::new();
        staged(&mut dag, |_| width);
        let planned: Plan<'_, PinnedStandIn, ()> = dag.plan();
        assert!(admit(planned.device_table(0), room.device, "gpu0").is_ok());
        admit(planned.host_table(0), room.pinned, "gpu0").is_ok()
    };
    assert!(fits(width));
    assert!(!fits(width + 1));
}

/// A room is asked once for each worker holding a part, after the first
/// declaration; a DAG with no part asks for none, and an error from
/// either closure is returned at once.
#[test]
fn rooms_are_asked_once_and_only_when_a_part_is_measured() {
    let part = Columns::new(8 * COLUMN, 64);
    let asked = std::cell::Cell::new(0);
    let room = |_| {
        asked.set(asked.get() + 1);
        Ok(Room {
            device: 1 << 20,
            pinned: 0,
        })
    };
    let mut runs = 0;
    Dag::plan_within(room, |dag: &mut Test| {
        runs += 1;
        Ok(two(dag, |dag, i| dag.width(["p0", "p1"][i], 0, &part)))
    })
    .unwrap();
    assert_eq!((runs, asked.get()), (3, 1));
    asked.set(0);
    Dag::plan_within(room, |dag: &mut Test| Ok(between(dag, |_| 4))).unwrap();
    assert_eq!(asked.get(), 0);
    let refused = Dag::plan_within(room, |_: &mut Test| {
        Err::<(), _>(fb_core_error_handling::Error::config("refused".to_owned()))
    });
    assert!(refused.is_err());
    assert_eq!(asked.get(), 0);
    let unread = Dag::plan_within(
        |_| Err(fb_core_error_handling::Error::config("no card".to_owned())),
        |dag: &mut Test| Ok(between(dag, |dag| dag.width("part", 0, &part))),
    );
    assert!(unread.is_err_and(|e| e.message.contains("no card")));
}
