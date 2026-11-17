//! Adaptive widths: a part of a plan whose size the planner picks, bisected
//! once at planning against the room the schedule leaves at its position
//! (`docs/memory.md`, "Adaptive widths: one planner routine").

use std::fmt;

use contracts::debug_requires;
use fb_core_arena::StagingKind;

use super::dag::{Dag, Plan};

/// An adaptive part's declaration: what it holds at each width, and its
/// natural full width (`docs/memory.md`, "Adaptive widths: one planner
/// routine"). Implemented by the dimensions of the tasks the part is,
/// beside their layouts, such as a streamed tensor's batch over its aux
/// functions, and handed to [`Dag::width`] before those tasks are added.
///
/// Its width is planned, never read from free memory at run time:
/// [`widest`] bisects [`Adaptive::bytes`] against the room once, the
/// width is then a dimension of the plan, and the part's tasks are built
/// with it like any other.
pub trait Adaptive {
    /// The natural full width, its dimensions' own (a batch of every aux
    /// function at once): the most the planner gives it. At least 1.
    fn width_max(&self) -> usize;

    /// The device bytes the part holds at `width`, all live at its
    /// position: its rows padded to `fb_core_arena::DEVICE_ALIGNMENT`, a
    /// counted loop's region with every copy. Monotone non-decreasing in
    /// `width`, which the part's own tier-3 fuzz holds: [`widest`] against
    /// a linear scan over random dimensions.
    fn bytes(&self, width: usize) -> usize;

    /// The width below which a narrowed part is logged at `warn`: 2 by
    /// default, so a width of 1 is; more for a part that is slow below a
    /// size (the streamed J/K batch below 32). A part at its
    /// [`width_max`](Adaptive::width_max) is not narrowed, whatever it is.
    fn warn_below(&self) -> usize {
        2
    }

    /// Whether the width is rounded down to a power of two once bisected,
    /// as a batch a kernel's tiles divide is; no by default. The bytes are
    /// monotone, so the rounded width still fits.
    fn power_of_two(&self) -> bool {
        false
    }
}

/// The width the planner gives `part` in `room` bytes: the largest from 1
/// to [`Adaptive::width_max`] whose [`bytes`](Adaptive::bytes) fit, found by
/// bisection, then rounded down to a power of two if the part asks; 1 when
/// nothing fits, which admission then refuses. The one routine every
/// adaptive part is sized by; [`Dag::plan_within`] calls it, and a part's
/// fuzz checks its bytes against it.
///
/// ```
/// use fb_core_planner::{Adaptive, widest};
///
/// /// A batch of `width` rows of 512 bytes, up to 64 of them.
/// struct Batch;
///
/// impl Adaptive for Batch {
///     fn width_max(&self) -> usize {
///         64
///     }
///     fn bytes(&self, width: usize) -> usize {
///         512 * width
///     }
/// }
///
/// assert_eq!(widest(&Batch, 512 * 20 + 511), 20);
/// assert_eq!(widest(&Batch, 1 << 20), 64);
/// assert_eq!(widest(&Batch, 0), 1);
/// ```
#[debug_requires(part.width_max() > 0, "an adaptive part has a width")]
pub fn widest(part: &impl Adaptive, room: usize) -> usize {
    let max = part.width_max();
    let fits = |width| part.bytes(width) <= room;
    let width = if fits(max) {
        max
    } else {
        // `low` fits or is 1, `high` does not.
        let (mut low, mut high) = (1, max);
        while high - low > 1 {
            let mid = low + (high - low) / 2;
            if fits(mid) {
                low = mid;
            } else {
                high = mid;
            }
        }
        low
    };
    if part.power_of_two() {
        1 << width.ilog2()
    } else {
        width
    }
}

/// An adaptive part as planned: its name, its worker and position, its
/// width and what it was chosen from, for the plan's text and its log.
pub(super) struct Part {
    name: String,
    pub(super) worker: usize,
    pub(super) position: usize,
    width: usize,
    width_max: usize,
    warn_below: usize,
    /// [`Adaptive::bytes`] at the width.
    pub(super) bytes: usize,
    /// The room it was bisected in; `None` when the plan had none.
    room: Option<usize>,
}

impl fmt::Display for Part {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "width {}: {} of {}, worker {} at node {}, {} bytes",
            self.name, self.width, self.width_max, self.worker, self.position, self.bytes
        )?;
        match self.room {
            Some(room) => write!(f, " in a room of {room}"),
            None => write!(f, ", no room given"),
        }
    }
}

/// What [`Dag::plan_within`] carries into a declaration: each worker's
/// room, the widths bisected so far with their rooms, in plan order, and
/// the fixed bytes measured at the next part's position, if one was.
pub(super) struct Within {
    rooms: Vec<usize>,
    widths: Vec<(usize, Option<usize>)>,
    fixed: Option<usize>,
}

impl<'a, H: StagingKind, S> Dag<'a, H, S> {
    /// The width of the adaptive part `part`, named `name`, whose tasks
    /// hold their rows on GPU worker `worker` and come next: its position
    /// is the next task's. Build the part's tasks with it, and a counted
    /// loop over its batches with its trip count, `total.div_ceil(width)`.
    ///
    /// Declared through [`Dag::plan_within`], it is [`widest`] in the room
    /// the schedule leaves there: the worker's room less the bytes the
    /// device table holds live at the position with the part at width 1
    /// ([`Table::live_at`](fb_core_arena::Table::live_at), a loop's region
    /// whole), less [`bytes(1)`](Adaptive::bytes). Planned by
    /// [`Dag::plan`] alone, with no room, it is its
    /// [`width_max`](Adaptive::width_max).
    pub fn width(&mut self, name: &str, worker: usize, part: &impl Adaptive) -> usize {
        self.worker(worker);
        let index = self.parts.len();
        let (width, room) = match &mut self.within {
            None => (part.width_max(), None),
            Some(within) => {
                if let Some(&chosen) = within.widths.get(index) {
                    chosen
                } else if let Some(fixed) = within.fixed.take() {
                    let room = within.rooms.get(worker).map(|r| r.saturating_sub(fixed));
                    let width = room.map_or(part.width_max(), |r| widest(part, r));
                    within.widths.push((width, room));
                    (width, room)
                } else {
                    // Not yet measured: at its least, as the parts after
                    // the one being measured are.
                    (1, None)
                }
            }
        };
        self.parts.push(Part {
            name: name.to_owned(),
            worker,
            position: self.len(),
            width,
            width_max: part.width_max(),
            warn_below: part.warn_below(),
            bytes: part.bytes(width),
            room,
        });
        width
    }

    /// Plan the DAG `build` declares, its adaptive parts' widths chosen in
    /// `rooms`, the device bytes each GPU worker's block may take, by
    /// index (`docs/memory.md`, "Adaptive widths: one planner routine");
    /// a worker past the slice has no room, and its parts their
    /// [`width_max`](Adaptive::width_max). Returns the plan, and what the
    /// last declaration returned: the rows a caller reads.
    ///
    /// One pass, in schedule order: the fixed layouts first, then each
    /// part takes the remainder at its position, the parts before it at
    /// their widths and those after it at 1. So `build` is called once per
    /// part ([`Dag::width`]) and once more, each time on a new DAG, and
    /// must declare the same DAG for the same widths; each width is
    /// bisected once. The last declaration is planned as [`Dag::plan`]
    /// plans, which logs every width.
    ///
    /// ```
    /// use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId, admit};
    /// use fb_core_error_handling::Status;
    /// use fb_core_planner::{Adaptive, Dag, GpuTask};
    /// use fb_core_thread_pool::DeviceStream;
    ///
    /// /// Batches of `width` columns of 64 doubles each, up to 64 columns.
    /// struct Batch;
    ///
    /// impl Adaptive for Batch {
    ///     fn width_max(&self) -> usize {
    ///         64
    ///     }
    ///     fn bytes(&self, width: usize) -> usize {
    ///         512 * width
    ///     }
    /// }
    ///
    /// /// A task holding one batch as scratch.
    /// struct Step(usize);
    ///
    /// impl GpuTask for Step {
    ///     type Rows = RowId<Device, f64>;
    ///     fn layout(&self, d: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
    ///         d.scratch("batch", 64 * self.0)
    ///     }
    ///     fn run(&self, _: &Self::Rows, _: &Bufs<'_, Device>, _: &CopyOnly<'_, Pinned>, _: &DeviceStream) -> Status {
    ///         Ok(()) // enqueue on the stream
    ///     }
    /// }
    ///
    /// let room = 512 * 20;
    /// let (planned, width) = Dag::plan_within(&[room], |dag| {
    ///     let width = dag.width("batch", 0, &Batch);
    ///     dag.add_gpu("step", 0, Step(width));
    ///     width
    /// });
    /// assert_eq!(width, 20);
    /// assert!(admit(planned.device_table(0), room, "gpu0").is_ok());
    /// ```
    pub fn plan_within<R>(
        rooms: &[usize],
        mut build: impl FnMut(&mut Self) -> R,
    ) -> (Plan<'a, H, S>, R) {
        let (mut widths, mut fixed) = (Vec::new(), None);
        loop {
            let mut dag = Dag::new();
            dag.within = Some(Within {
                rooms: rooms.to_vec(),
                widths,
                fixed,
            });
            let made = build(&mut dag);
            let chosen = dag.within.take().map_or(Vec::new(), |w| w.widths);
            let Some(part) = dag.parts.get(chosen.len()) else {
                return (dag.plan(), made);
            };
            let (worker, position, least) = (part.worker, part.position, part.bytes);
            let planned = dag.lay_out();
            let live = planned.device_table(worker).live_at(position);
            (widths, fixed) = (chosen, Some(live.saturating_sub(least)));
        }
    }
}

impl<H: StagingKind, S> Plan<'_, H, S> {
    /// Log each adaptive part's width at `debug`, and at `warn` one
    /// narrowed below its floor ([`Adaptive::warn_below`]).
    pub(super) fn log_widths(&self) {
        for part in &self.parts {
            if part.width < part.warn_below && part.width < part.width_max {
                log::warn!("planner: {part}, below {}", part.warn_below);
            } else {
                log::debug!("planner: {part}");
            }
        }
    }
}
