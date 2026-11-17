//! Adaptive widths: a part of a plan whose size the planner picks, bisected
//! once at planning against the room the schedule leaves at its position
//! (`docs/memory.md`, "Adaptive widths: one planner routine").

use std::fmt;

use contracts::debug_requires;
use fb_core_arena::StagingKind;
use fb_core_error_handling::Result;

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

    /// The pinned bytes the part holds at `width`, all live at its
    /// position: a staging window that grows with the width, a counted
    /// loop's region with every copy, as [`bytes`](Adaptive::bytes)
    /// counts the device's. Monotone non-decreasing in `width`, held as
    /// `bytes` is; 0 by default, a part with no pinned rows.
    fn pinned_bytes(&self, width: usize) -> usize {
        let _ = width;
        0
    }

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

/// The bytes an adaptive part may take on its GPU worker, of each kind its
/// width grows (`docs/memory.md`, "Adaptive widths: one planner routine"):
/// what [`Dag::plan_within`] is handed per worker, and what is left of it
/// at a part's position, which [`widest`] bisects in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Room {
    /// Device bytes, against [`Adaptive::bytes`].
    pub device: usize,
    /// Pinned bytes, against [`Adaptive::pinned_bytes`].
    pub pinned: usize,
}

/// The width the planner gives `part` in `room`: the largest from 1 to
/// [`Adaptive::width_max`] whose [`bytes`](Adaptive::bytes) fit the device
/// room and whose [`pinned_bytes`](Adaptive::pinned_bytes) fit the pinned
/// room, found by bisection, then rounded down to a power of two if the part asks; 1 when
/// nothing fits, which admission then refuses. The one routine every
/// adaptive part is sized by; [`Dag::plan_within`] calls it, and a part's
/// fuzz checks its bytes against it.
///
/// ```
/// use fb_core_planner::{Adaptive, Room, widest};
///
/// /// A batch of `width` rows of 512 bytes on the device, staged through
/// /// a pinned window of two of them, up to 64 rows.
/// struct Batch;
///
/// impl Adaptive for Batch {
///     fn width_max(&self) -> usize {
///         64
///     }
///     fn bytes(&self, width: usize) -> usize {
///         512 * width
///     }
///     fn pinned_bytes(&self, width: usize) -> usize {
///         2 * 512 * width
///     }
/// }
///
/// let room = |device, pinned| Room { device, pinned };
/// assert_eq!(widest(&Batch, room(512 * 20 + 511, 1 << 20)), 20);
/// assert_eq!(widest(&Batch, room(1 << 20, 2 * 512 * 12)), 12);
/// assert_eq!(widest(&Batch, room(1 << 20, 1 << 20)), 64);
/// assert_eq!(widest(&Batch, room(0, 1 << 20)), 1);
/// ```
#[debug_requires(part.width_max() > 0, "an adaptive part has a width")]
pub fn widest(part: &impl Adaptive, room: Room) -> usize {
    let max = part.width_max();
    let fits = |width| part.bytes(width) <= room.device && part.pinned_bytes(width) <= room.pinned;
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
    /// [`Adaptive::pinned_bytes`] at the width.
    pub(super) pinned: usize,
    /// The room it was bisected in; `None` when the plan had none.
    room: Option<Room>,
}

impl fmt::Display for Part {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "width {}: {} of {}, worker {} at node {}, {} bytes and {} pinned",
            self.name,
            self.width,
            self.width_max,
            self.worker,
            self.position,
            self.bytes,
            self.pinned
        )?;
        match self.room {
            Some(room) => write!(
                f,
                " in a room of {} and {} pinned",
                room.device, room.pinned
            ),
            None => write!(f, ", no room given"),
        }
    }
}

/// What [`Dag::plan_within`] carries into a declaration: the widths
/// bisected so far with their rooms, in plan order, and the room left at
/// the next part's position, if it was measured.
pub(super) struct Within {
    widths: Vec<(usize, Option<Room>)>,
    next: Option<Room>,
}

impl<'a, H: StagingKind, S> Dag<'a, H, S> {
    /// The width of the adaptive part `part`, named `name`, whose tasks
    /// hold their rows on GPU worker `worker` and come next: its position
    /// is the next task's. Build the part's tasks with it, and a counted
    /// loop over its batches with its trip count, `total.div_ceil(width)`.
    ///
    /// Declared through [`Dag::plan_within`], it is [`widest`] in the room
    /// the schedule leaves there, per kind: the worker's room less the
    /// bytes its table of that kind holds live at the position with the
    /// part at width 1 ([`Table::live_at`](fb_core_arena::Table::live_at),
    /// a loop's region whole), less the part's own at width 1
    /// ([`bytes(1)`](Adaptive::bytes) on the device,
    /// [`pinned_bytes(1)`](Adaptive::pinned_bytes) pinned). Planned by
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
                } else if let Some(room) = within.next.take() {
                    let width = widest(part, room);
                    within.widths.push((width, Some(room)));
                    (width, Some(room))
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
            pinned: part.pinned_bytes(width),
            room,
        });
        width
    }

    /// Plan the DAG `build` declares, its adaptive parts' widths chosen in
    /// each GPU worker's [`Room`], the bytes its device and pinned blocks
    /// may take, which `rooms(worker)` gives (`docs/memory.md`, "Adaptive
    /// widths: one planner routine"). Returns the plan, and what the last
    /// declaration returned: the rows a caller reads.
    ///
    /// One pass, in schedule order: the fixed layouts first, then each
    /// part takes the remainder at its position, the parts before it at
    /// their widths and those after it at 1. So `build` is called once per
    /// part ([`Dag::width`]) and once more, each time on a new DAG, and
    /// must declare the same DAG for the same widths; each width is
    /// bisected once. `rooms` is asked once for each worker that holds a
    /// part, when the first part on it is measured, after the first
    /// declaration: a DAG with no part asks for none, and `build`'s
    /// refusals come before any room is read. The last declaration is
    /// planned as [`Dag::plan`] plans, which logs every width.
    ///
    /// # Errors
    /// The first error `build` or `rooms` returns, as soon as it does.
    ///
    /// ```
    /// use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId, admit};
    /// use fb_core_error_handling::Status;
    /// use fb_core_planner::{Adaptive, Dag, GpuTask, Room};
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
    /// let room = Room { device: 512 * 20, pinned: 0 };
    /// let (planned, width) = Dag::plan_within(|_| Ok(room), |dag| {
    ///     let width = dag.width("batch", 0, &Batch);
    ///     dag.add_gpu("step", 0, Step(width));
    ///     Ok(width)
    /// })?;
    /// assert_eq!(width, 20);
    /// assert!(admit(planned.device_table(0), room.device, "gpu0").is_ok());
    /// # Ok::<(), fb_core_error_handling::Error>(())
    /// ```
    pub fn plan_within<R>(
        mut rooms: impl FnMut(usize) -> Result<Room>,
        mut build: impl FnMut(&mut Self) -> Result<R>,
    ) -> Result<(Plan<'a, H, S>, R)> {
        let (mut widths, mut next) = (Vec::new(), None);
        // Each worker's room, asked once.
        let mut asked: Vec<Option<Room>> = Vec::new();
        loop {
            let mut dag = Dag::new();
            dag.within = Some(Within { widths, next });
            let made = build(&mut dag)?;
            let chosen = dag.within.take().map_or(Vec::new(), |w| w.widths);
            let Some(part) = dag.parts.get(chosen.len()) else {
                return Ok((dag.plan(), made));
            };
            let (worker, position) = (part.worker, part.position);
            let (device_least, pinned_least) = (part.bytes, part.pinned);
            if asked.len() <= worker {
                asked.resize(worker + 1, None);
            }
            let room = match asked[worker] {
                Some(room) => room,
                None => *asked[worker].insert(rooms(worker)?),
            };
            let planned = dag.lay_out();
            let device = planned.device_table(worker).live_at(position);
            let pinned = planned.host_table(worker).live_at(position);
            let left = Room {
                device: room
                    .device
                    .saturating_sub(device.saturating_sub(device_least)),
                pinned: room
                    .pinned
                    .saturating_sub(pinned.saturating_sub(pinned_least)),
            };
            (widths, next) = (chosen, Some(left));
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
