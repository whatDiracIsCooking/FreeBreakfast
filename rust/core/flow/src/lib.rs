//! `core::flow`: the v0.0.4 layer's steps, the slots that hold their
//! `Output`s, the View a step's `run` reads through, and the one step DAG
//! built twice on rustflow, once calling each step's `mem_budget` and once
//! its `run` (`docs/programming_model.md`, "Steps on rustflow").
//!
//! Ported from the v0.0.4 prototype's `src/flow.rs` and `flow_tuples.rs`
//! (`../experimental/rust/v0.0.4/README.md`, "Steps and the MainBus", "The
//! step DAG on rustflow"), with the prototype's acquire and release, which
//! sat on its `MainBus`, moved here as the one step wrapper.
//!
//! - [`Step`]: one config table, one slot, `mem_budget` and `run`;
//!   [`Slot`], its `Output` behind a lock used plainly, and its [`Lane`],
//!   the streams the wrapper made for it.
//! - [`View`]: what `run` reads, the shared setup and the `Output`s of the
//!   steps its [`Step::Reads`] names, and nothing else.
//! - [`Bus`]: the run's shared object, its ledger and its [`Locks`], each
//!   GPU's `GpuResources` and each manager's lock, which a step reaches
//!   only as the guards in its [`Held`].
//! - [`Dag`], [`Plan`]: the steps a config asks for, checked to close
//!   and built over rustflow's [`Executor`], [`Plan::budget`] then
//!   [`Plan::execute`], which a replay calls again.
//! - [`fan_out`]: an all-GPU step's inner work, one scoped thread per GPU.
//!
//! **The declaration is the edge.** The builder makes the DAG's edges from
//! each step's `Reads` and from nothing else, and [`View::read`] compiles
//! only for a step in it, so a read with no edge behind it cannot be
//! written.
//!
//! **The wrapper.** Every `run` is entered and left by one function a step
//! cannot skip: make the step's device current, take the locks its
//! `mem_budget` declared in rank order ([`fb_core_budget::LockId`]), bind
//! a GPU's handle workspace the first time, fill the lease in the ledger's
//! NaN mode, call `run` with the held guards, synchronize every stream the
//! step has and each held GPU's stream, and only then release the locks.
//! So a finished node is finished device work, and the DAG's edges order
//! everything between steps.
//!
//! ```
//! use std::sync::Arc;
//! use fb_core_budget::{Backing, Budget, Cap, Caps, Lease, Ledger};
//! use fb_core_error_handling::{Result, Status};
//! use fb_core_flow::{Bus, Dag, Executor, Held, Lane, Locks, Slot, Step, View};
//!
//! struct Run { ledger: Ledger, locks: Locks, n: Slot<usize>, twice: Slot<usize> }
//! impl Bus for Run {
//!     type Shared = ();
//!     fn shared(&self) -> &() { &() }
//!     fn ledger(&self) -> &Ledger { &self.ledger }
//!     fn locks(&self) -> &Locks { &self.locks }
//! }
//!
//! #[derive(Clone, Copy)]
//! struct N;
//! impl Step<Run> for N {
//!     const NAME: &'static str = "n";
//!     type Output = usize;
//!     type Reads = ();
//!     fn slot(bus: &Run) -> &Slot<usize> { &bus.n }
//!     fn mem_budget(_: &(), _: &Budget<'_>) -> Result<usize> { Ok(0) }
//!     fn run(n: &mut usize, _: &View<'_, Run, Self>, _: &mut Held<'_>, _: &Lane, _: &Lease<'_>) -> Status {
//!         *n = 21;
//!         Ok(())
//!     }
//! }
//!
//! #[derive(Clone, Copy)]
//! struct Twice;
//! impl Step<Run> for Twice {
//!     const NAME: &'static str = "twice";
//!     type Output = usize;
//!     type Reads = (N,);
//!     fn slot(bus: &Run) -> &Slot<usize> { &bus.twice }
//!     fn mem_budget(_: &(), _: &Budget<'_>) -> Result<usize> { Ok(0) }
//!     fn run(out: &mut usize, view: &View<'_, Run, Self>, _: &mut Held<'_>, _: &Lane, _: &Lease<'_>) -> Status {
//!         *out = 2 * *view.read(N);
//!         Ok(())
//!     }
//! }
//!
//! let cap = Cap { written: None, used: 1 << 30 };
//! let caps = Caps { device: vec![cap], host: cap };
//! let bus = Arc::new(Run {
//!     ledger: Ledger::new(caps, Backing::Host, vec![0], false),
//!     locks: Locks::new(&[]),
//!     n: Slot::new(),
//!     twice: Slot::new(),
//! });
//! let mut dag = Dag::new();
//! dag.add(Twice).add(N);
//! let plan = dag.build(bus).unwrap();
//! let executor = Executor::new(2);
//! plan.budget(&executor).unwrap();
//! plan.execute(&executor).unwrap();
//! assert_eq!(*plan.output(Twice), 42);
//! ```

#![forbid(unsafe_code)]

mod dag;
mod fan_out;
#[cfg(test)]
mod tests;
mod wrapper;

use std::marker::PhantomData;
use std::ops::Deref;
use std::sync::{Mutex, MutexGuard, PoisonError, RwLock, RwLockReadGuard};

use contracts::debug_requires;
use fb_core_budget::{Budget, Ledger, ManagerId};
use fb_core_device::GpuResources;
use fb_core_error_handling::{Result, Status};
use fb_core_log::lt;

pub use dag::{Dag, Plan, StepTime};
pub use fan_out::fan_out;
/// rustflow's executor, the one both modes run on: one per process, its
/// thread count fixed at init from the config. Re-exported so that no
/// member but this one names rustflow (`docs/programming_model.md`, "CPU
/// workers").
pub use rustflow::Executor;
pub use wrapper::{GpuLane, Held, Lane};

/// `m`'s guard, poisoned or not: a step that panicked fails the run
/// (`RunError::Panicked`), so what it left behind is never read as a
/// result.
fn locked<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Where a step runs, fixed before budget mode: stealing moves which thread
/// runs a step, never this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// On no GPU.
    Host,
    /// On gpu0, the privileged GPU.
    Gpu0,
    /// On every GPU of the run, its result reduced onto gpu0.
    AllGpus,
}

/// The managers there are, for [`Locks`]: one lock each.
const MANAGERS: usize = 1;

/// Manager `id`'s lock in [`Locks`]. A match, so a new `ManagerId` does
/// not build until it has a lock.
const fn manager_index(id: ManagerId) -> usize {
    match id {
        ManagerId::Eri => 0,
    }
}

/// The run's locks: each GPU's `GpuResources` inside its `Mutex`, and each
/// manager's lock. The wrapper alone takes them, in rank order, before a
/// `run`; a step reaches a GPU's handles only through the guards in its
/// [`Held`], and `run` can reach no `Mutex` here, even when the bus puts
/// the locks where its View can see them:
///
/// ```compile_fail,E0616
/// fn run_reaching_for_a_lock(locks: &fb_core_flow::Locks) {
///     let _gpu0 = locks.gpus[0].lock();
/// }
/// ```
///
/// The one method that locks, [`Locks::query`], takes a [`Budget`], which
/// only `mem_budget` is given.
pub struct Locks {
    physical: Vec<i32>,
    gpus: Vec<Mutex<GpuResources>>,
    managers: [Mutex<()>; MANAGERS],
}

impl Locks {
    /// The locks of a run over GPUs on the devices `physical`, GPU `i` on
    /// `physical[i]`, its `GpuResources` made there, which aborts at the
    /// caller's line if the runtime refuses, as every create does. A run of
    /// no GPU, a device-free test's, passes none.
    #[track_caller]
    #[must_use]
    pub fn new(physical: &[i32]) -> Locks {
        Locks {
            physical: physical.to_vec(),
            gpus: physical
                .iter()
                .enumerate()
                .map(|(gpu, &p)| GpuResources::new(gpu, p))
                .collect(),
            managers: std::array::from_fn(|_| Mutex::new(())),
        }
    }

    /// The device each GPU is on.
    #[must_use]
    pub fn physical(&self) -> &[i32] {
        &self.physical
    }

    /// Call `f` with GPU `gpu`'s resources under its lock, in budget mode: a
    /// size query, which needs the live handles (`syevd_lwork`). `budget`
    /// is the proof of the mode, and `run` is never given one.
    #[debug_requires(lt!(gpu, self.gpus.len()), "gpu is a GPU of the run")]
    pub fn query<R>(
        &self,
        budget: &Budget<'_>,
        gpu: usize,
        f: impl FnOnce(&mut GpuResources) -> R,
    ) -> R {
        let _mode = budget;
        f(&mut locked(&self.gpus[gpu]))
    }
}

/// The shared object of a run: what setup filled, the ledger, and the
/// locks. A step's `run` is handed a [`View`], never the bus.
pub trait Bus: Send + Sync + Sized + 'static {
    /// What setup fills and every step may read without declaring it: the
    /// config, the dimensions.
    type Shared;

    /// What setup filled.
    fn shared(&self) -> &Self::Shared;
    /// The caps, counters and table.
    fn ledger(&self) -> &Ledger;
    /// Each GPU's resources and each manager's lock, for the wrapper.
    fn locks(&self) -> &Locks;
}

/// One step's place in the bus: its `Output` behind a lock used plainly,
/// and beside it the [`Lane`] the wrapper made for it. Empty until budget
/// mode fills it.
pub struct Slot<O> {
    output: RwLock<Option<O>>,
    lane: Mutex<Option<Lane>>,
}

impl<O> Default for Slot<O> {
    fn default() -> Self {
        Slot {
            output: RwLock::new(None),
            lane: Mutex::new(None),
        }
    }
}

impl<O> Slot<O> {
    /// An empty slot.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether budget mode has put the step's `Output` here.
    pub fn is_filled(&self) -> bool {
        self.output
            .read()
            .unwrap_or_else(PoisonError::into_inner)
            .is_some()
    }

    fn fill(&self, output: O, lane: Lane) {
        *self.output.write().unwrap_or_else(PoisonError::into_inner) = Some(output);
        *locked(&self.lane) = Some(lane);
    }

    /// # Panics
    /// If budget mode has not filled the slot.
    fn read(&self, step: &'static str) -> Read<'_, O> {
        let guard = self.output.read().unwrap_or_else(PoisonError::into_inner);
        assert!(
            guard.is_some(),
            "step '{step}' is read before it was budgeted"
        );
        Read(guard)
    }
}

/// A reader's hold on another step's `Output`.
pub struct Read<'a, O>(RwLockReadGuard<'a, Option<O>>);

impl<O> Deref for Read<'_, O> {
    type Target = O;
    fn deref(&self) -> &O {
        // INVARIANT: `Slot::read` made this guard only over a filled slot,
        // and nothing empties one.
        self.0.as_ref().expect("checked when the guard was taken")
    }
}

/// A step: one config table, one slot, one function that counts and
/// allocates and one that runs. A unit struct, named by value:
/// `view.read(EriStep)`.
pub trait Step<B: Bus>: Copy + Send + Sync + 'static {
    /// The step's name: its row of the budget table and its node.
    const NAME: &'static str;
    /// What outlives the step's `run`: what other steps read, or what
    /// carries over a replay. `'static`, so it cannot hold a lease.
    type Output: Send + Sync + 'static;
    /// The steps whose `Output`s `run` reads: a tuple of steps, `()` for
    /// none. The DAG's edges are exactly these.
    type Reads: ReadSet<B>;

    /// The streams of its own the step has on each GPU it spans.
    const STREAMS: usize = 1;

    /// The step's slot in the bus.
    fn slot(bus: &B) -> &Slot<Self::Output>;

    /// Where the step runs.
    fn placement(_shared: &B::Shared) -> Placement {
        Placement::Host
    }

    /// Charge what the step's `Output` holds and allocate it, declare the
    /// step's scratch and the locks its `run` takes, and return the
    /// `Output`; the wrapper puts it in the slot. Budget mode calls it once.
    /// It reads the dimensions, the step's options and the caps, never
    /// another step's `Output`.
    ///
    /// # Errors
    /// A size it cannot ask for: a limit too small for one batch.
    fn mem_budget(shared: &B::Shared, budget: &Budget<'_>) -> Result<Self::Output>;

    /// The step's work, into the buffers `mem_budget` made and the scratch
    /// it declared, on its `lane`'s streams and its held GPUs' streams. It
    /// has no `Budget`, so it allocates nothing budgeted; `scratch` is a
    /// borrow, so it keeps none of it; and it has no bus, so it takes no
    /// lock.
    ///
    /// # Errors
    /// The step's failure: the steps that read it are then skipped.
    fn run(
        out: &mut Self::Output,
        view: &View<'_, B, Self>,
        held: &mut Held<'_>,
        lane: &Lane,
        scratch: &fb_core_budget::Lease<'_>,
    ) -> Status;
}

/// A tuple of steps: what a step declares it reads.
pub trait ReadSet<B: Bus> {
    /// The names of the steps in the tuple.
    fn names() -> Vec<&'static str>;
}

impl<B: Bus> ReadSet<B> for () {
    fn names() -> Vec<&'static str> {
        Vec::new()
    }
}

/// `S` is in this tuple, at position `I`: the bound [`View::read`] puts on
/// a step's `Reads`. `I` is inferred and never written.
#[diagnostic::on_unimplemented(
    message = "this step reads a step its `Reads` does not name",
    label = "not in the reading step's `Reads`",
    note = "the DAG's edges are made from `Reads`: add this step to that tuple"
)]
pub trait Has<S, I> {}

/// A position in a tuple, for [`Has`].
pub struct At<const N: usize>;

/// `Has` at each position, and `ReadSet`, for one tuple of steps.
macro_rules! tuple {
    ($($t:ident $i:tt),+) => {
        tuple!(@has [$($t),+] $($t $i),+);
        impl<B: Bus, $($t: Step<B>),+> ReadSet<B> for ($($t,)+) {
            fn names() -> Vec<&'static str> {
                vec![$($t::NAME),+]
            }
        }
    };
    (@has $all:tt $($t:ident $i:tt),+) => {
        $(tuple!(@one $all $t $i);)+
    };
    (@one [$($a:ident),+] $t:ident $i:tt) => {
        impl<$($a),+> Has<$t, At<$i>> for ($($a,)+) {}
    };
}

tuple!(T0 0);
tuple!(T0 0, T1 1);
tuple!(T0 0, T1 1, T2 2);
tuple!(T0 0, T1 1, T2 2, T3 3);
tuple!(T0 0, T1 1, T2 2, T3 3, T4 4);
tuple!(T0 0, T1 1, T2 2, T3 3, T4 4, T5 5);
tuple!(T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6);
tuple!(T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7);

/// What a step's `run` is handed: what setup filled, and the `Output`s of
/// the steps it declared. Not the bus.
///
/// A read of a step its `Reads` does not name does not compile:
///
/// ```compile_fail,E0277
/// # use fb_core_budget::{Budget, Lease, Ledger};
/// # use fb_core_error_handling::{Result, Status};
/// # use fb_core_flow::{Bus, Held, Lane, Locks, Slot, Step, View};
/// # struct Run { ledger: Ledger, locks: Locks, a: Slot<f64>, b: Slot<f64> }
/// # impl Bus for Run {
/// #     type Shared = ();
/// #     fn shared(&self) -> &() { &() }
/// #     fn ledger(&self) -> &Ledger { &self.ledger }
/// #     fn locks(&self) -> &Locks { &self.locks }
/// # }
/// # #[derive(Clone, Copy)]
/// # struct A;
/// # impl Step<Run> for A {
/// #     const NAME: &'static str = "a";
/// #     type Output = f64;
/// #     type Reads = ();
/// #     fn slot(bus: &Run) -> &Slot<f64> { &bus.a }
/// #     fn mem_budget(_: &(), _: &Budget<'_>) -> Result<f64> { Ok(0.0) }
/// #     fn run(_: &mut f64, _: &View<'_, Run, Self>, _: &mut Held<'_>, _: &Lane, _: &Lease<'_>) -> Status { Ok(()) }
/// # }
/// #[derive(Clone, Copy)]
/// struct B;
/// impl Step<Run> for B {
///     const NAME: &'static str = "b";
///     type Output = f64;
///     type Reads = (); // A is not named
///     fn slot(bus: &Run) -> &Slot<f64> { &bus.b }
///     fn mem_budget(_: &(), _: &Budget<'_>) -> Result<f64> { Ok(0.0) }
///     fn run(out: &mut f64, view: &View<'_, Run, Self>, _: &mut Held<'_>, _: &Lane, _: &Lease<'_>) -> Status {
///         *out = *view.read(A);
///         Ok(())
///     }
/// }
/// ```
///
/// Nor does a `run` that reaches through its View for the bus, and so for
/// a GPU's `Mutex`:
///
/// ```compile_fail,E0616
/// # use fb_core_budget::{Budget, Lease, Ledger};
/// # use fb_core_error_handling::{Result, Status};
/// # use fb_core_flow::{Bus, Held, Lane, Locks, Slot, Step, View};
/// # struct Run { ledger: Ledger, locks: Locks, b: Slot<f64> }
/// # impl Bus for Run {
/// #     type Shared = ();
/// #     fn shared(&self) -> &() { &() }
/// #     fn ledger(&self) -> &Ledger { &self.ledger }
/// #     fn locks(&self) -> &Locks { &self.locks }
/// # }
/// #[derive(Clone, Copy)]
/// struct B;
/// impl Step<Run> for B {
///     const NAME: &'static str = "b";
///     type Output = f64;
///     type Reads = ();
///     fn slot(bus: &Run) -> &Slot<f64> { &bus.b }
///     fn mem_budget(_: &(), _: &Budget<'_>) -> Result<f64> { Ok(0.0) }
///     fn run(_: &mut f64, view: &View<'_, Run, Self>, _: &mut Held<'_>, _: &Lane, _: &Lease<'_>) -> Status {
///         let _gpu0 = view.bus.locks().gpus[0].lock();
///         Ok(())
///     }
/// }
/// ```
pub struct View<'a, B: Bus, S: Step<B>> {
    bus: &'a B,
    step: PhantomData<S>,
}

impl<'a, B: Bus, S: Step<B>> View<'a, B, S> {
    /// What setup filled.
    pub fn shared(&self) -> &'a B::Shared {
        self.bus.shared()
    }

    /// The `Output` of `step`, which this step's `Reads` names.
    ///
    /// # Panics
    /// If `step`'s slot is empty, which the builder's refusal of a read
    /// the config did not ask for, and the edge, rule out.
    pub fn read<T: Step<B>, I>(&self, _step: T) -> Read<'a, T::Output>
    where
        S::Reads: Has<T, I>,
    {
        T::slot(self.bus).read(T::NAME)
    }
}
