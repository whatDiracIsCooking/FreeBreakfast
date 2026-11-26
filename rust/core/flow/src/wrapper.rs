//! The step wrapper's halves: the [`Lane`] budget mode makes for a step,
//! and what is done around every `run` ([`acquire`], [`release`]), with
//! the guards it hands `run` as a [`Held`].

use std::sync::MutexGuard;

use fb_core_budget::{Lease, LockId};
use fb_core_device::{GpuResources, ScopedDevice, Stream};
use fb_core_error_handling::{Error, Result, Status};

use crate::{Bus, Placement, locked, manager_index};

/// A step's streams on one GPU it spans.
pub struct GpuLane {
    gpu: usize,
    streams: Vec<Stream>,
}

impl GpuLane {
    /// The GPU.
    #[must_use]
    pub fn gpu(&self) -> usize {
        self.gpu
    }

    /// The step's streams there: its kernels and copies go on them.
    #[must_use]
    pub fn streams(&self) -> &[Stream] {
        &self.streams
    }
}

/// A step's streams, per GPU it spans, gpu0 first: made by the wrapper in
/// budget mode and kept in the step's slot. None for a host step, or on a
/// plan with no device behind it.
pub struct Lane {
    gpus: Vec<GpuLane>,
}

impl Lane {
    /// The GPUs the step spans, ascending, each with its streams.
    #[must_use]
    pub fn gpus(&self) -> &[GpuLane] {
        &self.gpus
    }
}

/// What `run` receives of the locks: the guards the wrapper took, in rank
/// order, before `run`, and the step's device made current. `run` never
/// sees a `Mutex`, so nothing takes a lock inside it.
pub struct Held<'a> {
    _current: Option<ScopedDevice>,
    managers: Vec<MutexGuard<'a, ()>>,
    gpus: Vec<Option<MutexGuard<'a, GpuResources>>>,
    taken: Vec<LockId>,
}

impl Held<'_> {
    /// GPU `gpu`'s resources.
    ///
    /// # Panics
    /// If the step did not declare that GPU's lock.
    #[track_caller]
    pub fn gpu(&mut self, gpu: usize) -> &mut GpuResources {
        self.gpus
            .get_mut(gpu)
            .and_then(Option::as_mut)
            .unwrap_or_else(|| panic!("the step did not declare the lock of gpu {gpu}"))
    }

    /// Every GPU's resources, gpu0 first, for a fan-out across them
    /// ([`crate::fan_out`]).
    ///
    /// # Panics
    /// If the step did not declare every GPU's lock.
    #[track_caller]
    pub fn all_gpus(&mut self) -> Vec<&mut GpuResources> {
        self.gpus
            .iter_mut()
            .enumerate()
            .map(|(gpu, g)| {
                &mut **g
                    .as_mut()
                    .unwrap_or_else(|| panic!("the step did not declare the lock of gpu {gpu}"))
            })
            .collect()
    }

    /// The locks held, in the order the wrapper took them: the rank.
    #[must_use]
    pub fn taken(&self) -> &[LockId] {
        &self.taken
    }
}

/// The GPUs a step at `placement` spans, of `n`.
fn spans(placement: Placement, n: usize) -> std::ops::Range<usize> {
    match placement {
        Placement::Host => 0..0,
        Placement::Gpu0 => 0..n.min(1),
        Placement::AllGpus => 0..n,
    }
}

/// Budget mode's half, after the step's `mem_budget`: check the GPU locks
/// it declared are GPUs it runs on, of the plan's (the ledger's, so a
/// device-free plan is checked as a real one), and make its `streams`
/// streams on each of them that has a device behind it.
///
/// # Errors
/// `InvalidConfig` for a declared GPU lock outside the step's GPUs.
pub(crate) fn make_lane<B: Bus>(
    bus: &B,
    step: &'static str,
    placement: Placement,
    streams: usize,
) -> Result<Lane> {
    let physical = bus.locks().physical();
    let spanned = spans(placement, bus.ledger().gpus());
    for lock in bus.ledger().locks_of(step) {
        if let LockId::Gpu(g) = lock
            && !spanned.contains(&g)
        {
            return Err(Error::config(format!(
                "step '{step}' declares the lock of gpu {g}, which it does not run on"
            )));
        }
    }
    Ok(Lane {
        gpus: spanned
            .filter(|&gpu| gpu < physical.len())
            .map(|gpu| GpuLane {
                gpu,
                streams: (0..streams).map(|_| Stream::new(physical[gpu])).collect(),
            })
            .collect(),
    })
}

/// Before every `run`: make the step's device current on this thread (gpu0's;
/// a fan-out's threads make their own current), take the step's declared
/// locks in rank order, bind a GPU's handle workspace the first time its
/// lock is taken, and in the ledger's NaN mode fill the lease, its host
/// regions and, on the step's first stream on each GPU, its device ones.
///
/// # Errors
/// A workspace bind, a fill or its sync failing; `InvalidConfig` for a
/// declared GPU lock with no device behind it, on a host-backed plan.
#[allow(
    clippy::disallowed_methods,
    reason = "the step wrapper's syncs: the lease's fill before `run`, and every stream after it, the v0.0.4 layer's edge (docs/programming_model.md, \"Steps on rustflow\")"
)]
pub(crate) fn acquire<'a, B: Bus>(
    bus: &'a B,
    step: &'static str,
    placement: Placement,
    lane: &Lane,
    lease: &mut Lease<'_>,
) -> Result<Held<'a>> {
    let locks = bus.locks();
    let ledger = bus.ledger();
    let current = (placement != Placement::Host)
        .then(|| locks.physical.first())
        .flatten()
        .map(|&p| ScopedDevice::new(p));
    let mut held = Held {
        _current: current,
        managers: Vec::new(),
        gpus: (0..locks.gpus.len()).map(|_| None).collect(),
        taken: Vec::new(),
    };
    // The declared set, in rank order: managers, then GPUs by index. A
    // step that holds a higher lock never waits for a lower one.
    for lock in ledger.locks_of(step) {
        match lock {
            LockId::Manager(id) => held
                .managers
                .push(locked(&locks.managers[manager_index(id)])),
            LockId::Gpu(g) => {
                // `make_lane` held every GPU lock to the plan's GPUs; a
                // host-backed plan has no device behind them.
                let Some(gpu) = locks.gpus.get(g) else {
                    return Err(Error::config(format!(
                        "step '{step}' takes the lock of gpu {g}, which has no device in this run"
                    )));
                };
                let mut guard = locked(gpu);
                if let Some(workspace) = ledger.take_workspace(g) {
                    guard.bind_workspace(workspace)?;
                }
                held.gpus[g] = Some(guard);
            }
        }
        held.taken.push(lock);
    }
    if ledger.nan_fill() {
        lease.fill_host();
        for gpu in &lane.gpus {
            if let Some(stream) = gpu.streams.first() {
                lease.fill_device(gpu.gpu, stream)?;
                // The step's other streams, and the GPU's, must find the
                // fill done.
                stream.sync()?;
            }
        }
    }
    Ok(held)
}

/// After every `run`, failed or not: synchronize every stream the step has
/// and each held GPU's stream, so that a finished node is finished device
/// work, and only then release the locks. The first failure, after every
/// sync was tried.
#[allow(
    clippy::disallowed_methods,
    reason = "the step wrapper's syncs: the lease's fill before `run`, and every stream after it, the v0.0.4 layer's edge (docs/programming_model.md, \"Steps on rustflow\")"
)]
pub(crate) fn release(lane: &Lane, held: Held<'_>) -> Status {
    let mut status = Ok(());
    let mut note = |synced: Status| {
        if status.is_ok() {
            status = synced;
        }
    };
    for gpu in &lane.gpus {
        for stream in &gpu.streams {
            note(stream.sync());
        }
    }
    for guard in held.gpus.iter().flatten() {
        note(guard.stream().sync());
    }
    // Only now the locks: the next holder finds no call in flight.
    drop(held);
    status
}
