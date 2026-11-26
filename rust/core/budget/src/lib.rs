//! `core::budget`: the v0.0.4 layer's memory budget (`docs/memory.md`,
//! "The budget"). The ledger of caps and counters, the [`Budget`] handle a
//! step's `mem_budget` allocates through, the buffers it hands back, and
//! step scratch: declared per step, packed by the final node
//! ([`Ledger::finish`]), lent to `run` as a [`Lease`].
//!
//! Ported from the v0.0.4 prototype's `src/budget.rs` and `src/host.rs`
//! (`../experimental/rust/v0.0.4/README.md`, "The memory budget", "Step
//! scratch"); the C++ tree's counterpart is pre-flight and the footprints,
//! which this replaces by counting the allocations themselves.
//!
//! Each [`Budget`] call for memory a step's `Output` owns, in order: adds
//! the bytes to the kind's atomic counter and to the step's row of the
//! table; checks the counter against the cap, and sets the ledger's `over`
//! flag past it; allocates, unless the flag is set. [`Budget::scratch`] and
//! [`Budget::locks`] only record a row. [`Ledger::finish`], the budget
//! DAG's last node, packs each gpu and kind's scratch rows into one slab by
//! the DAG's closure ([`fb_core_pack`]), adds each packed peak and the
//! fixed handle workspace
//! (`fb_core_device::HANDLE_WORKSPACE_BYTES`) to its counter, checks again, and either
//! refuses the whole plan with the table or allocates the slabs and the
//! workspaces.
//!
//! ```
//! use fb_core_budget::{Backing, Cap, Caps, Kind, Layout, Ledger};
//!
//! let cap = |bytes| Cap { written: Some(bytes), used: bytes };
//! let caps = Caps { device: vec![cap(1 << 30)], host: cap(1 << 30) };
//! let ledger = Ledger::new(caps, Backing::Host, vec![0], false);
//!
//! // Two steps, `b` reading `a`: their scratch may share bytes.
//! let mut layout = Layout::new();
//! let span = layout.span::<f64>(1024);
//! for step in ["a", "b"] {
//!     ledger.budget(step).scratch(0, Kind::Pinned, layout.bytes());
//! }
//! ledger.finish(&["a", "b"], &[vec![], vec![0]]).unwrap();
//! assert_eq!(ledger.slabs()[1].peak, 8192);
//!
//! // `a`'s run is lent its region, and carves its tenant from it.
//! let lease = ledger.lease("a");
//! let mut part = lease.pinned(0, span);
//! part.fill(1.0);
//! assert_eq!(part.len(), 1024);
//! ```
//!
//! No step holds a `Budget` in execute mode: `run` is never given one, and
//! the buffers' constructors are private to this crate, the one crate
//! that may call the raw `wwr-sys` allocators (`clippy.toml`'s ban). What
//! a `run` takes from the plain heap is outside the budget, small host
//! scratch by policy: the [`AllocationRecord`] a test binary installs
//! holds it under [`RECORD_THRESHOLD`] per allocation, and the table
//! prints an execute's peak heap growth as its unbudgeted row
//! ([`Ledger::measure_execute`]).

mod buffers;
mod host;
mod lease;
mod ledger;
mod raw;
mod record;
mod tenant;
#[cfg(test)]
mod tests;

pub use buffers::{DeviceBuf, HeapSlab, PinnedBuf, Plain};
pub use host::host_available;
pub use lease::{HostPart, Layout, Lease, ScratchBuf, Span};
pub use ledger::{Budget, Ledger, RegionInfo, SlabInfo};
pub use record::{AllocationRecord, RECORD_THRESHOLD};
pub use tenant::{DeviceTenant, PinnedTenant};

use contracts::debug_requires;
use fb_core_error_handling::Result;
use fb_core_log::eq;

/// What a scratch slab is aligned to, and every region inside it.
const ALIGN: u64 = 256;

/// The two kinds of step scratch.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    /// Device memory on one gpu.
    Device,
    /// Page-locked host memory, staged for one gpu.
    Pinned,
}

/// Where a ledger's bytes come from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backing {
    /// Device and pinned memory through wwr.
    Real,
    /// Everything on the plain host heap, device and pinned kinds included:
    /// the device-free tests run their steps with it, on no gpu.
    Host,
    /// Nothing is allocated: counted and checked only, every buffer empty.
    /// A plan budgeted so cannot be executed.
    Dry,
}

/// A cap as the config wrote it and as the plan is held to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cap {
    /// What the config gave; `None` for a defaulted cap.
    pub written: Option<u64>,
    /// What the plan is held against.
    pub used: u64,
}

impl Cap {
    /// Clamp a cap to what the machine has at init. A defaulted cap is the
    /// reading less `margin`; an explicit one is used as written, the user
    /// owning the margin, unless it is past the reading, and then it is the
    /// reading itself. With no reading (a dry budget on no gpu) the cap is
    /// as written, or none.
    ///
    /// ```
    /// use fb_core_budget::Cap;
    ///
    /// let gib = 1u64 << 30;
    /// // Past the machine: the reading itself, no margin taken.
    /// assert_eq!(Cap::resolve(Some(64 * gib), Some(10 * gib), gib).used, 10 * gib);
    /// ```
    #[must_use]
    pub fn resolve(written: Option<u64>, reading: Option<u64>, margin: u64) -> Cap {
        let used = match (written, reading) {
            (Some(w), Some(r)) => w.min(r),
            (Some(w), None) => w,
            (None, Some(r)) => r.saturating_sub(margin),
            (None, None) => u64::MAX,
        };
        Cap { written, used }
    }
}

/// The device margin of a defaulted cap: the larger of 512 MiB and 5% of
/// the gpu's total memory.
#[must_use]
pub fn device_margin(total: u64) -> u64 {
    (512 << 20).max(total / 20)
}

/// The host margin of a defaulted cap: the larger of 1 GiB and 5% of the
/// available memory.
#[must_use]
pub fn host_margin(available: u64) -> u64 {
    (1 << 30).max(available / 20)
}

/// The caps the counters are held against.
#[derive(Clone, Debug)]
pub struct Caps {
    /// Device memory, per gpu.
    pub device: Vec<Cap>,
    /// The host: heap and pinned bytes together.
    pub host: Cap,
}

impl Caps {
    /// The caps of a run over the gpus `physical` (the device each is on),
    /// each resolved against the machine ([`Cap::resolve`]): `device[i]`
    /// and `host` as the config wrote them. Under [`Backing::Real`] a
    /// gpu's reading is its device's free memory and its margin
    /// [`device_margin`] of the total, each split evenly between the gpus
    /// on one device; read it after the handles exist, so the reading sees
    /// them. Otherwise a device cap is as written. The host's reading is
    /// [`host_available`], less [`host_margin`] when defaulted.
    ///
    /// # Errors
    /// The device's or the host's reading.
    #[debug_requires(eq!(device.len(), physical.len()), "one written device cap per gpu")]
    pub fn read(
        physical: &[i32],
        device: &[Option<u64>],
        host: Option<u64>,
        backing: Backing,
    ) -> Result<Caps> {
        let mut caps = Vec::with_capacity(physical.len());
        for (&p, &written) in physical.iter().zip(device) {
            caps.push(if backing == Backing::Real {
                let sharing = physical.iter().filter(|&&q| q == p).count() as u64;
                let (free, total) = fb_core_device::memory(p)?;
                Cap::resolve(
                    written,
                    Some(free / sharing),
                    device_margin(total) / sharing,
                )
            } else {
                Cap::resolve(written, None, 0)
            });
        }
        let available = host_available()?;
        Ok(Caps {
            device: caps,
            host: Cap::resolve(host, Some(available), host_margin(available)),
        })
    }
}

/// A manager: a shared product with mutable state of its own, reached by
/// its readers under its lock.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ManagerId {
    /// The `EriManager`.
    Eri,
}

/// A lock, in the one global rank: every manager first, by its id, then
/// the GPUs by index. `Ord` is derived in that order, and the wrapper takes
/// a step's declared set ascending, so no step holds a higher lock while it
/// waits for a lower one.
///
/// ```
/// use fb_core_budget::{LockId, ManagerId};
///
/// assert!(LockId::Manager(ManagerId::Eri) < LockId::Gpu(0));
/// assert!(LockId::Gpu(0) < LockId::Gpu(1));
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum LockId {
    /// A manager's lock.
    Manager(ManagerId),
    /// A GPU's `GpuResources`.
    Gpu(usize),
}

/// `n` bytes in the largest binary unit that keeps three digits; `u64::MAX`,
/// no cap, is `none`.
///
/// ```
/// assert_eq!(fb_core_budget::bytes(1536), "1.50 KiB");
/// ```
#[must_use]
pub fn bytes(n: u64) -> String {
    const UNITS: [&str; 4] = ["KiB", "MiB", "GiB", "TiB"];
    if n == u64::MAX {
        return "none".to_owned();
    }
    if n < 1024 {
        return format!("{n} B");
    }
    let mut value = n as f64 / 1024.0;
    let mut unit = 0;
    while value >= 1024.0 && unit + 1 < UNITS.len() {
        value /= 1024.0;
        unit += 1;
    }
    format!("{value:.2} {}", UNITS[unit])
}
