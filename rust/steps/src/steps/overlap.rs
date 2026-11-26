//! The overlap S: built on the host directly in the solid harmonics, on
//! rayon's threads, and uploaded to gpu0. A host step: it calls no
//! library, takes no lock, and runs beside the GPU work.

use fb_core_budget::{Budget, DeviceBuf, Kind, Lease};
use fb_core_error_handling::{Result, Status};
use fb_core_flow::{Held, Lane, Placement, Slot, Step, View};

use crate::bus::{MainBus, Shared};
use crate::steps::{GPU0, staging, unpack_lower, upload};

/// The step.
#[derive(Clone, Copy)]
pub struct OverlapStep;

/// S on gpu0.
pub struct OverlapOutput {
    s: DeviceBuf<f64>,
}

impl OverlapOutput {
    /// S, `n_ao × n_ao`, symmetric.
    #[must_use]
    pub fn s(&self) -> &DeviceBuf<f64> {
        &self.s
    }
}

impl Step<MainBus> for OverlapStep {
    const NAME: &'static str = "overlap";
    type Output = OverlapOutput;
    type Reads = ();

    fn slot(bus: &MainBus) -> &Slot<OverlapOutput> {
        &bus.overlap
    }

    fn placement(_: &Shared) -> Placement {
        Placement::Gpu0
    }

    fn mem_budget(shared: &Shared, budget: &Budget<'_>) -> Result<OverlapOutput> {
        let n = shared.dims.n_ao;
        budget.scratch(GPU0, Kind::Pinned, staging(n).0.bytes());
        Ok(OverlapOutput {
            s: budget.device(GPU0, n * n),
        })
    }

    fn run(
        out: &mut OverlapOutput,
        view: &View<'_, MainBus, Self>,
        _: &mut Held<'_>,
        lane: &Lane,
        scratch: &Lease<'_>,
    ) -> Status {
        let shared = view.shared();
        let n = shared.dims.n_ao;
        // The host build returns its packed triangle, an O(n_ao²)
        // allocation inside `run`: small host scratch.
        let packed = fb_integrals_s_overlap_host::overlap(&shared.basis);
        let mut staged = scratch.pinned(GPU0, staging(n).1);
        unpack_lower(&packed, n, &mut staged);
        upload(lane, GPU0, &mut out.s, staged)
    }
}
