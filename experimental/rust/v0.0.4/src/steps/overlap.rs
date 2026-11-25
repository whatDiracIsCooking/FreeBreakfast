//! The overlap S: built on the host directly in the solid harmonics, on
//! rayon's threads, and uploaded to gpu0. A host step: it calls no
//! library, takes no lock, and runs beside the GPU work.

use fb_core_error_handling::{Result, Status};

use crate::budget::{Budget, DeviceBuf, Kind, Layout, Lease};
use crate::bus::{Held, Lane, MainBus, Shared};
use crate::flow::{Placement, Slot, Step, View};
use crate::steps::{GPU0, unpack_lower};

/// The step.
#[derive(Clone, Copy)]
pub struct OverlapStep;

/// S on gpu0.
pub struct OverlapOutput {
    s: DeviceBuf<f64>,
}

impl OverlapOutput {
    /// S, `n_ao × n_ao`, symmetric.
    pub fn s(&self) -> &DeviceBuf<f64> {
        &self.s
    }
}

impl Step<MainBus> for OverlapStep {
    const NAME: &'static str = "overlap";
    type Output = OverlapOutput;
    type Reads = ();

    fn slot(bus: &MainBus) -> &Slot<OverlapOutput, Lane> {
        &bus.overlap
    }

    fn placement(_: &Shared) -> Placement {
        Placement::Gpu0
    }

    fn mem_budget(shared: &Shared, budget: &Budget<'_>) -> Result<OverlapOutput> {
        let n = shared.dims.n_ao;
        budget.scratch(GPU0, Kind::Pinned, n * n * size_of::<f64>());
        Ok(OverlapOutput {
            s: budget.device(GPU0, n * n),
        })
    }

    fn run(
        out: &mut OverlapOutput,
        view: &View<'_, MainBus, Self>,
        _: &mut Held<'_>,
        lane: &mut Lane,
        scratch: &Lease<'_>,
    ) -> Status {
        let shared = view.shared();
        let n = shared.dims.n_ao;
        // The host build returns its packed triangle, an O(n_ao²)
        // allocation inside `run`: within the small-host-scratch policy.
        let packed = fb_s_overlap_host::overlap(&shared.basis);
        let staged = scratch.pinned(GPU0, Layout::new().span::<f64>(n * n));
        unpack_lower(&packed, n, staged);
        lane.gpu0().stream(0).upload(&mut out.s, staged)
    }
}
