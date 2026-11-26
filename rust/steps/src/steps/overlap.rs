//! The overlap S: built on the host directly in the solid harmonics, in
//! the host integrals' pool, into the step's pinned staging, and uploaded
//! to gpu0. A host step: it calls no
//! library, takes no lock, and runs beside the GPU work.

use fb_core_budget::{Budget, DeviceBuf, HeapSlab, Kind, Lease};
use fb_core_error_handling::{Result, Status};
use fb_core_flow::{Held, Lane, Placement, Slot, Step, View};

use crate::bus::{Calculation, MainBus, Shared};
use crate::steps::{GPU0, staging, unpack_lower_in_place, upload};

/// The step.
#[derive(Clone, Copy)]
pub struct OverlapStep;

/// S on gpu0.
pub struct OverlapOutput {
    s: DeviceBuf<f64>,
    s_host: Option<HeapSlab<f64>>,
}

impl OverlapOutput {
    /// S, `n_ao × n_ao`, symmetric.
    #[must_use]
    pub fn s(&self) -> &DeviceBuf<f64> {
        &self.s
    }

    /// S on the host, `n_ao × n_ao`, for `calculation = "overlap"`,
    /// which writes it as `overlap.npy`; `None` for any other.
    #[must_use]
    pub fn s_host(&self) -> Option<&[f64]> {
        self.s_host.as_deref()
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
        budget.scratch(GPU0, Kind::Pinned, staging(n * n).0.bytes());
        Ok(OverlapOutput {
            s: budget.device(GPU0, n * n),
            s_host: (shared.config.calculation == Some(Calculation::Overlap))
                .then(|| budget.heap(n * n)),
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
        let mut staged = scratch.pinned(GPU0, staging(n * n).1);
        let triangle = &mut staged[..n * (n + 1) / 2];
        shared
            .host_pool()
            .install(|| fb_integrals_s_overlap_host::overlap_into(&shared.basis, triangle));
        unpack_lower_in_place(n, &mut staged);
        if let Some(host) = out.s_host.as_mut() {
            host.copy_from_slice(&staged[..n * n]);
        }
        upload(lane, std::slice::from_mut(&mut out.s), staged)
    }
}
