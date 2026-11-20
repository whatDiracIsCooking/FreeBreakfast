//! U, the orbital basis's Cartesian-to-spherical transform, on the device:
//! computed once on the host and read by every step that transforms.

use fb_core_error_handling::Result;

use crate::budget::{Budget, DeviceBuf, PinnedBuf};
use crate::bus::{Lane, MainBus, Shared};
use crate::device::upload;
use crate::flow::{Next, Placement, Slot, Step, View};
use crate::steps::GPU0;

/// The step.
#[derive(Clone, Copy)]
pub struct HarmonicsStep;

/// U, `n_cart × n_ao` column-major, and the pinned copy it is staged in.
pub struct HarmonicsOutput {
    u: DeviceBuf<f64>,
    staged: PinnedBuf<f64>,
}

impl HarmonicsOutput {
    /// U on the device.
    pub fn u(&self) -> &DeviceBuf<f64> {
        &self.u
    }
}

impl Step<MainBus> for HarmonicsStep {
    const NAME: &'static str = "harmonics";
    type Output = HarmonicsOutput;
    type Reads = ();

    fn slot(bus: &MainBus) -> &Slot<HarmonicsOutput, Lane> {
        &bus.harmonics
    }

    fn placement(_: &Shared) -> Placement {
        Placement::Gpu(GPU0)
    }

    fn mem_budget(shared: &Shared, _: &Lane, budget: &Budget<'_>) -> Result<HarmonicsOutput> {
        let len = shared.dims.n_cart * shared.dims.n_ao;
        Ok(HarmonicsOutput {
            u: budget.device(GPU0, len),
            staged: budget.pinned(GPU0, len),
        })
    }

    fn run(out: &mut HarmonicsOutput, view: &View<'_, MainBus, Self>) -> Result<Next> {
        fb_basis::spherical_transform_matrix(&view.shared().basis, &mut out.staged);
        upload(&mut out.u, &out.staged, view.lane().gpu().stream())?;
        Ok(Next::Done)
    }
}
