//! S, the spherical AO overlap: the class kernels on the step's own stream,
//! the transform on the library's.

use fb_core_error_handling::Result;
use fb_core_wwr::device_slice::DeviceSliceMut;
use fb_core_wwr::handle::DeviceHandleStream;
use fb_core_wwr::memset;
use fb_integrals_sys::fb_overlap_class;
use fb_recipes_sys::mirror_triangle;

use crate::budget::{Budget, DeviceBuf};
use crate::bus::{Lane, MainBus, Shared};
use crate::device::launched;
use crate::flow::{Next, Placement, Slot, Step, View};
use crate::steps::harmonics::HarmonicsStep;
use crate::steps::records::RecordsStep;
use crate::steps::{GPU0, to_spherical};

/// The step.
#[derive(Clone, Copy)]
pub struct OverlapStep;

/// S, and the Cartesian matrix and the half-transformed one behind it.
pub struct OverlapOutput {
    s: DeviceBuf<f64>,
    cart: DeviceBuf<f64>,
    work: DeviceBuf<f64>,
}

impl OverlapOutput {
    /// S, `n_ao × n_ao`.
    pub fn s(&self) -> &DeviceBuf<f64> {
        &self.s
    }
}

impl Step<MainBus> for OverlapStep {
    const NAME: &'static str = "overlap";
    type Output = OverlapOutput;
    type Reads = (RecordsStep, HarmonicsStep);

    fn slot(bus: &MainBus) -> &Slot<OverlapOutput, Lane> {
        &bus.overlap
    }

    fn placement(_: &Shared) -> Placement {
        Placement::Gpu(GPU0)
    }

    fn mem_budget(shared: &Shared, _: &Lane, budget: &Budget<'_>) -> Result<OverlapOutput> {
        let d = shared.dims;
        Ok(OverlapOutput {
            s: budget.device(GPU0, d.n_ao * d.n_ao),
            cart: budget.device(GPU0, d.n_cart * d.n_cart),
            work: budget.device(GPU0, d.n_cart * d.n_ao),
        })
    }

    fn run(out: &mut OverlapOutput, view: &View<'_, MainBus, Self>) -> Result<Next> {
        let d = view.shared().dims;
        let lane = view.lane().gpu();
        let stream = lane.stream();
        let records = view.read(RecordsStep);
        let harmonics = view.read(HarmonicsStep);

        memset::zero(&mut out.cart, stream)?;
        for class in records.classes() {
            let cls = class.view();
            // SAFETY: the class's records are the records step's, finished
            // before this step ran (the edge); `cart` is n_cart² doubles, by
            // `mem_budget`, and this step's alone.
            let status = unsafe {
                fb_overlap_class(
                    stream.stream(),
                    &cls,
                    d.n_cart as i32,
                    out.cart.as_mut_ptr(),
                )
            };
            launched(
                &format!("fb_overlap_class ({}, {})", class.la(), class.lb()),
                status,
            )?;
        }
        // The kernels add into the column-major upper triangle alone.
        mirror_triangle(stream, false, d.n_cart, &mut out.cart, d.n_cart)?;
        lane.library(view.shared().library(GPU0), |library| {
            to_spherical(
                library.blas(),
                d.n_cart,
                d.n_ao,
                harmonics.u(),
                &out.cart,
                &mut out.work,
                &mut out.s,
            )
        })?;
        Ok(Next::Done)
    }
}
