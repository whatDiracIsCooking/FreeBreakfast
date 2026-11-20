//! X = S^(-1/2), the symmetric orthogonalizer: an eigensolve of S on the
//! library stream, over the device's workspace, then `V diag(w^(-1/2)) Vᵀ`.

use fb_core_error_handling::{Error, Result};
use fb_core_wwr::blas::{dgmm, gemm};
use fb_core_wwr::copy::d2d;
use fb_recipes_sys::inverse_sqrt;
use wwr_sys::wwr_blas_side_mode::WWR_BLAS_SIDE_RIGHT;

use crate::budget::{Budget, DeviceBuf, PinnedBuf};
use crate::bus::{Lane, MainBus, Shared};
use crate::device::download;
use crate::flow::{Next, Placement, Slot, Step, View};
use crate::steps::overlap::OverlapStep;
use crate::steps::{GPU0, OP_N, OP_T, syevd_workspace};

/// The eigenvalues of S under which a direction is dropped: the old tree's
/// `ORTHOGONALIZE_EIGENVALUE_FLOOR`.
pub const EIGENVALUE_FLOOR: f64 = 1e-10;

/// The step.
#[derive(Clone, Copy)]
pub struct OrthogonalizerStep;

/// X, and the eigensolve's products behind it.
pub struct OrthogonalizerOutput {
    x: DeviceBuf<f64>,
    vectors: DeviceBuf<f64>,
    values: DeviceBuf<f64>,
    scaled: DeviceBuf<f64>,
    info: DeviceBuf<i32>,
    info_host: PinnedBuf<i32>,
}

impl OrthogonalizerOutput {
    /// X, `n_ao × n_ao`, symmetric.
    pub fn x(&self) -> &DeviceBuf<f64> {
        &self.x
    }
}

impl Step<MainBus> for OrthogonalizerStep {
    const NAME: &'static str = "orthogonalizer";
    type Output = OrthogonalizerOutput;
    type Reads = (OverlapStep,);

    fn slot(bus: &MainBus) -> &Slot<OrthogonalizerOutput, Lane> {
        &bus.orthogonalizer
    }

    fn placement(_: &Shared) -> Placement {
        Placement::Gpu(GPU0)
    }

    fn mem_budget(shared: &Shared, _: &Lane, budget: &Budget<'_>) -> Result<OrthogonalizerOutput> {
        let n = shared.dims.n_ao;
        let out = OrthogonalizerOutput {
            x: budget.device(GPU0, n * n),
            vectors: budget.device(GPU0, n * n),
            values: budget.device(GPU0, n),
            scaled: budget.device(GPU0, n * n),
            info: budget.device(GPU0, 1),
            info_host: budget.pinned(GPU0, 1),
        };
        // The one library call with a workspace: a max, the handle's.
        budget.workspace(GPU0, syevd_workspace(shared, n)?);
        Ok(out)
    }

    fn run(out: &mut OrthogonalizerOutput, view: &View<'_, MainBus, Self>) -> Result<Next> {
        let n = view.shared().dims.n_ao;
        let lane = view.lane().gpu();
        let overlap = view.read(OverlapStep);
        let ni = n as i32;
        lane.library(view.shared().library(GPU0), |library| {
            d2d(&mut out.vectors, overlap.s(), library.stream())?;
            library.syevd(
                Self::NAME,
                n,
                &mut out.vectors,
                &mut out.values,
                &mut out.info,
            )?;
            inverse_sqrt(library.stream(), n, &mut out.values, EIGENVALUE_FLOOR, None)?;
            let blas = library.blas();
            // scaled = V diag(w^(-1/2)); X = scaled Vᵀ.
            dgmm(
                blas,
                WWR_BLAS_SIDE_RIGHT,
                ni,
                ni,
                &out.vectors,
                ni,
                &out.values,
                1,
                &mut out.scaled,
                ni,
            )?;
            gemm(
                blas,
                OP_N,
                OP_T,
                ni,
                ni,
                ni,
                1.0,
                &out.scaled,
                ni,
                &out.vectors,
                ni,
                0.0,
                &mut out.x,
                ni,
            )?;
            download(&mut out.info_host, &out.info, library.stream())
        })?;
        // The solver's flag is read on the host: the step waits for it.
        lane.sync()?;
        match out.info_host[0] {
            0 => Ok(Next::Done),
            info => Err(Error::numerical(format!("syevd of S: info = {info}"))),
        }
    }
}
