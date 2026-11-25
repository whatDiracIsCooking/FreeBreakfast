//! X = S^(-1/2), the symmetric orthogonalizer: an eigensolve of S on
//! gpu0's stream, under gpu0's lock, then `V diag(w^(-1/2)) Vᵀ`. The
//! simple library user: everything goes on the GPU's stream, in the order
//! it is enqueued, with no event. The eigenvectors and syevd's work buffer
//! are step scratch.

use fb_core_error_handling::{Error, Result, Status};

use crate::budget::{Budget, DeviceBuf, Kind, Layout, Lease, LockId, Span};
use crate::bus::{Held, Lane, MainBus, Shared};
use crate::device::{OP_N, OP_T};
use crate::flow::{Placement, Slot, Step, View};
use crate::steps::GPU0;
use crate::steps::overlap::OverlapStep;

/// The eigenvalues of S under which a direction is dropped.
pub const EIGENVALUE_FLOOR: f64 = 1e-10;

/// The step.
#[derive(Clone, Copy)]
pub struct OrthogonalizerStep;

/// X on gpu0.
pub struct OrthogonalizerOutput {
    x: DeviceBuf<f64>,
}

impl OrthogonalizerOutput {
    /// X, `n_ao × n_ao`, symmetric.
    pub fn x(&self) -> &DeviceBuf<f64> {
        &self.x
    }
}

/// The step's one layout: `mem_budget` declares its bytes, `run` carves
/// its spans.
struct OrthogonalizerLayout {
    bytes: usize,
    vectors: Span<f64>,
    values: Span<f64>,
    scaled: Span<f64>,
    work: Span<f64>,
    info: Span<i32>,
}

fn layout(n: usize, lwork: usize) -> OrthogonalizerLayout {
    let mut l = Layout::new();
    OrthogonalizerLayout {
        vectors: l.span(n * n),
        values: l.span(n),
        scaled: l.span(n * n),
        work: l.span(lwork),
        info: l.span(1),
        bytes: l.bytes(),
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
        Placement::Gpu0
    }

    fn mem_budget(shared: &Shared, budget: &Budget<'_>) -> Result<OrthogonalizerOutput> {
        let n = shared.dims.n_ao;
        // The work buffer is the step's, sized by the query at the call's
        // arguments.
        let l = layout(n, shared.syevd_lwork(n)?);
        budget.scratch(GPU0, Kind::Device, l.bytes);
        budget.scratch(GPU0, Kind::Pinned, size_of::<i32>());
        budget.locks([LockId::Gpu(GPU0)]);
        Ok(OrthogonalizerOutput {
            x: budget.device(GPU0, n * n),
        })
    }

    fn run(
        out: &mut OrthogonalizerOutput,
        view: &View<'_, MainBus, Self>,
        held: &mut Held<'_>,
        _: &mut Lane,
        scratch: &Lease<'_>,
    ) -> Status {
        let n = view.shared().dims.n_ao;
        let ni = n as i32;
        let overlap = view.read(OverlapStep);
        let gpu = held.gpu(GPU0);
        let l = layout(n, gpu.syevd_lwork(n)?);
        let mut vectors = scratch.device(GPU0, l.vectors);
        let mut values = scratch.device(GPU0, l.values);
        let mut scaled = scratch.device(GPU0, l.scaled);
        let mut work = scratch.device(GPU0, l.work);
        let mut info = scratch.device(GPU0, l.info);
        let h_info = scratch.pinned(GPU0, Layout::new().span::<i32>(1));

        gpu.stream().copy_device(&mut vectors, overlap.s())?;
        gpu.syevd(n, &mut vectors, &mut values, &mut work, &mut info)?;
        gpu.inverse_sqrt(&mut values, EIGENVALUE_FLOOR)?;
        // scaled = V diag(w^(-1/2)); X = scaled Vᵀ.
        gpu.scale_columns(ni, &vectors, &values, &mut scaled)?;
        gpu.gemm(
            OP_N, OP_T, ni, ni, ni, 1.0, &scaled, ni, &vectors, ni, 0.0, &mut out.x, ni,
        )?;
        gpu.stream().download(h_info, &info)?;
        // The solver's flag is read on the host: the step waits for it.
        gpu.stream().sync()?;
        match h_info[0] {
            0 => Ok(()),
            info => Err(Error::numerical(format!("syevd of S: info = {info}"))),
        }
    }
}
