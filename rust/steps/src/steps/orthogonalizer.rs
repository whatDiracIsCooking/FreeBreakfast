//! X = S^(-1/2), the symmetric orthogonalizer: an eigensolve of S on
//! gpu0's stream, under gpu0's lock, then `V diag(w^(-1/2)) Vᵀ`, each
//! eigenvalue at or below [`EIGENVALUE_FLOOR`] dropped. The simple library
//! user: everything goes on the GPU's stream, in the order it is enqueued,
//! with no event. The eigenvectors, the scaled copy, syevd's work buffer
//! and its flag are step scratch.

use fb_core_budget::{Budget, DeviceBuf, Kind, Layout, Lease, LockId, Span};
use fb_core_device::{wwr_blas_operation, wwr_blas_side_mode};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_flow::{Held, Lane, Placement, Slot, Step, View};

use crate::bus::{MainBus, Shared};
use crate::steps::GPU0;
use crate::steps::overlap::OverlapStep;

/// The eigenvalues of S at or below which a direction is dropped: the
/// C++ tree's `kOrthogonalizeEigenvalueFloor`.
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
    #[must_use]
    pub fn x(&self) -> &DeviceBuf<f64> {
        &self.x
    }
}

/// The step's device scratch, one layout: `mem_budget` declares its
/// bytes, `run` carves its spans.
struct DeviceLayout {
    bytes: usize,
    vectors: Span<f64>,
    values: Span<f64>,
    scaled: Span<f64>,
    work: Span<f64>,
    info: Span<i32>,
}

fn device_layout(n: usize, lwork: usize) -> DeviceLayout {
    let mut l = Layout::new();
    DeviceLayout {
        vectors: l.span(n * n),
        values: l.span(n),
        scaled: l.span(n * n),
        work: l.span(lwork),
        info: l.span(1),
        bytes: l.bytes(),
    }
}

/// The step's pinned scratch: the solver's flag, read on the host.
fn pinned_layout() -> (Layout, Span<i32>) {
    let mut l = Layout::new();
    let info = l.span(1);
    (l, info)
}

impl Step<MainBus> for OrthogonalizerStep {
    const NAME: &'static str = "orthogonalizer";
    type Output = OrthogonalizerOutput;
    type Reads = (OverlapStep,);

    fn slot(bus: &MainBus) -> &Slot<OrthogonalizerOutput> {
        &bus.orthogonalizer
    }

    fn placement(_: &Shared) -> Placement {
        Placement::Gpu0
    }

    fn mem_budget(shared: &Shared, budget: &Budget<'_>) -> Result<OrthogonalizerOutput> {
        let n = shared.dims.n_ao;
        // The work buffer is the step's, sized by the query at the call's
        // arguments.
        let l = device_layout(n, shared.syevd_lwork(budget, n)?);
        budget.scratch(GPU0, Kind::Device, l.bytes);
        budget.scratch(GPU0, Kind::Pinned, pinned_layout().0.bytes());
        budget.locks([LockId::Gpu(GPU0)]);
        Ok(OrthogonalizerOutput {
            x: budget.device(GPU0, n * n),
        })
    }

    fn run(
        out: &mut OrthogonalizerOutput,
        view: &View<'_, MainBus, Self>,
        held: &mut Held<'_>,
        _: &Lane,
        scratch: &Lease<'_>,
    ) -> Status {
        let n = view.shared().dims.n_ao;
        let ni = i32::try_from(n)
            .map_err(|_| Error::unsupported(format!("{n} AOs is past the solver's i32")))?;
        let overlap = view.read(OverlapStep);
        let gpu = held.gpu(GPU0);
        let l = device_layout(n, gpu.syevd_lwork(n)?);
        let mut vectors = scratch.device(GPU0, l.vectors);
        let mut values = scratch.device(GPU0, l.values);
        let mut scaled = scratch.device(GPU0, l.scaled);
        let mut work = scratch.device(GPU0, l.work);
        let mut info = scratch.device(GPU0, l.info);

        gpu.stream().copy_device(&mut vectors, overlap.s())?;
        gpu.syevd(n, &mut vectors, &mut values, &mut work, &mut info)?;
        gpu.inverse_sqrt(&mut values, EIGENVALUE_FLOOR)?;
        // scaled = V diag(w^(-1/2)); X = scaled Vᵀ.
        gpu.dgmm(
            wwr_blas_side_mode::WWR_BLAS_SIDE_RIGHT,
            ni,
            ni,
            &vectors,
            ni,
            &values,
            &mut scaled,
            ni,
        )?;
        gpu.gemm(
            wwr_blas_operation::WWR_BLAS_OP_N,
            wwr_blas_operation::WWR_BLAS_OP_T,
            ni,
            ni,
            ni,
            1.0,
            &scaled,
            ni,
            &vectors,
            ni,
            0.0,
            &mut out.x,
            ni,
        )?;
        let flag = scratch.pinned(GPU0, pinned_layout().1);
        // The solver's flag, read on the host after the step's last
        // enqueue: the one sync inside a `run` this layer allows, which
        // waits for no part of the step still to come
        // (docs/gpu_layer.md, "The v0.0.4 runtime").
        #[allow(
            clippy::disallowed_methods,
            reason = "a step reading a solver's flag after its last enqueue (docs/gpu_layer.md, \"The v0.0.4 runtime\")"
        )]
        let flag = gpu.stream().download(flag, &info)?.sync()?;
        match flag[0] {
            0 => Ok(()),
            info => Err(Error::numerical(format!("syevd of S: info = {info}"))),
        }
    }
}
