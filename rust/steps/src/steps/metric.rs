//! The fitting metric (P|Q): built on the host directly in the solid
//! harmonics, in the host integrals' pool, into gpu0's pinned staging, and
//! uploaded to every GPU. A real
//! `Output`: the ERI step factors a copy of it, and a gradient reads it. A
//! host step: no library, no lock.
//!
//! One staging, in gpu0's pinned scratch, uploaded to every GPU under one
//! token: each later GPU's copy joins the first
//! ([`Pending::and_upload`](fb_core_device::Pending::and_upload)), so the
//! staging is held until every copy of it has run, and its pinned bytes do
//! not grow with the GPUs (`n_aux² × 8`, 104 MB at gly10's 3606 aux, once
//! rather than once per GPU).

use fb_core_budget::{Budget, DeviceBuf, Kind, Lease};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_flow::{Held, Lane, Placement, Slot, Step, View};

use crate::bus::{MainBus, Shared};
use crate::steps::{GPU0, staging, unpack_lower_in_place, upload};

/// The step.
#[derive(Clone, Copy)]
pub struct MetricStep;

/// (P|Q) on every GPU.
pub struct MetricOutput {
    gpus: Vec<DeviceBuf<f64>>,
}

impl MetricOutput {
    /// (P|Q) on each GPU, gpu0 first, `n_aux × n_aux`, symmetric.
    #[must_use]
    pub fn gpus(&self) -> &[DeviceBuf<f64>] {
        &self.gpus
    }
}

impl Step<MainBus> for MetricStep {
    const NAME: &'static str = "metric";
    type Output = MetricOutput;
    type Reads = ();

    fn slot(bus: &MainBus) -> &Slot<MetricOutput> {
        &bus.metric
    }

    fn placement(_: &Shared) -> Placement {
        Placement::AllGpus
    }

    fn mem_budget(shared: &Shared, budget: &Budget<'_>) -> Result<MetricOutput> {
        let q = shared.dims.n_aux;
        budget.scratch(GPU0, Kind::Pinned, staging(q * q).0.bytes());
        Ok(MetricOutput {
            gpus: (0..budget.gpus())
                .map(|gpu| budget.device(gpu, q * q))
                .collect(),
        })
    }

    fn run(
        out: &mut MetricOutput,
        view: &View<'_, MainBus, Self>,
        _: &mut Held<'_>,
        lane: &Lane,
        scratch: &Lease<'_>,
    ) -> Status {
        let shared = view.shared();
        let q = shared.dims.n_aux;
        let aux = shared
            .aux
            .as_ref()
            .ok_or_else(|| Error::config("the metric step runs with [eri]'s fitting set"))?;
        let mut staged = scratch.pinned(GPU0, staging(q * q).1);
        let triangle = &mut staged[..q * (q + 1) / 2];
        shared
            .host_pool()
            .install(|| fb_integrals_df_metric::metric_into(aux, triangle));
        unpack_lower_in_place(q, &mut staged);
        upload(lane, &mut out.gpus, staged)
    }
}
