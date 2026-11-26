//! The fitting metric (P|Q): built on the host directly in the solid
//! harmonics, on rayon's threads, and uploaded to every GPU. A real
//! `Output`: the ERI step factors a copy of it, and a gradient reads it. A
//! host step: no library, no lock.
//!
//! One staging per GPU, each in that GPU's pinned scratch: a copy holds its
//! host end until a sync covers it, so one staging cannot be in flight on
//! several GPUs' streams at once. The interim until a fan-out of one host
//! end is expressible.

use fb_core_budget::{Budget, DeviceBuf, Kind, Lease};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_flow::{Held, Lane, Placement, Slot, Step, View};

use crate::bus::{MainBus, Shared};
use crate::steps::{staging, unpack_lower, upload};

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
        let bytes = staging(q).0.bytes();
        Ok(MetricOutput {
            gpus: (0..budget.gpus())
                .map(|gpu| {
                    budget.scratch(gpu, Kind::Pinned, bytes);
                    budget.device(gpu, q * q)
                })
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
        let packed = fb_integrals_df_metric::metric(aux);
        for (gpu, metric) in out.gpus.iter_mut().enumerate() {
            let mut staged = scratch.pinned(gpu, staging(q).1);
            unpack_lower(&packed, q, &mut staged);
            upload(lane, gpu, metric, staged)?;
        }
        Ok(())
    }
}
