//! The fitting metric (P|Q): built on the host directly in the solid
//! harmonics, on rayon's threads, and uploaded to every GPU. A real
//! `Output`: `eri` factors a copy of it, and a gradient reads it. A host
//! step: no library, no lock.

use fb_core_error_handling::{Result, Status};

use crate::budget::{Budget, DeviceBuf, Kind, Layout, Lease};
use crate::bus::{Held, Lane, MainBus, Shared};
use crate::flow::{Placement, Slot, Step, View};
use crate::steps::{GPU0, unpack_lower};

/// The step.
#[derive(Clone, Copy)]
pub struct MetricStep;

/// (P|Q) on every GPU.
pub struct MetricOutput {
    gpus: Vec<DeviceBuf<f64>>,
}

impl MetricOutput {
    /// (P|Q) on `gpu`, `n_aux × n_aux`, symmetric.
    pub fn on(&self, gpu: usize) -> &DeviceBuf<f64> {
        &self.gpus[gpu]
    }
}

impl Step<MainBus> for MetricStep {
    const NAME: &'static str = "metric";
    type Output = MetricOutput;
    type Reads = ();

    fn slot(bus: &MainBus) -> &Slot<MetricOutput, Lane> {
        &bus.metric
    }

    fn placement(_: &Shared) -> Placement {
        Placement::AllGpus
    }

    fn mem_budget(shared: &Shared, budget: &Budget<'_>) -> Result<MetricOutput> {
        let q = shared.dims.n_aux;
        // One staging, in gpu0's pinned scratch; every GPU's upload reads
        // it.
        budget.scratch(GPU0, Kind::Pinned, q * q * size_of::<f64>());
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
        lane: &mut Lane,
        scratch: &Lease<'_>,
    ) -> Status {
        let shared = view.shared();
        let q = shared.dims.n_aux;
        let aux = shared.aux.as_ref().expect("metric runs with a fitting set");
        let packed = fb_df_metric_host::metric(aux);
        let staged = scratch.pinned(GPU0, Layout::new().span::<f64>(q * q));
        unpack_lower(&packed, q, staged);
        for (gpu, metric) in lane.gpus.iter().zip(&mut out.gpus) {
            gpu.stream(0).upload(metric, staged)?;
        }
        Ok(())
    }
}
