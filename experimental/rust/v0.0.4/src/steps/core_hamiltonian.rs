//! The core Hamiltonian H = T + V: both built on the host directly in the
//! solid harmonics, on rayon's threads, summed and uploaded to gpu0. A
//! host step: no library, no lock.

use fb_core_error_handling::{Result, Status};

use crate::budget::{Budget, DeviceBuf, Kind, Layout, Lease};
use crate::bus::{Held, Lane, MainBus, Shared};
use crate::flow::{Placement, Slot, Step, View};
use crate::steps::{GPU0, unpack_lower};

/// The step.
#[derive(Clone, Copy)]
pub struct CoreHamiltonianStep;

/// H on gpu0.
pub struct CoreHamiltonianOutput {
    h: DeviceBuf<f64>,
}

impl CoreHamiltonianOutput {
    /// H, `n_ao × n_ao`, symmetric.
    pub fn h(&self) -> &DeviceBuf<f64> {
        &self.h
    }
}

/// H's packed lower triangle on the host: the step's, and a test's
/// reference.
pub fn core_hamiltonian_host(shared: &Shared) -> Vec<f64> {
    let mut h = fb_t_kinetic_host::kinetic(&shared.basis);
    let charges = fb_v_potential_en_host::point_charges(&shared.molecule);
    let v = fb_v_potential_en_host::potential_en(&shared.basis, &charges);
    for (h, v) in h.iter_mut().zip(&v) {
        *h += v;
    }
    h
}

impl Step<MainBus> for CoreHamiltonianStep {
    const NAME: &'static str = "core_hamiltonian";
    type Output = CoreHamiltonianOutput;
    type Reads = ();

    fn slot(bus: &MainBus) -> &Slot<CoreHamiltonianOutput, Lane> {
        &bus.core_hamiltonian
    }

    fn placement(_: &Shared) -> Placement {
        Placement::Gpu0
    }

    fn mem_budget(shared: &Shared, budget: &Budget<'_>) -> Result<CoreHamiltonianOutput> {
        let n = shared.dims.n_ao;
        budget.scratch(GPU0, Kind::Pinned, n * n * size_of::<f64>());
        Ok(CoreHamiltonianOutput {
            h: budget.device(GPU0, n * n),
        })
    }

    fn run(
        out: &mut CoreHamiltonianOutput,
        view: &View<'_, MainBus, Self>,
        _: &mut Held<'_>,
        lane: &mut Lane,
        scratch: &Lease<'_>,
    ) -> Status {
        let shared = view.shared();
        let n = shared.dims.n_ao;
        let packed = core_hamiltonian_host(shared);
        let staged = scratch.pinned(GPU0, Layout::new().span::<f64>(n * n));
        unpack_lower(&packed, n, staged);
        lane.gpu0().stream(0).upload(&mut out.h, staged)
    }
}
