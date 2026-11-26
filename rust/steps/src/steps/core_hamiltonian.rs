//! The core Hamiltonian H = T + V: both built on the host directly in the
//! solid harmonics, on rayon's threads, summed and uploaded to gpu0. A
//! host step: no library, no lock.

use fb_core_budget::{Budget, DeviceBuf, Kind, Lease};
use fb_core_error_handling::{Result, Status};
use fb_core_flow::{Held, Lane, Placement, Slot, Step, View};

use crate::bus::{MainBus, Shared};
use crate::steps::{GPU0, staging, unpack_lower, upload};

/// The step.
#[derive(Clone, Copy)]
pub struct CoreHamiltonianStep;

/// H on gpu0.
pub struct CoreHamiltonianOutput {
    h: DeviceBuf<f64>,
}

impl CoreHamiltonianOutput {
    /// H, `n_ao × n_ao`, symmetric.
    #[must_use]
    pub fn h(&self) -> &DeviceBuf<f64> {
        &self.h
    }
}

/// H's packed lower triangle on the host, T + V over the molecule's point
/// nuclei.
fn core_hamiltonian_host(shared: &Shared) -> Vec<f64> {
    let mut h = fb_integrals_t_kinetic_host::kinetic(&shared.basis);
    let charges = fb_integrals_v_potential_en_host::point_charges(&shared.molecule);
    let v = fb_integrals_v_potential_en_host::potential_en(&shared.basis, &charges);
    for (h, v) in h.iter_mut().zip(&v) {
        *h += v;
    }
    h
}

impl Step<MainBus> for CoreHamiltonianStep {
    const NAME: &'static str = "core_hamiltonian";
    type Output = CoreHamiltonianOutput;
    type Reads = ();

    fn slot(bus: &MainBus) -> &Slot<CoreHamiltonianOutput> {
        &bus.core_hamiltonian
    }

    fn placement(_: &Shared) -> Placement {
        Placement::Gpu0
    }

    fn mem_budget(shared: &Shared, budget: &Budget<'_>) -> Result<CoreHamiltonianOutput> {
        let n = shared.dims.n_ao;
        budget.scratch(GPU0, Kind::Pinned, staging(n).0.bytes());
        Ok(CoreHamiltonianOutput {
            h: budget.device(GPU0, n * n),
        })
    }

    fn run(
        out: &mut CoreHamiltonianOutput,
        view: &View<'_, MainBus, Self>,
        _: &mut Held<'_>,
        lane: &Lane,
        scratch: &Lease<'_>,
    ) -> Status {
        let shared = view.shared();
        let n = shared.dims.n_ao;
        let packed = core_hamiltonian_host(shared);
        let mut staged = scratch.pinned(GPU0, staging(n).1);
        unpack_lower(&packed, n, &mut staged);
        upload(lane, GPU0, &mut out.h, staged)
    }
}
