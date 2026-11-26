//! The core Hamiltonian H = T + V: both built on the host directly in the
//! solid harmonics, in the host integrals' pool, side by side in the
//! step's pinned staging, summed and uploaded to gpu0. A
//! host step: no library, no lock.

use fb_core_budget::{Budget, DeviceBuf, Kind, Lease};
use fb_core_error_handling::{Result, Status};
use fb_core_flow::{Held, Lane, Placement, Slot, Step, View};

use crate::bus::{MainBus, Shared};
use crate::steps::{GPU0, staging, unpack_lower_in_place, upload};

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

/// The staging's length: T's packed triangle and V's after it, two of
/// `n (n + 1) / 2`, which is `n` past the `n × n` H written over them.
fn staging_len(n: usize) -> usize {
    n * (n + 1)
}

/// H's packed lower triangle at the front of `staged`, T + V over the
/// molecule's point nuclei, V built behind it.
fn core_hamiltonian_host(shared: &Shared, staged: &mut [f64]) {
    let n = shared.dims.n_ao;
    let (t, v) = staged.split_at_mut(n * (n + 1) / 2);
    let charges = fb_integrals_v_potential_en_host::point_charges(&shared.molecule);
    shared.host_pool().install(|| {
        fb_integrals_t_kinetic_host::kinetic_into(&shared.basis, t);
        fb_integrals_v_potential_en_host::potential_en_into(&shared.basis, &charges, v);
    });
    for (t, v) in t.iter_mut().zip(&*v) {
        *t += v;
    }
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
        budget.scratch(GPU0, Kind::Pinned, staging(staging_len(n)).0.bytes());
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
        let mut staged = scratch.pinned(GPU0, staging(staging_len(n)).1);
        core_hamiltonian_host(shared, &mut staged);
        unpack_lower_in_place(n, &mut staged);
        upload(lane, std::slice::from_mut(&mut out.h), staged)
    }
}
