//! The initial guess: the superposition of atomic densities, read from the
//! compiled-in stamps (or computed, for a basis that carries none), or the
//! density `[guess] from` names, read from its `.npy`, on the host into
//! the step's pinned staging, and uploaded to gpu0. A host step: no
//! library, no lock. Either way the staging is the same `n_ao²` doubles,
//! so `from` leaves the budget as it was; setup has checked the file's
//! shape, and the read checks it again.

use fb_core_budget::{Budget, DeviceBuf, Kind, Lease};
use fb_core_error_handling::{Result, Status};
use fb_core_flow::{Held, Lane, Placement, Slot, Step, View};
use fb_io_sad::builtin_sad_density;
use fb_sad::SadStampLookup;

use crate::bus::{MainBus, Shared};
use crate::steps::{GPU0, staging, upload};

/// The step.
#[derive(Clone, Copy)]
pub struct GuessStep;

/// The guess density D on gpu0.
pub struct GuessOutput {
    d: DeviceBuf<f64>,
}

impl GuessOutput {
    /// D, `n_ao × n_ao`, on gpu0.
    #[must_use]
    pub fn d(&self) -> &DeviceBuf<f64> {
        &self.d
    }
}

/// The SAD density of the run's molecule, on the host, into `density`. A
/// stamp is taken only for a basis resolved by name.
fn sad_density(shared: &Shared, density: &mut [f64]) -> Status {
    let baked: Option<SadStampLookup> =
        (!shared.basis.name().is_empty()).then_some(builtin_sad_density);
    fb_sad::sad_density_into(&shared.basis, shared.molecule.atoms(), baked, density)
}

impl Step<MainBus> for GuessStep {
    const NAME: &'static str = "guess";
    type Output = GuessOutput;
    type Reads = ();

    fn slot(bus: &MainBus) -> &Slot<GuessOutput> {
        &bus.guess
    }

    fn placement(_: &Shared) -> Placement {
        Placement::Gpu0
    }

    fn mem_budget(shared: &Shared, budget: &Budget<'_>) -> Result<GuessOutput> {
        let n = shared.dims.n_ao;
        budget.scratch(GPU0, Kind::Pinned, staging(n * n).0.bytes());
        Ok(GuessOutput {
            d: budget.device(GPU0, n * n),
        })
    }

    fn run(
        out: &mut GuessOutput,
        view: &View<'_, MainBus, Self>,
        _: &mut Held<'_>,
        lane: &Lane,
        scratch: &Lease<'_>,
    ) -> Status {
        let shared = view.shared();
        let n = shared.dims.n_ao;
        let mut staged = scratch.pinned(GPU0, staging(n * n).1);
        match shared.config.guess.as_ref().and_then(|g| g.from.as_deref()) {
            Some(from) => fb_io_npy::read_npy_into(from, &[n, n], &mut staged)?,
            None => sad_density(shared, &mut staged)?,
        }
        upload(lane, std::slice::from_mut(&mut out.d), staged)
    }
}
