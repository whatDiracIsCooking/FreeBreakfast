//! The initial guess: the superposition of atomic densities, read from the
//! compiled-in stamps (or computed, for a basis that carries none) on the
//! host, and copied up to gpu0. It needs the orbital basis and nothing
//! else. D is the `Output`; its pinned staging is scratch.

use fb_core_error_handling::{Result, Status};
use fb_io_sad::builtin_sad_density;
use fb_sad::SadStampLookup;

use crate::budget::{Budget, DeviceBuf, Kind, Layout, Lease};
use crate::bus::{MainBus, Shared};
use crate::device::{Lane, upload};
use crate::flow::{Placement, Slot, Step, View};
use crate::steps::GPU0;

/// The step.
#[derive(Clone, Copy)]
pub struct GuessStep;

/// The guess density D on gpu0.
pub struct GuessOutput {
    d: DeviceBuf<f64>,
}

impl GuessOutput {
    /// D, `n_ao × n_ao`, on gpu0.
    pub fn d(&self) -> &DeviceBuf<f64> {
        &self.d
    }
}

/// The SAD density of the run's molecule, on the host: the step's, and a
/// test's reference.
///
/// # Errors
/// The atomic SCF's, for a basis with no stamp.
pub fn sad_density(shared: &Shared) -> Result<Vec<f64>> {
    // A stamp is taken only for a basis that was resolved by name.
    let baked: Option<SadStampLookup> =
        (!shared.basis.name().is_empty()).then_some(builtin_sad_density);
    fb_sad::sad_density(&shared.basis, shared.molecule.atoms(), baked)
}

impl Step<MainBus> for GuessStep {
    const NAME: &'static str = "guess";
    type Output = GuessOutput;
    type Reads = ();

    fn slot(bus: &MainBus) -> &Slot<GuessOutput, Lane> {
        &bus.guess
    }

    fn placement(_: &Shared) -> Placement {
        Placement::Gpu0
    }

    fn mem_budget(shared: &Shared, budget: &Budget<'_>) -> Result<GuessOutput> {
        let n = shared.dims.n_ao;
        budget.scratch(GPU0, Kind::Pinned, n * n * size_of::<f64>());
        Ok(GuessOutput {
            d: budget.device(GPU0, n * n),
        })
    }

    fn run(
        out: &mut GuessOutput,
        view: &View<'_, MainBus, Self>,
        lane: &mut Lane,
        scratch: &Lease<'_>,
    ) -> Status {
        let n = view.shared().dims.n_ao;
        // The host SAD assembly returns a Vec, an n_ao² allocation inside
        // `run` that no Budget saw: within the small-host-scratch policy.
        let density = sad_density(view.shared())?;
        let staged = scratch.pinned(GPU0, Layout::new().span::<f64>(n * n));
        staged.copy_from_slice(&density);
        upload(&mut out.d, staged, lane.gpu0().stream())
    }
}
