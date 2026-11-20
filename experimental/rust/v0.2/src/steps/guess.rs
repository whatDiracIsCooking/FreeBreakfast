//! The initial guess: the superposition of atomic densities, read from the
//! compiled-in stamps (or computed, for a basis that carries none) on the
//! host, and copied up. It needs the orbital basis and nothing else.

use fb_core_error_handling::Result;
use fb_io_sad::builtin_sad_density;
use fb_sad::SadStampLookup;

use crate::budget::{Budget, DeviceBuf, PinnedBuf};
use crate::bus::{Lane, MainBus, Shared};
use crate::device::upload;
use crate::flow::{Next, Placement, Slot, Step, View};
use crate::steps::GPU0;

/// The step.
#[derive(Clone, Copy)]
pub struct GuessStep;

/// The guess density D, and the pinned copy it is assembled in.
pub struct GuessOutput {
    d: DeviceBuf<f64>,
    staged: PinnedBuf<f64>,
}

impl GuessOutput {
    /// D, `n_ao × n_ao`.
    pub fn d(&self) -> &DeviceBuf<f64> {
        &self.d
    }
}

impl Step<MainBus> for GuessStep {
    const NAME: &'static str = "guess";
    type Output = GuessOutput;
    type Reads = ();

    fn slot(bus: &MainBus) -> &Slot<GuessOutput, Lane> {
        &bus.guess
    }

    fn placement(_: &Shared) -> Placement {
        Placement::Gpu(GPU0)
    }

    fn mem_budget(shared: &Shared, _: &Lane, budget: &Budget<'_>) -> Result<GuessOutput> {
        let n = shared.dims.n_ao;
        Ok(GuessOutput {
            d: budget.device(GPU0, n * n),
            staged: budget.pinned(GPU0, n * n),
        })
    }

    fn run(out: &mut GuessOutput, view: &View<'_, MainBus, Self>) -> Result<Next> {
        let shared = view.shared();
        // A stamp is taken only for a basis that was resolved by name.
        let baked: Option<SadStampLookup> =
            (!shared.basis.name().is_empty()).then_some(builtin_sad_density);
        // FINDING: the host SAD assembly returns a Vec, a large-ish
        // allocation inside `run` that no Budget saw; the old tree wrote it
        // straight into a pinned row.
        let density = fb_sad::sad_density(&shared.basis, shared.molecule.atoms(), baked)?;
        out.staged.copy_from_slice(&density);
        upload(&mut out.d, &out.staged, view.lane().gpu().stream())?;
        Ok(Next::Done)
    }
}
