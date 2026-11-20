//! H = T + V, the spherical core Hamiltonian: the kinetic and the
//! nuclear-attraction class kernels added into one Cartesian matrix on the
//! step's own stream, the transform on the library's.

use fb_core_error_handling::Result;
use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut};
use fb_core_wwr::handle::DeviceHandleStream;
use fb_core_wwr::memset;
use fb_integrals_sys::{fb_kinetic_class, fb_nuclei_view, fb_potential_en_class};
use fb_molecule::nuclear_charge;
use fb_recipes_sys::mirror_triangle;

use crate::budget::{Budget, DeviceBuf, PinnedBuf};
use crate::bus::{Lane, MainBus, Shared};
use crate::device::{launched, upload};
use crate::flow::{Next, Placement, Slot, Step, View};
use crate::steps::harmonics::HarmonicsStep;
use crate::steps::records::RecordsStep;
use crate::steps::{GPU0, to_spherical};

/// The step.
#[derive(Clone, Copy)]
pub struct CoreHamiltonianStep;

/// H, the Cartesian matrix and half-transformed one behind it, and the
/// nuclei the attraction kernels read.
pub struct CoreHamiltonianOutput {
    h: DeviceBuf<f64>,
    cart: DeviceBuf<f64>,
    work: DeviceBuf<f64>,
    /// The charges (one per nucleus), then the positions (x y z per nucleus).
    nuclei: DeviceBuf<f64>,
    nuclei_host: PinnedBuf<f64>,
}

impl CoreHamiltonianOutput {
    /// H, `n_ao × n_ao`.
    pub fn h(&self) -> &DeviceBuf<f64> {
        &self.h
    }
}

impl Step<MainBus> for CoreHamiltonianStep {
    const NAME: &'static str = "core_hamiltonian";
    type Output = CoreHamiltonianOutput;
    type Reads = (RecordsStep, HarmonicsStep);

    fn slot(bus: &MainBus) -> &Slot<CoreHamiltonianOutput, Lane> {
        &bus.core_hamiltonian
    }

    fn placement(_: &Shared) -> Placement {
        Placement::Gpu(GPU0)
    }

    fn mem_budget(shared: &Shared, _: &Lane, budget: &Budget<'_>) -> Result<CoreHamiltonianOutput> {
        let d = shared.dims;
        Ok(CoreHamiltonianOutput {
            h: budget.device(GPU0, d.n_ao * d.n_ao),
            cart: budget.device(GPU0, d.n_cart * d.n_cart),
            work: budget.device(GPU0, d.n_cart * d.n_ao),
            nuclei: budget.device(GPU0, 4 * d.n_atom),
            nuclei_host: budget.pinned(GPU0, 4 * d.n_atom),
        })
    }

    fn run(out: &mut CoreHamiltonianOutput, view: &View<'_, MainBus, Self>) -> Result<Next> {
        let shared = view.shared();
        let d = shared.dims;
        let lane = view.lane().gpu();
        let stream = lane.stream();
        let records = view.read(RecordsStep);
        let harmonics = view.read(HarmonicsStep);

        for (a, atom) in shared.molecule.atoms().iter().enumerate() {
            out.nuclei_host[a] = f64::from(nuclear_charge(atom));
            let c = d.n_atom + 3 * a;
            out.nuclei_host[c] = atom.position.x;
            out.nuclei_host[c + 1] = atom.position.y;
            out.nuclei_host[c + 2] = atom.position.z;
        }
        upload(&mut out.nuclei, &out.nuclei_host, stream)?;
        let nuclei = fb_nuclei_view {
            n_nuc: d.n_atom as i32,
            z: out.nuclei.as_ptr(),
            c: out.nuclei.as_ptr().wrapping_add(d.n_atom),
        };

        memset::zero(&mut out.cart, stream)?;
        let n_cart = d.n_cart as i32;
        for class in records.classes() {
            let cls = class.view();
            let momenta = format!("({}, {})", class.la(), class.lb());
            // SAFETY: as the overlap step's launch: the records step's
            // class, and this step's n_cart² matrix.
            let status =
                unsafe { fb_kinetic_class(stream.stream(), &cls, n_cart, out.cart.as_mut_ptr()) };
            launched(&format!("fb_kinetic_class {momenta}"), status)?;
            // SAFETY: as above, and the nuclei are this step's upload,
            // n_atom charges and 3 n_atom coordinates, enqueued above.
            let status = unsafe {
                fb_potential_en_class(
                    stream.stream(),
                    &cls,
                    &nuclei,
                    n_cart,
                    out.cart.as_mut_ptr(),
                )
            };
            launched(&format!("fb_potential_en_class {momenta}"), status)?;
        }
        mirror_triangle(stream, false, d.n_cart, &mut out.cart, d.n_cart)?;
        lane.library(shared.library(GPU0), |library| {
            to_spherical(
                library.blas(),
                d.n_cart,
                d.n_ao,
                harmonics.u(),
                &out.cart,
                &mut out.work,
                &mut out.h,
            )
        })?;
        Ok(Next::Done)
    }
}
