//! The AO core Hamiltonian `H = T + V`, the kinetic energy and the
//! nuclear attraction of point nuclei, as tasks of the model
//! (`docs/programming_model.md`, "A task declares"). The C++
//! `src/integrals/one_electron/core_hamiltonian` and the pipeline's
//! spherical transform (`MatrixAOPair::cartesian_to_spherical`) are the
//! reference design.
//!
//! [`add_core_hamiltonian`] adds, after the records
//! ([`fb_integrals_mmd::add_records`]) and the nuclei's upload
//! ([`fb_molecule_nuclei::add_nuclei`]), both passed in so one build and
//! one upload serve every reader of a plan, one build task,
//! **`core hamiltonian`**, on the worker's stream: the Cartesian matrix
//! zeroed, every class's kinetic launch
//! ([`fb_integrals_one_electron_kinetic::kinetic_classes`]) and then every
//! class's nuclear-attraction launch on top
//! ([`fb_integrals_one_electron_potential_en::potential_en_classes`]). A
//! chain of launches on one stream is one task (`docs/programming_model.md`,
//! "Open questions"): both kernels only add, so H is formed in place, with
//! no T or V matrix of its own and no separate addition. Then the shared
//! spherical transform, `H = Uᵀ H_cart U`
//! ([`fb_recipes_matrix_ao::add_cartesian_to_spherical`]), U's slot passed
//! in so a plan stages U once.
//!
//! H is not renormalized, as T and V are not: an AO's norm is its published
//! contraction's ([`fb_basis::ShellStamp::primitive_normalized`]).
//!
//! | C++ | Here |
//! |---|---|
//! | `core_hamiltonian(res, classes, molecule, h.cartesian())`: `kinetic` (zeroed, then one launch per class) and `add_nuclear_attraction` on top, each class on a stream forked from `res.stream()` and joined back | the `core hamiltonian` task of [`add_core_hamiltonian`]: zeroed, then [`kinetic_classes`] and [`potential_en_classes`] on the worker's one stream, so every V launch follows every T launch by stream order; the records' [`ClassRows`] passed in by the caller |
//! | `check_one_electron_build`'s `InvalidConfig` for a class above f, `la > lb` or `n_ao²` past `INT32_MAX` | the two safe layers' `debug_requires`; [`add_core_hamiltonian`] asserts the `i32` bound at planning, and the records hold only `la <= lb` |
//! | `check_nuclei`'s `InvalidConfig`, before anything launches | [`add_core_hamiltonian`]'s `InvalidConfig` at planning for [`NucleiRows`] with no nucleus, adding nothing; nuclei of another worker are another table's rows, which [`Dag::add`] refuses with a panic |
//! | `molecule.view()`, the `NucleiView` | [`NucleiRows`] from [`fb_molecule_nuclei::add_nuclei`], passed in, so one upload serves every reader of the plan |
//! | `upload_spherical_transform_matrix`, `MatrixAOPair::cartesian_to_spherical` | [`fb_recipes_matrix_ao::add_cartesian_to_spherical`], U's slot passed in by the caller |
//! | `res.dev_idx()` made current for the call | nothing: a stream task runs with its worker's device current |
//!
//! [`kinetic_classes`]: fb_integrals_one_electron_kinetic::kinetic_classes
//! [`potential_en_classes`]: fb_integrals_one_electron_potential_en::potential_en_classes

use contracts::debug_requires;
use fb_basis::BasisSet;
use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::gpu_status;
use fb_core_wwr::handle::DeviceHandleStream;
use fb_integrals_mmd::ClassRows;
use fb_integrals_one_electron_kinetic::kinetic_classes;
use fb_integrals_one_electron_potential_en::potential_en_classes;
use fb_molecule_nuclei::NucleiRows;
use fb_recipes_matrix_ao::add_cartesian_to_spherical;

/// Add the spherical AO core Hamiltonian of `basis` over `nuclei` to `dag`
/// on GPU worker `worker`, and return H: an output row of `nao_sphe²`
/// doubles, column-major, symmetric up to rounding.
///
/// `core hamiltonian` (stream: the Cartesian matrix zeroed, [`kinetic_classes`] and then
/// [`potential_en_classes`] into it, reading the records and the nuclei as
/// edges), then the spherical transform of [`add_cartesian_to_spherical`]:
/// `harmonics` (CPU: U into a pinned row), only when `u` is `None`, and
/// `spherical transform` (stream: `H = Uᵀ H_cart U`). `nuclei` are the
/// molecule's, uploaded on `worker` by [`fb_molecule_nuclei::add_nuclei`]
/// earlier in `dag`, the molecule the basis is placed on; `classes` are the
/// records of `basis`, added on `worker` by
/// [`fb_integrals_mmd::add_records`] earlier in `dag`, so one build serves
/// every reader of the plan; `u` is the plan's slot for U over `basis` on
/// `worker`, shared with every other matrix the plan transforms.
///
/// # Errors
/// `InvalidConfig` for `nuclei` with no nucleus (`n_nuc` not positive: the
/// nuclear-attraction wrapper's refusal, at planning), before anything is
/// added.
///
/// Requires, checked in debug builds: `basis` has at least one shell.
///
/// # Panics
/// If `nao_cart²` does not fit an `i32` (the kernels index the matrix in
/// 32 bits), or `classes` or `nuclei` are rows of another worker's or
/// another plan's table ([`Dag::add`]).
///
/// [`kinetic_classes`]: fb_integrals_one_electron_kinetic::kinetic_classes
/// [`potential_en_classes`]: fb_integrals_one_electron_potential_en::potential_en_classes
#[debug_requires(basis.nshells() > 0, "a basis of at least one shell")]
#[track_caller]
pub fn add_core_hamiltonian(
    dag: &mut Dag<'_>,
    worker: usize,
    basis: &BasisSet,
    classes: &[ClassRows],
    nuclei: NucleiRows,
    u: &mut Option<RowId<Pinned, f64>>,
) -> Result<RowId<Device, f64>> {
    if nuclei.n_nuc < 1 {
        return Err(Error::config(format!(
            "add_core_hamiltonian: {} nuclei; the nuclear-attraction kernels need at least one",
            nuclei.n_nuc
        )));
    }
    let n_cart = basis.nao_cart() as usize;
    assert!(
        i32::try_from(n_cart * n_cart).is_ok(),
        "add_core_hamiltonian: {n_cart}² Cartesian AOs do not fit the kernels' 32-bit index"
    );
    let h_cart = dag.add(
        "core hamiltonian",
        worker,
        CoreHamiltonian {
            classes: classes.to_vec(),
            nuclei,
            n_cart,
        },
    );
    Ok(add_cartesian_to_spherical(
        dag, worker, basis, u, "H", h_cart,
    ))
}

/// GPU: the Cartesian core Hamiltonian, `nao_cart²`, zeroed, then T summed
/// class by class ([`kinetic_classes`]) and V on top
/// ([`potential_en_classes`]).
struct CoreHamiltonian {
    classes: Vec<ClassRows>,
    nuclei: NucleiRows,
    n_cart: usize,
}

impl Task for CoreHamiltonian {
    type Rows = RowId<Device, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        for c in &self.classes {
            c.read(device);
        }
        self.nuclei.read(device);
        device.output("H cartesian", self.n_cart * self.n_cart)
    }

    fn run(
        &self,
        h: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let h = device.take_mut(*h);
        // SAFETY: `h` is a device row of `h.len()` doubles, on the task's
        // stream.
        gpu_status(unsafe {
            wwr_sys::wwr_memset_async(
                h.as_mut_ptr().cast(),
                0,
                h.len() * size_of::<f64>(),
                stream.stream(),
            )
        })?;
        // Both on the one stream: every V launch follows every T launch, and
        // each adds into `h` in turn.
        kinetic_classes(stream, device, &self.classes, self.n_cart, &h)?;
        potential_en_classes(stream, device, &self.classes, &self.nuclei, self.n_cart, &h)
    }
}
