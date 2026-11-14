//! The AO nuclear-attraction matrix
//! `V_mn = −Σ_C Z_C ⟨m| 1/|r − C| |n⟩` (point nuclei) as tasks of the model
//! (`docs/programming_model.md`, "A task declares"). The C++
//! `src/integrals/one_electron/potential_en` and the pipeline's spherical
//! transform (`MatrixAOPair::cartesian_to_spherical`) are the reference
//! design; `fb_integrals_one_electron_kinetic` is the same shape for T.
//!
//! Two layers:
//!
//! - [`potential_en_classes`], the safe layer over [`fb_potential_en_class`]:
//!   every class's launch on one stream, each adding into a Cartesian matrix
//!   the caller zeroed (or filled), the wrapper's [`fb_nuclei_view`] formed
//!   from the [`NucleiRows`]' device rows, with the launcher's requirements
//!   as `debug_requires`. It neither zeroes nor transforms, so a task that
//!   launches several families on one stream into one matrix (the core
//!   Hamiltonian's, T and then V on top) calls it after
//!   `fb_integrals_one_electron_kinetic::kinetic_classes`.
//! - [`add_nuclear_attraction`], V of a [`BasisSet`] and a molecule's nuclei
//!   added to a [`Dag`] after the records
//!   ([`fb_integrals_mmd::add_records`]) and the nuclei's upload
//!   ([`fb_molecule_nuclei::add_nuclei`]): **`potential en`**, on the
//!   worker's stream, the Cartesian matrix zeroed and then
//!   [`potential_en_classes`], reading each class's records and the nuclei
//!   as edges; then the spherical transform, `V = Uᵀ V_cart U`
//!   ([`fb_recipes_matrix_ao::add_cartesian_to_spherical`]), U's slot passed
//!   in so a plan stages U once. The records and the nuclei are passed in
//!   too, so one build and one upload serve every reader of the plan.
//!
//! V is not renormalized, as S and T are not: an AO's norm is its published
//! contraction's ([`fb_basis::ShellStamp::primitive_normalized`]).
//!
//! | C++ | Here |
//! |---|---|
//! | `potential_en_class(stream, cls, nuclei, n_ao, v)` (`potential_en_bridge.h`), one class | [`potential_en_classes`]`(stream, device, classes, nuclei, n_cart, v)`, every class in order on one stream, the `fb_nuclei_view` formed from the [`NucleiRows`], its requirements as `debug_requires` |
//! | `nuclear_attraction(res, classes, molecule, v)`: zeroed, then one launch per class, each on a stream forked from `res.stream()` | the `potential en` task of [`add_nuclear_attraction`]: zeroed, then [`potential_en_classes`] on the worker's one stream, the records' [`ClassRows`] passed in by the caller |
//! | `add_nuclear_attraction(res, classes, molecule, m)`, V added into `m` unzeroed | [`potential_en_classes`], called by the task that zeroed the matrix |
//! | `molecule.view()`, the `NucleiView` | [`NucleiRows`] from [`fb_molecule_nuclei::add_nuclei`], passed in; the view formed from its rows' pointers in [`potential_en_classes`] |
//! | `check_nuclei`'s `InvalidConfig` for nuclei on another device | [`add_nuclear_attraction`]'s `InvalidConfig` at planning for [`NucleiRows`] with no nucleus (the wrapper's `n_nuc >= 1`), adding nothing; nuclei of another worker are another table's rows, which [`Dag::add`] refuses with a panic |
//! | `check_one_electron_build`'s `InvalidConfig` for a class above f, `la > lb` or `n_ao²` past `INT32_MAX` | [`potential_en_classes`]' `debug_requires`; [`add_nuclear_attraction`] asserts the `i32` bound at planning, and the records hold only `la <= lb` |
//! | `upload_spherical_transform_matrix`, `MatrixAOPair::cartesian_to_spherical` | [`fb_recipes_matrix_ao::add_cartesian_to_spherical`], U's slot passed in by the caller |
//! | `res.dev_idx()` made current for the call | nothing: a stream task runs with its worker's device current |
//!
//! [`BasisSet`]: fb_basis::BasisSet

use contracts::debug_requires;
use fb_basis::BasisSet;
use fb_core_arena::{Bufs, Device, DeviceMut, Layout, Pinned, RowId};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::handle::DeviceHandleStream;
use fb_core_wwr::memset;
use fb_integrals_mmd::{ClassRows, launch};
use fb_integrals_sys::{FB_POTENTIAL_EN_MAX_MOMENTUM, fb_nuclei_view, fb_potential_en_class};
use fb_molecule_nuclei::NucleiRows;
use fb_recipes_matrix_ao::add_cartesian_to_spherical;

/// Add every class's nuclear-attraction contributions over `nuclei` into the
/// Cartesian matrix `v`, `n_cart` × `n_cart`, on `stream`: one
/// [`fb_potential_en_class`] launch per class, in order, each adding at
/// `(m, n)` and `(n, m)`. Enqueues only, and zeroes nothing: `v` must be
/// zeroed (or hold what the caller adds V to) before the call, and is
/// symmetric after it when it was before.
///
/// Over the C call, the types carry what its header asks in words: every
/// class's fields, the nuclei's two rows and `v` are rows of `device`'s
/// view, declared by the calling task's layout ([`ClassRows::read`],
/// [`NucleiRows::read`], an output for `v`), so none is null and none
/// overlaps another; `v` is the row's only writer while the call lasts.
/// Its contracts hold the bounds. Not checked: that the records
/// are of the basis `v` is the matrix of, since their AO offsets are device
/// data.
///
/// # Errors
/// The first launch's refusal or launch error, as
/// [`fb_integrals_mmd::launch`] names it with the class; the classes before
/// it are enqueued, and `v` is then unspecified. The kernels' own faults
/// surface at the stream's next synchronization.
///
/// # Panics
/// If a class's field or a nuclei row was not declared by the task's layout
/// ([`Bufs::get`]), or the nuclei's rows are not `n_nuc` and `3 n_nuc`
/// doubles long.
#[debug_requires(n_cart > 0, "a matrix of at least one AO")]
#[debug_requires(
    n_cart.checked_mul(n_cart).is_some_and(|n2| i32::try_from(n2).is_ok()),
    "n_cart² fits the kernels' 32-bit index"
)]
#[debug_requires(v.len() == n_cart * n_cart, "v is n_cart² doubles")]
#[debug_requires(nuclei.n_nuc > 0, "at least one nucleus")]
#[debug_requires(
    classes.iter().all(|c| 0 <= c.la && c.la <= c.lb && c.lb <= FB_POTENTIAL_EN_MAX_MOMENTUM),
    "every class has la <= lb <= FB_POTENTIAL_EN_MAX_MOMENTUM"
)]
#[debug_requires(classes.iter().all(|c| c.n_rec > 0), "every class has records")]
pub fn potential_en_classes(
    stream: &DeviceStream,
    device: &Bufs<'_, Device>,
    classes: &[ClassRows],
    nuclei: &NucleiRows,
    n_cart: usize,
    v: &DeviceMut<'_, f64>,
) -> Status {
    let (z, c) = (device.get(nuclei.z), device.get(nuclei.c));
    let n_nuc = nuclei.n_nuc as usize;
    // The kernels read n_nuc charges and 3 n_nuc coordinates through the
    // view's pointers, so these lengths are what the launch trusts: always
    // on (docs/testing.md, "Contracts are part of the spec").
    assert!(
        z.len() == n_nuc && c.len() == 3 * n_nuc,
        "the nuclei's rows are n_nuc and 3 n_nuc long"
    );
    let view = fb_nuclei_view {
        n_nuc: nuclei.n_nuc,
        z: z.as_ptr(),
        c: c.as_ptr(),
    };
    for cls in classes {
        let records = cls.view(device);
        // SAFETY: the records' arrays are the class's n_rec-long device rows,
        // and the nuclei's the n_nuc- and 3 n_nuc-long rows, all declared read
        // by the calling task, which the plan orders after the tasks that
        // write them; `v` is n_cart² doubles on the device, the task's own
        // output, and every offset the records hold is a Cartesian AO of their
        // basis, the one `v` is the matrix of. Both views are host structs
        // that outlive the call, which only enqueues.
        launch("fb_potential_en_class", cls, unsafe {
            fb_potential_en_class(
                stream.stream(),
                &records,
                &view,
                n_cart as i32,
                v.as_mut_ptr(),
            )
        })?;
    }
    Ok(())
}

/// Add the spherical AO nuclear-attraction matrix of `basis` over `nuclei`
/// to `dag` on GPU worker `worker`, and return V: an output row of
/// `nao_sphe²` doubles, column-major, symmetric up to rounding.
///
/// `potential en` (stream: the Cartesian matrix zeroed and
/// [`potential_en_classes`], reading the records and the nuclei as edges),
/// then the spherical transform of
/// [`add_cartesian_to_spherical`]: `harmonics` (CPU: U into a pinned row),
/// only when `u` is `None`, and `spherical transform` (stream:
/// `V = Uᵀ V_cart U`). `nuclei` are the molecule's, uploaded on `worker` by
/// [`fb_molecule_nuclei::add_nuclei`] earlier in `dag`, the molecule the
/// basis is placed on; `classes` are the records of `basis`, added on
/// `worker` by [`fb_integrals_mmd::add_records`] earlier in `dag`, so one
/// build serves every reader of the plan; `u` is the plan's slot for U over
/// `basis` on `worker`, shared with every other matrix the plan transforms.
///
/// # Errors
/// `InvalidConfig` for `nuclei` with no nucleus (`n_nuc` not positive: the
/// launch wrapper's refusal, at planning), before anything is added.
///
/// # Panics
/// If `nao_cart²` does not fit an `i32` (the kernels index the matrix in
/// 32 bits), or `classes` or `nuclei` are rows of another worker's or
/// another plan's table ([`Dag::add`]).
#[debug_requires(basis.nshells() > 0, "a basis of at least one shell")]
#[track_caller]
pub fn add_nuclear_attraction(
    dag: &mut Dag<'_>,
    worker: usize,
    basis: &BasisSet,
    classes: &[ClassRows],
    nuclei: NucleiRows,
    u: &mut Option<RowId<Pinned, f64>>,
) -> Result<RowId<Device, f64>> {
    if nuclei.n_nuc < 1 {
        return Err(Error::config(format!(
            "add_nuclear_attraction: {} nuclei; the kernels need at least one",
            nuclei.n_nuc
        )));
    }
    let n_cart = basis.nao_cart() as usize;
    assert!(
        i32::try_from(n_cart * n_cart).is_ok(),
        "add_nuclear_attraction: {n_cart}² Cartesian AOs do not fit the kernels' 32-bit index"
    );
    let v_cart = dag.add(
        "potential en",
        worker,
        PotentialEn {
            classes: classes.to_vec(),
            nuclei,
            n_cart,
        },
    );
    Ok(add_cartesian_to_spherical(
        dag, worker, basis, u, "V", v_cart,
    ))
}

/// GPU: the Cartesian nuclear-attraction matrix, `nao_cart²`, zeroed and
/// then summed class by class ([`potential_en_classes`]).
struct PotentialEn {
    classes: Vec<ClassRows>,
    nuclei: NucleiRows,
    n_cart: usize,
}

impl Task for PotentialEn {
    type Rows = RowId<Device, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        for c in &self.classes {
            c.read(device);
        }
        self.nuclei.read(device);
        device.output("V cartesian", self.n_cart * self.n_cart)
    }

    fn run(
        &self,
        v: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let mut v = device.take_mut(*v);
        memset::zero(&mut v, stream)?;
        potential_en_classes(stream, device, &self.classes, &self.nuclei, self.n_cart, &v)
    }
}
