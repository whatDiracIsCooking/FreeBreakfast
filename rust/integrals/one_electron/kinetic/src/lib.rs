//! The AO kinetic-energy matrix `T_mn = ⟨m| −½∇² |n⟩` as tasks of the model
//! (`docs/programming_model.md`, "A task declares"). The C++
//! `src/integrals/one_electron/kinetic` and the pipeline's spherical
//! transform (`MatrixAOPair::cartesian_to_spherical`) are the reference
//! design; `fb_integrals_one_electron_overlap` is the same shape for S.
//!
//! Two layers:
//!
//! - [`kinetic_classes`], the safe layer over [`fb_kinetic_class`]: every
//!   class's launch on one stream, each adding into a Cartesian matrix the
//!   caller zeroed, with the launcher's requirements as `debug_requires`.
//!   It neither zeroes nor transforms, so a task that launches several
//!   families on one stream into one matrix (the core Hamiltonian's, T and
//!   then V on top) calls it between its own launches.
//! - [`add_kinetic`], T of a [`BasisSet`] added to a
//!   [`Dag`] after the records
//!   ([`fb_integrals_mmd::add_records`]), which the caller adds and passes
//!   in so one build serves every reader of a plan: **`kinetic`**, on the worker's
//!   stream, the Cartesian matrix zeroed and then [`kinetic_classes`],
//!   reading each class's records as an edge; then the spherical transform,
//!   `T = Uᵀ T_cart U` ([`fb_recipes_matrix_ao::add_cartesian_to_spherical`]),
//!   U's slot passed in so a plan stages U once.
//!
//! T is not renormalized, as S is not: an AO's norm is its published
//! contraction's ([`fb_basis::ShellStamp::primitive_normalized`]).
//!
//! | C++ | Here |
//! |---|---|
//! | `kinetic_class(stream, cls, n_ao, t)` (`kinetic_bridge.h`), one class | [`kinetic_classes`]`(stream, device, classes, n_cart, t)`, every class in order on one stream, its requirements as `debug_requires` |
//! | `kinetic(res, classes, t.cartesian())`: zeroed, then one launch per class, each on a stream forked from `res.stream()` | the `kinetic` task of [`add_kinetic`]: zeroed, then [`kinetic_classes`] on the worker's one stream, the records' [`ClassRows`] passed in by the caller |
//! | `kinetic`'s `InvalidConfig` for a class above f, `la > lb` or `n_ao²` past `INT32_MAX` | [`kinetic_classes`]' `debug_requires`; [`add_kinetic`] asserts the `i32` bound at planning, and the records hold only `la <= lb` |
//! | `upload_spherical_transform_matrix`, `MatrixAOPair::cartesian_to_spherical` | [`fb_recipes_matrix_ao::add_cartesian_to_spherical`], U's slot passed in by the caller |
//! | `res.dev_idx()` made current for the call | nothing: a stream task runs with its worker's device current |
//!
//! [`BasisSet`]: fb_basis::BasisSet

use contracts::debug_requires;
use fb_basis::BasisSet;
use fb_core_arena::{Bufs, Device, DeviceMut, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::gpu_status;
use fb_core_wwr::handle::DeviceHandleStream;
use fb_integrals_mmd::{ClassRows, launch};
use fb_integrals_sys::{FB_KINETIC_MAX_MOMENTUM, fb_kinetic_class};
use fb_recipes_matrix_ao::add_cartesian_to_spherical;

/// Add every class's kinetic-energy contributions into the Cartesian matrix
/// `t`, `n_cart` × `n_cart`, on `stream`: one [`fb_kinetic_class`] launch
/// per class, in order, each adding at `(m, n)` and `(n, m)`. Enqueues
/// only, and zeroes nothing: `t` must be zeroed (or hold what the caller
/// adds T to) before the call, and is symmetric after it when it was
/// before.
///
/// Over the C call, the types carry what its header asks in words: every
/// class's fields and `t` are rows of `device`'s view, declared by the
/// calling task's layout ([`ClassRows::read`], an output for `t`), so none
/// is null and none overlaps another; `t` is the row's only writer while
/// the call lasts. Checked in debug builds: the bounds below. Not checked:
/// that the records are of the basis `t` is the matrix of, since their AO
/// offsets are device data.
///
/// # Errors
/// The first launch's refusal or launch error, as
/// [`fb_integrals_mmd::launch`] names it with the class; the classes before
/// it are enqueued, and `t` is then unspecified. The kernels' own faults
/// surface at the stream's next synchronization.
///
/// # Panics
/// If a class's field was not declared by the task's layout ([`Bufs::get`]).
#[debug_requires(n_cart > 0, "a matrix of at least one AO")]
#[debug_requires(
    n_cart.checked_mul(n_cart).is_some_and(|n2| i32::try_from(n2).is_ok()),
    "n_cart² fits the kernels' 32-bit index"
)]
#[debug_requires(t.len() == n_cart * n_cart, "t is n_cart² doubles")]
#[debug_requires(
    classes.iter().all(|c| 0 <= c.la && c.la <= c.lb && c.lb <= FB_KINETIC_MAX_MOMENTUM),
    "every class has la <= lb <= FB_KINETIC_MAX_MOMENTUM"
)]
#[debug_requires(classes.iter().all(|c| c.n_rec > 0), "every class has records")]
pub fn kinetic_classes(
    stream: &DeviceStream,
    device: &Bufs<'_, Device>,
    classes: &[ClassRows],
    n_cart: usize,
    t: &DeviceMut<'_, f64>,
) -> Status {
    for c in classes {
        let view = c.view(device);
        // SAFETY: the view's arrays are the class's n_rec-long device rows,
        // declared read by the calling task, which the plan orders after the
        // records task that writes them; `t` is n_cart² doubles on the
        // device, the task's own output, and every offset the records hold
        // is a Cartesian AO of their basis, the one `t` is the matrix of.
        launch("fb_kinetic_class", c, unsafe {
            fb_kinetic_class(stream.stream(), &view, n_cart as i32, t.as_mut_ptr())
        })?;
    }
    Ok(())
}

/// Add the spherical AO kinetic-energy matrix of `basis` to `dag` on GPU
/// worker `worker`, and return T: an output row of `nao_sphe²` doubles,
/// column-major, symmetric up to rounding.
///
/// `kinetic` (stream: the Cartesian matrix zeroed and [`kinetic_classes`],
/// reading the records as an edge), then the spherical transform of
/// [`add_cartesian_to_spherical`]: `harmonics` (CPU: U into a pinned row),
/// only when `u` is `None`, and `spherical transform` (stream:
/// `T = Uᵀ T_cart U`). `classes` are the records of `basis`, added on
/// `worker` by [`fb_integrals_mmd::add_records`] earlier in `dag`, so one
/// build serves every reader of the plan; `u` is the plan's slot for U over
/// `basis` on `worker`, shared with every other matrix the plan transforms.
///
/// # Panics
/// If `basis` has no shells, or `nao_cart²` does not fit an `i32` (the
/// kinetic kernels index the matrix in 32 bits), or `classes` are rows of
/// another worker's or another plan's table ([`Dag::add`]).
#[track_caller]
pub fn add_kinetic(
    dag: &mut Dag<'_>,
    worker: usize,
    basis: &BasisSet,
    classes: &[ClassRows],
    u: &mut Option<RowId<Pinned, f64>>,
) -> RowId<Device, f64> {
    assert!(basis.nshells() > 0, "add_kinetic: a basis with no shells");
    let n_cart = basis.nao_cart() as usize;
    assert!(
        i32::try_from(n_cart * n_cart).is_ok(),
        "add_kinetic: {n_cart}² Cartesian AOs do not fit the kernels' 32-bit index"
    );
    let t_cart = dag.add(
        "kinetic",
        worker,
        Kinetic {
            classes: classes.to_vec(),
            n_cart,
        },
    );
    add_cartesian_to_spherical(dag, worker, basis, u, "T", t_cart)
}

/// GPU: the Cartesian kinetic-energy matrix, `nao_cart²`, zeroed and then
/// summed class by class ([`kinetic_classes`]).
struct Kinetic {
    classes: Vec<ClassRows>,
    n_cart: usize,
}

impl Task for Kinetic {
    type Rows = RowId<Device, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        for c in &self.classes {
            c.read(device);
        }
        device.output("T cartesian", self.n_cart * self.n_cart)
    }

    fn run(
        &self,
        t: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let t = device.take_mut(*t);
        // SAFETY: `t` is a device row of `t.len()` doubles, on the task's
        // stream.
        gpu_status(unsafe {
            wwr_sys::wwr_memset_async(
                t.as_mut_ptr().cast(),
                0,
                t.len() * size_of::<f64>(),
                stream.stream(),
            )
        })?;
        kinetic_classes(stream, device, &self.classes, self.n_cart, &t)
    }
}
