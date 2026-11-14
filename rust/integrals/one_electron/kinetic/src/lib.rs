//! The AO kinetic-energy matrix `T_mn = ⟨m| −½∇² |n⟩` as tasks of the model
//! (`docs/programming_model.md`, "A task declares"). The C++
//! `src/integrals/one_electron/kinetic` and the pipeline's spherical
//! transform (`MatrixAOPair::cartesian_to_spherical`) are the reference
//! design; `fb_integrals_one_electron_overlap` is the same shape for S.
//!
//! Two layers:
//!
//! - [`kinetic_classes`]: every class's [`kinetic_class`] launch on one
//!   stream, each adding into a Cartesian matrix the caller zeroed.
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
//! | `kinetic_class(stream, cls, n_ao, t)` (`kinetic_bridge.h`), one class | [`kinetic_class`], `fb-integrals-sys`'s safe launcher over a [`ClassRows`]; [`kinetic_classes`]`(stream, device, classes, t)`, every class in order on one stream |
//! | `kinetic(res, classes, t.cartesian())`: zeroed, then one launch per class, each on a stream forked from `res.stream()` | the `kinetic` task of [`add_kinetic`]: zeroed, then [`kinetic_classes`] on the worker's one stream, the records' [`ClassRows`] passed in by the caller |
//! | `kinetic`'s `InvalidConfig` for a class above f, `la > lb` or `n_ao²` past `INT32_MAX` | [`kinetic_class`]'s contract for the class, and its always-on `assert!` of the matrix's `n_cart²` and the `i32` bound; [`add_kinetic`] asserts the `i32` bound at planning too, and the records hold only `la <= lb` |
//! | `upload_spherical_transform_matrix`, `MatrixAOPair::cartesian_to_spherical` | [`fb_recipes_matrix_ao::add_cartesian_to_spherical`], U's slot passed in by the caller |
//! | `res.dev_idx()` made current for the call | nothing: a stream task runs with its worker's device current |
//!
//! [`BasisSet`]: fb_basis::BasisSet

#![forbid(unsafe_code)]

use contracts::debug_requires;
use fb_basis::BasisSet;
use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::device_slice::DeviceSliceMut;
use fb_core_wwr::memset;
use fb_core_wwr::runtime::Stream;
use fb_integrals_sys::{ClassRows, kinetic_class};
use fb_recipes_matrix_ao::add_cartesian_to_spherical;

/// Add every class's kinetic-energy contributions into the Cartesian matrix
/// `t`, `n_cart` × `n_cart`, on `stream`: one [`kinetic_class`] launch per
/// class, in order, each adding at `(m, n)` and `(n, m)`. Enqueues only,
/// and zeroes nothing: `t` must be zeroed (or hold what the caller adds T
/// to) before the call, and is symmetric after it when it was before.
///
/// `stream` is the calling task's, `device` its view, and every class's
/// fields were declared read by its layout ([`ClassRows::read`]).
///
/// # Errors
/// The first launch's refusal or launch error, as [`kinetic_class`] names
/// it with the class; the classes before it are enqueued, and `t` is then
/// unspecified. The kernels' own faults surface at the stream's next
/// synchronization.
///
/// # Panics
/// As [`kinetic_class`]: if `t` is not the records' `n_cart²` doubles, or
/// a class's field was not declared by the task's layout.
#[track_caller]
pub fn kinetic_classes(
    stream: &Stream,
    device: &Bufs<'_, Device>,
    classes: &[ClassRows],
    t: &mut impl DeviceSliceMut<f64>,
) -> Status {
    for c in classes {
        kinetic_class(stream, device, c, t)?;
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
/// If `nao_cart²` does not fit an `i32` (the kinetic kernels index the
/// matrix in 32 bits), or `classes` are rows of another worker's or another
/// plan's table ([`Dag::add`]).
#[debug_requires(basis.nshells() > 0, "a basis of at least one shell")]
#[track_caller]
pub fn add_kinetic(
    dag: &mut Dag<'_>,
    worker: usize,
    basis: &BasisSet,
    classes: &[ClassRows],
    u: &mut Option<RowId<Pinned, f64>>,
) -> RowId<Device, f64> {
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
        let mut t = device.take_mut(*t);
        memset::zero(&mut t, stream)?;
        kinetic_classes(stream, device, &self.classes, &mut t)
    }
}
