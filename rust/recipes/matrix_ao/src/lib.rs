//! The Cartesian-to-spherical transform of a square AO matrix,
//! `M_sph = Uᵀ M_cart U`, as tasks of a [`Dag`]: every one-electron matrix
//! is built Cartesian (the kernels evaluate Cartesian Gaussians) and means
//! the basis's real solid harmonics. The C++ `src/recipes/matrix_ao` is the
//! reference design.
//!
//! [`add_cartesian_to_spherical`] adds, for one Cartesian matrix row:
//!
//! - **`harmonics`**, on the CPU, once per plan and basis: U, the
//!   Cartesian-to-spherical transform
//!   ([`fb_basis::spherical_transform_matrix`]), written into a pinned row
//!   that every later transform over the basis reads.
//! - **`spherical transform`**, on the worker's stream: U copied into device
//!   scratch, then `M = Uᵀ M_cart U` as two GEMMs through `wwr-sys`'s BLAS
//!   on the worker's handle, the intermediate `M_cart U` scratch too.
//!
//! | C++ | Here |
//! |---|---|
//! | `upload_spherical_transform_matrix(res, basis)` (nothing for an identity U), shared by every pair over the basis | the `harmonics` task's pinned row, added by the first call and handed to the later ones through the caller's slot, always |
//! | `MatrixAOPair`'s U and own scratch, so two pairs may run on different streams | each transform's own device copy of U and its own `M_cart U` scratch |
//! | `MatrixAOPair::cartesian_to_spherical` (`gemmwich`; a copy for an identity U) | the `spherical transform` task's two GEMMs, always |
//! | `res.blas()`, the `DeviceResources`' handle | the stream worker's handle, [`DeviceStream::blas`], made with the backend and bound to the worker's stream |
//! | `MatrixAO<L>`, `BasisLayout`, `ao_count` | nothing: a matrix is a device row, its order the basis's `nao_cart` or `nao_sphe` |
//! | `MatrixAOPair::spherical_to_cartesian` | not ported yet |
//!
//! [`add_blas_workspace`] gives a worker's BLAS handle its workspace from
//! the arena, a worker row, rather than the one the library allocates for
//! itself: added first by a plan that runs these GEMMs, or any other BLAS.

use contracts::debug_requires;
use fb_basis::BasisSet;
use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_h2p, memcpy_p2d};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::gpu_status;
use wwr_sys::wwr_blas_operation;

mod workspace;
pub use workspace::add_blas_workspace;

/// Add the spherical form of the Cartesian AO matrix `cartesian` of `basis`
/// to `dag` on GPU worker `worker`, and return it: an output row of
/// `nao_sphe²` doubles, column-major, named `name`.
///
/// `cartesian` is an earlier task's output of `nao_cart²` doubles,
/// column-major. The task added is `spherical transform` (stream: U copied
/// into device scratch `U`, then `M = Uᵀ M_cart U` as two GEMMs, the
/// intermediate `M_cart U` scratch `"{name} U"`). Before it, when `u` is
/// `None`, `harmonics` (CPU: U, [`fb_basis::spherical_transform_matrix`],
/// written into a pinned row), whose row is then left in `u`: pass the same
/// slot for every matrix of the plan over `basis` on `worker`, and U is
/// staged once and read by every transform.
///
/// Requires, checked in debug builds: `basis` has at least one shell.
///
/// # Panics
/// If `u` holds a row of another worker's table ([`Dag::add`]).
#[debug_requires(basis.nshells() > 0, "a basis of at least one shell")]
#[track_caller]
pub fn add_cartesian_to_spherical(
    dag: &mut Dag<'_>,
    worker: usize,
    basis: &BasisSet,
    u: &mut Option<RowId<Pinned, f64>>,
    name: &str,
    cartesian: RowId<Device, f64>,
) -> RowId<Device, f64> {
    let u = *u.get_or_insert_with(|| {
        dag.add(
            "harmonics",
            worker,
            Harmonics {
                u: fb_basis::spherical_transform_matrix(basis),
            },
        )
    });
    dag.add(
        "spherical transform",
        worker,
        Transform {
            m_cart: cartesian,
            u,
            n_cart: basis.nao_cart() as usize,
            n_sph: basis.nao_sphe() as usize,
            name: name.to_owned(),
        },
    )
    .m
}

/// CPU: U written into a pinned row, for the transform to copy up.
struct Harmonics {
    /// `nao_cart` × `nao_sphe`, column-major.
    u: Vec<f64>,
}

impl Task for Harmonics {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        host.output("U", self.u.len())
    }

    fn run(
        &self,
        u: &Self::Rows,
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        memcpy_h2p(host, *u, &self.u);
        Ok(())
    }
}

/// GPU: `M = Uᵀ M_cart U`, U copied from its pinned row into device
/// scratch, then `M_cart U` into scratch and `Uᵀ (M_cart U)` into M, two
/// GEMMs on the worker's stream, as the C++ `gemmwich` orders them.
struct Transform {
    m_cart: RowId<Device, f64>,
    u: RowId<Pinned, f64>,
    n_cart: usize,
    n_sph: usize,
    /// M's row name; the intermediate's is `"{name} U"`.
    name: String,
}

#[derive(Clone, Copy)]
struct TransformRows {
    u: RowId<Device, f64>,
    work: RowId<Device, f64>,
    m: RowId<Device, f64>,
}

impl Task for Transform {
    type Rows = TransformRows;
    const GPU: bool = true;

    fn layout(
        &self,
        device: &mut Layout<'_, Device>,
        host: &mut Layout<'_, Pinned>,
    ) -> TransformRows {
        host.read(self.u);
        device.read(self.m_cart);
        let n_u = self.n_cart * self.n_sph;
        TransformRows {
            u: device.scratch("U", n_u),
            work: device.scratch(&format!("{} U", self.name), n_u),
            m: device.output(&self.name, self.n_sph * self.n_sph),
        }
    }

    fn run(
        &self,
        rows: &TransformRows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream: worker, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let u = device.take_mut(rows.u);
        memcpy_p2d(&u, host, self.u, worker)?;

        // The worker's handle, bound to `stream` when the backend was made.
        let handle = worker.blas().get();

        let (n_cart, n_sph) = (self.n_cart as i32, self.n_sph as i32);
        let (one, zero) = (1.0_f64, 0.0_f64);
        let m_cart = device.get(self.m_cart);
        let work = device.take_mut(rows.work);
        let m = device.take_mut(rows.m);
        use wwr_blas_operation::{WWR_BLAS_OP_N, WWR_BLAS_OP_T};
        // work = M_cart U: (n_cart × n_cart)(n_cart × n_sph).
        // SAFETY: every matrix is a device row of the declared size, each
        // ld its row count; the scalars are host memory, the handle's
        // default pointer mode; the GEMM enqueues on `stream`.
        gpu_status(unsafe {
            wwr_sys::wwr_blas_dgemm(
                handle,
                WWR_BLAS_OP_N,
                WWR_BLAS_OP_N,
                n_cart,
                n_sph,
                n_cart,
                &one,
                m_cart.as_ptr(),
                n_cart,
                u.as_ptr(),
                n_cart,
                &zero,
                work.as_mut_ptr(),
                n_cart,
            )
        })?;
        // M = Uᵀ work: (n_sph × n_cart)(n_cart × n_sph).
        // SAFETY: as above.
        gpu_status(unsafe {
            wwr_sys::wwr_blas_dgemm(
                handle,
                WWR_BLAS_OP_T,
                WWR_BLAS_OP_N,
                n_sph,
                n_sph,
                n_cart,
                &one,
                u.as_ptr(),
                n_cart,
                work.as_ptr(),
                n_cart,
                &zero,
                m.as_mut_ptr(),
                n_sph,
            )
        })
    }
}
