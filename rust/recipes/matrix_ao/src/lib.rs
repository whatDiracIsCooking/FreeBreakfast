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
//!   scratch, then `M = Uᵀ M_cart U` as two GEMMs through `wwr-sys`'s BLAS,
//!   the intermediate `M_cart U` scratch too.
//!
//! | C++ | Here |
//! |---|---|
//! | `upload_spherical_transform_matrix(res, basis)` (nothing for an identity U), shared by every pair over the basis | the `harmonics` task's pinned row, added by the first call and handed to the later ones through the caller's slot, always |
//! | `MatrixAOPair`'s U and own scratch, so two pairs may run on different streams | each transform's own device copy of U and its own `M_cart U` scratch |
//! | `MatrixAOPair::cartesian_to_spherical` (`gemmwich`; a copy for an identity U) | the `spherical transform` task's two GEMMs, always |
//! | `res.blas()`, the `DeviceResources`' handle | each transform's own handle, made at its first run |
//! | `MatrixAO<L>`, `BasisLayout`, `ao_count` | nothing: a matrix is a device row, its order the basis's `nao_cart` or `nao_sphe` |
//! | `MatrixAOPair::spherical_to_cartesian` | not ported yet |

use std::sync::Mutex;

use fb_basis::BasisSet;
use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::{Error, Status};
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::gpu_status;
use fb_core_wwr::handle::DeviceHandleStream;
use wwr_sys::{wwr_blas_handle, wwr_blas_operation, wwr_blas_status, wwr_memcpy_kind, wwr_stream};

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
/// # Panics
/// If `basis` has no shells, or `u` holds a row of another worker's table
/// ([`Dag::add`]).
#[track_caller]
pub fn add_cartesian_to_spherical(
    dag: &mut Dag<'_>,
    worker: usize,
    basis: &BasisSet,
    u: &mut Option<RowId<Pinned, f64>>,
    name: &str,
    cartesian: RowId<Device, f64>,
) -> RowId<Device, f64> {
    assert!(
        basis.nshells() > 0,
        "add_cartesian_to_spherical: a basis with no shells"
    );
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
            blas: Mutex::new(None),
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
        host.take_mut(*u).copy_from_slice(&self.u);
        Ok(())
    }
}

/// A BLAS handle, made on the stream worker's thread at the transform's
/// first run and destroyed with the plan.
struct Blas(wwr_blas_handle);

// SAFETY: a cuBLAS or hipBLAS handle may be used from any thread, one call
// at a time; the transform holds it behind a mutex, so no two threads ever
// call through it at once.
unsafe impl Send for Blas {}

impl Drop for Blas {
    fn drop(&mut self) {
        // SAFETY: the handle is live, made by `wwr_blas_create`, and nothing
        // uses it after this.
        let status = unsafe { wwr_sys::wwr_blas_destroy(self.0) };
        if status != wwr_blas_status::WWR_BLAS_STATUS_SUCCESS {
            log::error!("wwr_blas_destroy: {status:?}");
        }
    }
}

/// A BLAS status as a `Status`, naming the call.
#[track_caller]
fn blas(what: &str, status: wwr_blas_status) -> Status {
    if status == wwr_blas_status::WWR_BLAS_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(Error::gpu(format!("{what}: {status:?}")))
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
    /// The handle, made at the first run: WarpWraps' C ABI has no
    /// `wwr_blas_set_workspace`, so the handle cannot take a worker row yet
    /// (`fb_core_arena`, "Not yet"), and a plan is laid out with no device.
    blas: Mutex<Option<Blas>>,
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
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let stream: wwr_stream = stream.stream();
        let u_host = host.get(self.u);
        let u = device.take_mut(rows.u);
        // SAFETY: two rows of `u_host.len()` doubles, on the task's stream.
        // The pinned row is not rewritten before the copy completes: its
        // bytes' next owner is ordered after this task's event by a handoff.
        gpu_status(unsafe {
            wwr_sys::wwr_memcpy_async(
                u.as_mut_ptr().cast(),
                u_host.as_ptr().cast(),
                size_of_val(u_host),
                wwr_memcpy_kind::WWR_MEMCPY_HOST_TO_DEVICE,
                stream,
            )
        })?;

        let mut held = self.blas.lock().unwrap_or_else(|e| e.into_inner());
        if held.is_none() {
            let mut raw: wwr_blas_handle = std::ptr::null_mut();
            // SAFETY: `raw` is valid for one write.
            blas("wwr_blas_create", unsafe {
                wwr_sys::wwr_blas_create(&mut raw)
            })?;
            *held = Some(Blas(raw));
        }
        let handle = held.as_ref().map(|b| b.0).unwrap();
        // SAFETY: the handle is live, and the stream is the worker's, live
        // for the run.
        blas("wwr_blas_set_stream", unsafe {
            wwr_sys::wwr_blas_set_stream(handle, stream)
        })?;

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
        blas("wwr_blas_dgemm (M_cart U)", unsafe {
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
        blas("wwr_blas_dgemm (Uᵀ M_cart U)", unsafe {
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
