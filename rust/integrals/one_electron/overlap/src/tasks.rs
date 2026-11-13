//! The three tasks after the records: the Cartesian overlap on a stream,
//! U staged on the CPU, and the spherical transform on the stream.

use std::sync::Mutex;

use fb_basis::BasisSet;
use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::gpu_status;
use fb_core_wwr::handle::DeviceHandleStream;
use fb_integrals_mmd::{ClassRows, add_records, launch};
use fb_integrals_sys::fb_overlap_class;
use wwr_sys::{wwr_blas_handle, wwr_blas_operation, wwr_blas_status, wwr_memcpy_kind, wwr_stream};

/// Add the spherical AO overlap of `basis` to `dag` on GPU worker
/// `worker`, and return S: an output row of `nao_sphe²` doubles,
/// column-major, symmetric up to rounding.
///
/// Five tasks, in this order: the records' two
/// ([`add_records`]), then `overlap` (stream: the Cartesian matrix zeroed
/// and one `fb_overlap_class` launch per class, reading the records as an
/// edge), `harmonics` (CPU: U, [`fb_basis::spherical_transform_matrix`],
/// written into a pinned row) and `spherical transform` (stream: U copied
/// into device scratch, then `S = Uᵀ S_cart U` as two GEMMs, the
/// intermediate `S_cart U` scratch too).
///
/// # Errors
/// [`add_records`]' load check of the kernel library, before anything is
/// added.
///
/// # Panics
/// If `basis` has no shells, or `nao_cart²` does not fit an `i32` (the
/// overlap kernels index the matrix in 32 bits).
#[track_caller]
pub fn add_overlap(
    dag: &mut Dag<'_>,
    worker: usize,
    basis: &BasisSet,
) -> Result<RowId<Device, f64>> {
    assert!(basis.nshells() > 0, "add_overlap: a basis with no shells");
    let n_cart = basis.nao_cart() as usize;
    let n_sph = basis.nao_sphe() as usize;
    assert!(
        i32::try_from(n_cart * n_cart).is_ok(),
        "add_overlap: {n_cart}² Cartesian AOs do not fit the kernels' 32-bit index"
    );
    let classes = add_records(dag, worker, basis)?;
    let s_cart = dag.add("overlap", worker, Overlap { classes, n_cart });
    let u = dag.add(
        "harmonics",
        worker,
        Harmonics {
            u: fb_basis::spherical_transform_matrix(basis),
        },
    );
    Ok(dag
        .add(
            "spherical transform",
            worker,
            Transform {
                s_cart,
                u,
                n_cart,
                n_sph,
                blas: Mutex::new(None),
            },
        )
        .s)
}

/// GPU: the Cartesian overlap, `nao_cart²`, zeroed and then summed class by
/// class, every class's contributions added atomically at both mirror
/// positions.
struct Overlap {
    classes: Vec<ClassRows>,
    n_cart: usize,
}

impl Task for Overlap {
    type Rows = RowId<Device, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        for c in &self.classes {
            c.read(device);
        }
        device.output("S cartesian", self.n_cart * self.n_cart)
    }

    fn run(
        &self,
        s: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let stream = stream.stream();
        let s = device.take_mut(*s);
        // SAFETY: `s` is a device row of `s.len()` doubles, on the task's
        // stream.
        gpu_status(unsafe {
            wwr_sys::wwr_memset_async(s.as_mut_ptr().cast(), 0, s.len() * size_of::<f64>(), stream)
        })?;
        for c in &self.classes {
            let view = c.view(device);
            // SAFETY: the view's arrays are the class's n_rec-long rows,
            // written by the records task this one is ordered after, and
            // `s` is n_cart² doubles, zeroed above on this stream; every
            // offset the records hold is a Cartesian AO of the basis.
            launch("fb_overlap_class", c, unsafe {
                fb_overlap_class(stream, &view, self.n_cart as i32, s.as_mut_ptr())
            })?;
        }
        Ok(())
    }
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

/// GPU: `S = Uᵀ S_cart U`, U copied from its pinned row into device
/// scratch, then `S_cart U` into scratch and `Uᵀ (S_cart U)` into S, two
/// GEMMs on the worker's stream, as the C++ `gemmwich` orders them.
struct Transform {
    s_cart: RowId<Device, f64>,
    u: RowId<Pinned, f64>,
    n_cart: usize,
    n_sph: usize,
    /// The handle, made at the first run: WarpWraps' C ABI has no
    /// `wwr_blas_set_workspace`, so the handle cannot take a worker row yet
    /// (`fb_core_arena`, "Not yet"), and a plan is laid out with no device.
    blas: Mutex<Option<Blas>>,
}

#[derive(Clone, Copy)]
struct TransformRows {
    u: RowId<Device, f64>,
    work: RowId<Device, f64>,
    s: RowId<Device, f64>,
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
        device.read(self.s_cart);
        let n_u = self.n_cart * self.n_sph;
        TransformRows {
            u: device.scratch("U", n_u),
            work: device.scratch("S U", n_u),
            s: device.output("S", self.n_sph * self.n_sph),
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
        let s_cart = device.get(self.s_cart);
        let work = device.take_mut(rows.work);
        let s = device.take_mut(rows.s);
        use wwr_blas_operation::{WWR_BLAS_OP_N, WWR_BLAS_OP_T};
        // work = S_cart U: (n_cart × n_cart)(n_cart × n_sph).
        // SAFETY: every matrix is a device row of the declared size, each
        // ld its row count; the scalars are host memory, the handle's
        // default pointer mode; the GEMM enqueues on `stream`.
        blas("wwr_blas_dgemm (S_cart U)", unsafe {
            wwr_sys::wwr_blas_dgemm(
                handle,
                WWR_BLAS_OP_N,
                WWR_BLAS_OP_N,
                n_cart,
                n_sph,
                n_cart,
                &one,
                s_cart.as_ptr(),
                n_cart,
                u.as_ptr(),
                n_cart,
                &zero,
                work.as_mut_ptr(),
                n_cart,
            )
        })?;
        // S = Uᵀ work: (n_sph × n_cart)(n_cart × n_sph).
        // SAFETY: as above.
        blas("wwr_blas_dgemm (Uᵀ S_cart U)", unsafe {
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
                s.as_mut_ptr(),
                n_sph,
            )
        })
    }
}
