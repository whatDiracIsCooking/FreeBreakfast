//! The Cartesian-to-spherical transform of a square AO matrix,
//! `M_sph = Uᵀ M_cart U`, as tasks of a [`Dag`]: every one-electron matrix
//! is built Cartesian (the kernels evaluate Cartesian Gaussians) and means
//! the basis's real solid harmonics. The C++ `src/recipes/matrix_ao` is the
//! reference design.
//!
//! [`add_cartesian_to_spherical`] adds, for one Cartesian matrix row:
//!
//! - **`harmonics`**, on the CPU, and **`harmonics upload`**, on the
//!   worker's stream, once per plan, worker and basis ([`add_harmonics`]):
//!   U, the Cartesian-to-spherical transform
//!   ([`fb_basis::spherical_transform_matrix`]), written into a pinned row
//!   and copied once into a device row that every later transform over the
//!   basis on the worker reads, as does any other task that reads U.
//! - **`spherical transform`**, on the worker's stream: `M = Uᵀ M_cart U`
//!   as two GEMMs ([`fb_core_wwr::blas::gemm`]) on the worker's handle,
//!   reading U's device row, the intermediate `M_cart U` scratch.
//!
//! | C++ | Here |
//! |---|---|
//! | `upload_spherical_transform_matrix(res, basis)` (nothing for an identity U), one device buffer co-owned by every pair over the basis | the `harmonics upload` task's device row ([`add_harmonics`]), added by the first call and handed to the later ones through the caller's slot, always; one per worker, since an edge between workers needs a transport row |
//! | `MatrixAOPair`'s own `work_` scratch | each transform's own `M_cart U` scratch |
//! | `MatrixAOPair::cartesian_to_spherical` (`gemmwich`; a copy for an identity U) | the `spherical transform` task's two GEMMs, always |
//! | `res.blas()`, the `DeviceResources`' handle | the stream worker's handle, [`DeviceStream::blas`], made with the backend and bound to the worker's stream |
//! | `MatrixAO<L>`, `BasisLayout`, `ao_count` | nothing: a matrix is a device row, its order the basis's `nao_cart` or `nao_sphe` |
//! | `MatrixAOPair::spherical_to_cartesian` | not ported yet |
//!
//! The GEMMs run on the worker's BLAS workspace, a worker row of the arena,
//! when the plan declares one ([`Dag::blas_workspace`]), as a plan that runs
//! these transforms, or any other BLAS, does.

#![forbid(unsafe_code)]

use contracts::debug_requires;
use fb_basis::BasisSet;
use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId, memcpy_h2p, memcpy_p2d};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::blas::gemm;
use wwr_sys::wwr_blas_operation::{WWR_BLAS_OP_N as OP_N, WWR_BLAS_OP_T as OP_T};

/// Add the spherical form of the Cartesian AO matrix `cartesian` of `basis`
/// to `dag` on GPU worker `worker`, and return it: an output row of
/// `nao_sphe²` doubles, column-major, named `name`.
///
/// `cartesian` is an earlier task's output of `nao_cart²` doubles,
/// column-major. The task added is `spherical transform` (stream:
/// `M = Uᵀ M_cart U` as two GEMMs reading U's device row, the intermediate
/// `M_cart U` scratch `"{name} U"`). Before it, when `u` is `None`,
/// `harmonics` and `harmonics upload` ([`add_harmonics`]), whose device row
/// is then left in `u`: pass the same slot for every reader of U over
/// `basis` on `worker`, and U is copied up once and read by every one.
///
/// # Panics
/// If `u` holds a row of another worker's table ([`Dag::add_gpu`]).
#[debug_requires(basis.nshells() > 0, "a basis of at least one shell")]
#[track_caller]
pub fn add_cartesian_to_spherical(
    dag: &mut Dag<'_>,
    worker: usize,
    basis: &BasisSet,
    u: &mut Option<RowId<Device, f64>>,
    name: &str,
    cartesian: RowId<Device, f64>,
) -> RowId<Device, f64> {
    let u = add_harmonics(dag, worker, basis, u);
    dag.add_gpu(
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

/// Return U of `basis`, the Cartesian-to-spherical transform, as the
/// device row in `u`: `nao_cart` × `nao_sphe` doubles, column-major. When
/// `u` is `None`, two tasks are added to `dag` on worker `worker` first,
/// and the second's output row left in `u`: `harmonics` (CPU: U,
/// [`fb_basis::spherical_transform_matrix`], written into a pinned row)
/// and `harmonics upload` (stream: that row copied into a device output
/// row, `U`).
///
/// `u` is the plan's slot for U over `basis` on `worker`: every reader of U
/// passes the same one, so U is copied up once per run and every reader
/// reads the one row, a broadcast on the worker
/// (`docs/programming_model.md`, "Workers are a hierarchy").
/// [`add_cartesian_to_spherical`] calls this; a task that reads U itself,
/// such as the three-center integrals' transform, reads the row it
/// returns.
#[debug_requires(basis.nshells() > 0, "a basis of at least one shell")]
#[track_caller]
pub fn add_harmonics(
    dag: &mut Dag<'_>,
    worker: usize,
    basis: &BasisSet,
    u: &mut Option<RowId<Device, f64>>,
) -> RowId<Device, f64> {
    *u.get_or_insert_with(|| {
        let u = fb_basis::spherical_transform_matrix(basis);
        let len = u.len();
        let host = dag.add_cpu("harmonics", worker, Harmonics { u });
        dag.add_gpu("harmonics upload", worker, Upload { host, len })
    })
}

/// CPU: U written into a pinned row, for `harmonics upload` to copy up.
struct Harmonics {
    /// `nao_cart` × `nao_sphe`, column-major.
    u: Vec<f64>,
}

impl CpuTask for Harmonics {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        host.output("U", self.u.len())
    }

    fn run(&self, u: &Self::Rows, host: &Bufs<'_, Pinned>) -> Status {
        memcpy_h2p(host, *u, &self.u);
        Ok(())
    }
}

/// GPU: U's pinned row copied into its device row, the one every reader of
/// U over the basis on the worker reads.
struct Upload {
    host: RowId<Pinned, f64>,
    /// `nao_cart · nao_sphe`.
    len: usize,
}

impl GpuTask for Upload {
    type Rows = RowId<Device, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        host.read(self.host);
        device.output("U", self.len)
    }

    fn run(
        &self,
        u: &Self::Rows,
        device: &Bufs<'_, Device>,
        host: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        memcpy_p2d(&mut device.take_mut(*u), host, self.host, worker)
    }
}

/// GPU: `M = Uᵀ M_cart U` over U's device row, `M_cart U` into scratch and
/// `Uᵀ (M_cart U)` into M, two GEMMs on the worker's stream, as the C++
/// `gemmwich` orders them.
struct Transform {
    m_cart: RowId<Device, f64>,
    u: RowId<Device, f64>,
    n_cart: usize,
    n_sph: usize,
    /// M's row name; the intermediate's is `"{name} U"`.
    name: String,
}

#[derive(Clone, Copy)]
struct TransformRows {
    work: RowId<Device, f64>,
    m: RowId<Device, f64>,
}

impl GpuTask for Transform {
    type Rows = TransformRows;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> TransformRows {
        device.read(self.u);
        device.read(self.m_cart);
        TransformRows {
            work: device.scratch(&format!("{} U", self.name), self.n_cart * self.n_sph),
            m: device.output(&self.name, self.n_sph * self.n_sph),
        }
    }

    fn run(
        &self,
        rows: &TransformRows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        let u = device.get(self.u);

        // The worker's handle, bound to `stream` when the backend was made.
        let handle = worker.blas();

        let (n_cart, n_sph) = (self.n_cart as i32, self.n_sph as i32);
        let m_cart = device.get(self.m_cart);
        let mut work = device.take_mut(rows.work);
        let mut m = device.take_mut(rows.m);
        // work = M_cart U: (n_cart × n_cart)(n_cart × n_sph).
        gemm(
            handle, OP_N, OP_N, n_cart, n_sph, n_cart, 1.0, &m_cart, n_cart, &u, n_cart, 0.0,
            &mut work, n_cart,
        )?;
        // M = Uᵀ work: (n_sph × n_cart)(n_cart × n_sph).
        gemm(
            handle, OP_T, OP_N, n_sph, n_sph, n_cart, 1.0, &u, n_cart, &work, n_cart, 0.0, &mut m,
            n_sph,
        )
    }
}
