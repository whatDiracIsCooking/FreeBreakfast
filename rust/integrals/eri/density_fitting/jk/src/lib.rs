//! Closed-shell J and K over the fitted factor B on the GPU, as tasks of a
//! [`Dag`]:
//!
//! ```text
//! γ      = B vec(D)        (gemv)
//! vec(J) = Bᵀ γ            (gemv)
//!
//! X_m    = B_m C           (one strided-batched gemm over m)    D = 2 C Cᵀ
//! K      = 2 Xᵀ X          (one syrk, lower)
//! upper  = lowerᵀ          (fb-recipes-symmetrize's mirror)
//! ```
//!
//! so that `J_mn = Σ_ls (mn|ls) D_ls` and `K_mn = Σ_ls (ml|ns) D_ls` with
//! `(mn|ls) ≈ Σ_Q B_Q,mn B_Q,ls`, B from
//! `fb_integrals_eri_density_fitting_fit::add_density_fit`. The C++
//! `src/integrals/eri/density_fitting/jk` is the reference design; its
//! README has the layout and why K is built from a factor, never from D.
//!
//! [`add_df_coulomb`] adds **`df coulomb`**, on the stream: the two gemvs,
//! γ the task's scratch. [`add_df_exchange`] adds **`df exchange`**, on the
//! stream: the batched gemm into the scratch X, the syrk into K's lower
//! triangle, and the mirror of it onto the upper, so K's `(m, n)` and
//! `(n, m)` carry the same bits. Each reads B and its density's row and
//! writes an output row of its own, so an SCF iteration's J and K, two
//! tasks with no edge between them, may run on two workers' streams.
//!
//! **The layout** is the C++ one, every matrix column-major and packed:
//!
//! | Row | Shape | Element | Size (doubles) |
//! |---|---|---|---|
//! | B | `naux` × `nao²` | `Q + naux (n + nao m)`, Q fastest | `naux · nao²` |
//! | D, J, K | `nao` × `nao` | `n + nao m` | `nao²` each |
//! | C | `nao` × `nocc` | `n + nao i` | `nao · nocc` |
//! | γ (scratch, J) | `naux` | `Q` | `naux` |
//! | X (scratch, K) | `(naux · nocc)` × `nao` | `Q + naux (i + nocc m)` | `naux · nocc · nao` |
//!
//! `B_m`, the `naux` × `nao` block of B at `naux · nao · m`, is read in
//! place by the batch at stride `naux · nao`; C is shared by it, stride 0.
//! X is B's size times `nocc / nao`, below it whenever the occupied space
//! is smaller than the basis.
//!
//! | C++ | Here |
//! |---|---|
//! | `df_coulomb(res, nao, naux, b, density, j)`, `j` the caller's | [`add_df_coulomb`], J an output row of its own |
//! | `df_exchange(res, nao, naux, nocc, b, c, k)`, `k` the caller's | [`add_df_exchange`], K an output row of its own |
//! | gamma and X carved from `res.device_workspace()` by `device_regions` | the tasks' scratch rows, read off the plan's table |
//! | `df_coulomb_footprint(nao, naux)`, `df_exchange_footprint(nao, naux, nocc)` | the tasks' layouts |
//! | `wwr::gemv<double, int32_t>`, `wwr::gemmStridedBatched<…>`, `wwr::syrk<…>` with `kOne`, `kZero`, `kTwo` | [`gemv`], [`gemm_strided_batched`] and [`syrk`], the same modes, on the stream worker's handle, the scalars by value |
//! | `symmetrize(res, WWRBLAS_FILL_MODE_LOWER, nao, k, nao)` | [`symmetrize_from`], the same fill mode |
//! | `nao == 0`: an empty success; `naux` (or `nocc`) `== 0`: J (or K) zeroed; a negative count: `InvalidConfig` | refused at planning: every count is at least 1 |
//! | `nao²` (J) or `naux · nocc` (K) past `int32_t`: `InvalidConfig` | an always-on assert, since the BLAS's counts are `i32`s |
//! | `ScopedDeviceIndex` for the call | nothing: a stream worker's tasks run on its device |
//! | nothing: the C++ tree compiles its kernels into the binary | [`fb_recipes_sys::check_library`] first, at planning, for K's mirror |
//! | `df_exchange(res, nao, naux, rank, b, left, right, k)`, the two-factor K | not ported: an SCF's density is `2 C Cᵀ`; a response density's port brings it |
//! | the block forms, `df_coulomb(…, count, …)` and `df_exchange(…, rank, count, …)`, and `df_exchange_symmetrized` | not ported: their callers (CPHF, the CASSCF response) are not yet |
//!
//! [`gemv`]: fb_core_wwr::blas::gemv
//! [`gemm_strided_batched`]: fb_core_wwr::blas::gemm_strided_batched
//! [`syrk`]: fb_core_wwr::blas::syrk

#![forbid(unsafe_code)]

use contracts::debug_requires;
use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_core_wwr::blas::{gemm_strided_batched, gemv, syrk};
use fb_recipes_symmetrize::symmetrize_from;
use wwr_sys::{wwr_blas_fill_mode, wwr_blas_operation};

/// Add the Coulomb matrix J of the density `density` over the fitted factor
/// `b` to `dag` on GPU worker `worker`, and return it: an output row of
/// `nao²` doubles named `"J"`, column-major, `vec(J) = Bᵀ (B vec(D))`.
///
/// `b` is an earlier task's output of `naux · nao²` doubles, B's layout (Q
/// fastest, then n, then m); `density` one of `nao²` doubles, column-major.
/// Both are left as they were. The task added is `df coulomb` (stream: two
/// gemvs, γ = B vec(D) the scratch `"gamma"` of `naux` doubles). J is
/// symmetric to rounding when D is, not bit for bit.
///
/// # Panics
/// If `naux` or `nao²` does not fit an `i32`, the BLAS's integer, or a row
/// is of another worker's table ([`Dag::add`]).
#[debug_requires(naux > 0, "a fitting set of at least one function")]
#[debug_requires(nao > 0, "an orbital basis of at least one function")]
#[track_caller]
pub fn add_df_coulomb(
    dag: &mut Dag<'_>,
    worker: usize,
    nao: usize,
    naux: usize,
    b: RowId<Device, f64>,
    density: RowId<Device, f64>,
) -> RowId<Device, f64> {
    assert!(
        i32::try_from(naux).is_ok(),
        "add_df_coulomb: naux = {naux} does not fit the BLAS's i32"
    );
    assert!(
        nao.checked_mul(nao)
            .is_some_and(|pairs| i32::try_from(pairs).is_ok()),
        "add_df_coulomb: nao² = {nao}² does not fit the BLAS's i32"
    );
    dag.add(
        "df coulomb",
        worker,
        Coulomb {
            b,
            density,
            nao,
            naux,
        },
    )
    .j
}

/// Add the exchange matrix K of the density `D = 2 C Cᵀ` over the fitted
/// factor `b` to `dag` on GPU worker `worker`, and return it: an output row
/// of `nao²` doubles named `"K"`, column-major, `K_mn = Σ_Q (B_Q D B_Q)_mn`,
/// exactly symmetric: `(m, n)` and `(n, m)` carry the same bits.
///
/// `b` is an earlier task's output of `naux · nao²` doubles, B's layout (Q
/// fastest, then n, then m); `c` one of `nao · nocc` doubles, the factor C,
/// column-major, nao × nocc (C's occupied columns in an SCF, or any factor
/// of a positive semidefinite density: K depends on C only through C Cᵀ).
/// Both are left as they were. The task added is `df exchange` (stream: X =
/// [B_m C] by one strided-batched gemm into the scratch `"X"` of
/// `naux · nocc · nao` doubles, `K = 2 Xᵀ X` by one syrk into K's lower
/// triangle, then [`symmetrize_from`] mirrors it onto the upper).
///
/// First, before anything is added, the kernel library's load check
/// ([`fb_recipes_sys::check_library`]), since the mirror is its launch.
///
/// # Errors
/// A `GpuRuntime` error with the check's message, if the loaded
/// `libfb_recipes_kernels.so` is not the one the bindings describe.
///
/// # Panics
/// If `nao`, `naux · nao` or `naux · nocc` does not fit an `i32`, the
/// BLAS's integer, or a row is of another worker's table ([`Dag::add`]).
#[debug_requires(naux > 0, "a fitting set of at least one function")]
#[debug_requires(nao > 0, "an orbital basis of at least one function")]
#[debug_requires(nocc > 0, "a factor of at least one column")]
#[track_caller]
pub fn add_df_exchange(
    dag: &mut Dag<'_>,
    worker: usize,
    nao: usize,
    naux: usize,
    nocc: usize,
    b: RowId<Device, f64>,
    c: RowId<Device, f64>,
) -> Result<RowId<Device, f64>> {
    for (what, count) in [
        ("nao", Some(nao)),
        ("naux · nao", naux.checked_mul(nao)),
        ("naux · nocc", naux.checked_mul(nocc)),
    ] {
        assert!(
            count.is_some_and(|count| i32::try_from(count).is_ok()),
            "add_df_exchange: {what} (nao = {nao}, naux = {naux}, nocc = {nocc}) does not fit \
             the BLAS's i32"
        );
    }
    // Not `map_err(Error::gpu)`: through a function pointer the error's
    // origin would be core's, not this call's caller.
    if let Err(e) = fb_recipes_sys::check_library() {
        return Err(Error::gpu(e));
    }
    Ok(dag
        .add(
            "df exchange",
            worker,
            Exchange {
                b,
                c,
                nao,
                naux,
                nocc,
            },
        )
        .k)
}

/// GPU: `γ = B vec(D)`, then `vec(J) = Bᵀ γ`.
struct Coulomb {
    b: RowId<Device, f64>,
    density: RowId<Device, f64>,
    nao: usize,
    naux: usize,
}

/// The Coulomb task's rows: J, and its scratch γ.
#[derive(Clone, Copy)]
struct CoulombRows {
    j: RowId<Device, f64>,
    gamma: RowId<Device, f64>,
}

impl Task for Coulomb {
    type Rows = CoulombRows;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> CoulombRows {
        device.read(self.b);
        device.read(self.density);
        CoulombRows {
            gamma: device.scratch("gamma", self.naux),
            j: device.output("J", self.nao * self.nao),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("nao", self.nao), ("naux", self.naux)]
    }

    fn run(
        &self,
        rows: &CoulombRows,
        device: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream: worker, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let (mut gamma, mut j) = (device.take_mut(rows.gamma), device.take_mut(rows.j));
        let b = device.get(self.b);
        // add_df_coulomb checked that naux and nao² fit an i32; the handle
        // is the worker's, bound to its stream, so the two run in order.
        let (naux, pairs) = (self.naux as i32, (self.nao * self.nao) as i32);
        gemv(
            worker.blas(),
            wwr_blas_operation::WWR_BLAS_OP_N,
            naux,
            pairs,
            1.0,
            &b,
            naux,
            &device.get(self.density),
            1,
            0.0,
            &mut gamma,
            1,
        )?;
        gemv(
            worker.blas(),
            wwr_blas_operation::WWR_BLAS_OP_T,
            naux,
            pairs,
            1.0,
            &b,
            naux,
            &gamma,
            1,
            0.0,
            &mut j,
            1,
        )
    }
}

/// GPU: `X_m = B_m C`, `K = 2 Xᵀ X` on the lower triangle, then its mirror.
struct Exchange {
    b: RowId<Device, f64>,
    c: RowId<Device, f64>,
    nao: usize,
    naux: usize,
    nocc: usize,
}

/// The exchange task's rows: K, and its scratch X.
#[derive(Clone, Copy)]
struct ExchangeRows {
    k: RowId<Device, f64>,
    x: RowId<Device, f64>,
}

impl Task for Exchange {
    type Rows = ExchangeRows;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> ExchangeRows {
        device.read(self.b);
        device.read(self.c);
        ExchangeRows {
            x: device.scratch("X", self.naux * self.nocc * self.nao),
            k: device.output("K", self.nao * self.nao),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("nao", self.nao), ("naux", self.naux), ("nocc", self.nocc)]
    }

    fn run(
        &self,
        rows: &ExchangeRows,
        device: &Bufs<'_, Device>,
        _: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream: worker, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        let (mut x, mut k) = (device.take_mut(rows.x), device.take_mut(rows.k));
        // add_df_exchange checked that nao, naux · nao and naux · nocc fit
        // an i32 (so do naux and nocc); the handle is the worker's, bound to
        // its stream, so the three run in order.
        let (nao, naux, nocc) = (self.nao as i32, self.naux as i32, self.nocc as i32);
        let rows_x = naux * nocc;
        gemm_strided_batched(
            worker.blas(),
            wwr_blas_operation::WWR_BLAS_OP_N,
            wwr_blas_operation::WWR_BLAS_OP_N,
            naux,
            nocc,
            nao,
            1.0,
            &device.get(self.b),
            naux,
            i64::from(naux * nao),
            &device.get(self.c),
            nao,
            0,
            0.0,
            &mut x,
            naux,
            i64::from(rows_x),
            nao,
        )?;
        syrk(
            worker.blas(),
            wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER,
            wwr_blas_operation::WWR_BLAS_OP_T,
            nao,
            rows_x,
            2.0,
            &x,
            rows_x,
            0.0,
            &mut k,
            nao,
        )?;
        symmetrize_from(
            worker,
            wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER,
            self.nao,
            &mut k,
            self.nao,
        )
    }
}
