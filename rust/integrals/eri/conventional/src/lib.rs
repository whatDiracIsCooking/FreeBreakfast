//! The conventional two-electron integrals on the GPU, as tasks of a
//! [`Dag`]: the whole four-index tensor of a small basis stored once,
//! packed by AO pair, with its exchange-ordered gather beside it, and each
//! J and K then one GEMV over one of them:
//!
//! ```text
//! ERI_pp[pidx(m, n), pidx(l, s)] = (mn|ls)                     (once)
//! X_pp[pidx(l, s), pidx(m, n)]   = ((ml|ns) + (ms|nl)) / 2     (once)
//!
//! Dp   = pack(U D Uᵀ)              Dp[pidx(l, s)] = D_ls + D_sl, D_ll
//! J_pp = ERI_pp Dp,  K_pp = X_pp Dp                          (two gemvs)
//! J    = Uᵀ unpack(J_pp) U,  K = Uᵀ unpack(K_pp) U
//! ```
//!
//! so that `J_mn = Σ_ls (mn|ls) D_ls` and `K_mn = Σ_ls (ml|ns) D_ls` exactly,
//! for a symmetric D, with no fit: the reference is the exact `get_jk`.
//! `pidx(m, n) = M (M + 1) / 2 + N`, `M = max(m, n)`, `N = min(m, n)`, and
//! `n_pp = n_cart (n_cart + 1) / 2` pairs (`fb_integrals_sys`'s
//! `conventional` module has the packing). This is FelizJueves's packed
//! full ERI, its two-tensor design (`eri/full_eri/`): one tensor for J,
//! one exchange-ordered for K, each read by the same product.
//!
//! **Cartesian inside, spherical outside.** The kernels evaluate Cartesian
//! Gaussians, so `ERI_pp` and `X_pp` are over the basis's `n_cart`
//! Cartesian AOs, and a spherical density enters as `D_cart = U D Uᵀ` and
//! its J and K leave as `Uᵀ J_cart U`, U the basis's Cartesian-to-spherical
//! transform, the one every one-electron matrix reads
//! ([`fb_recipes_matrix_ao::add_harmonics`]). That is two `n_cart²`
//! transforms a build, and the tensor's size is the Cartesian count's: the
//! [`FB_CONVENTIONAL_MAX_AO`] cap and admission both see `n_cart`.
//!
//! [`add_conventional`] adds the tensor and the gather once per plan
//! (`fb_integrals_sys::add_conventional_tasks`: `conventional tables`,
//! `conventional eri`, `exchange gather`), [`add_conventional_jk`] one
//! build of J and K, `conventional jk`, for each density a reader has.
//!
//! **Memory is admission's.** `ERI_pp` and `X_pp` are `n_pp²` doubles each,
//! output rows of the plan's table that every reader's edge keeps live, so
//! a basis whose pair does not fit the card is refused at planning by
//! [`fb_core_arena::admit`], naming the bytes over and the rows at the peak.
//! At the cap, 150 Cartesian AOs, each is 1.03 GB.
//!
//! **What it is for.** A small system: the tensor grows as `n_cart⁴ / 4`.
//! It is exact where density fitting is not, and once built each build is
//! two GEMVs, so a run that builds many Fock matrices over a few dozen AOs
//! pays the tensor once.
//!
//! | C++ (`free_breakfast`) | Here |
//! |---|---|
//! | nothing: the C++ tree stores no four-index tensor | [`add_conventional`] and [`add_conventional_jk`], rust/'s own, over FelizJueves's `eri/full_eri` kernels |
//!
//! [`FB_CONVENTIONAL_MAX_AO`]: fb_integrals_sys::FB_CONVENTIONAL_MAX_AO

#![forbid(unsafe_code)]

use contracts::debug_requires;
use fb_basis::BasisSet;
use fb_core_arena::{Bufs, CopyOnly, Device, Layout, Pinned, RowId};
use fb_core_error_handling::{Error, Result, Status};
use fb_core_planner::{Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::blas::{gemm, gemv};
use fb_integrals_mmd::record_tables;
use fb_integrals_sys::{
    ClassRows, FB_CONVENTIONAL_MAX_AO, FB_CONVENTIONAL_MAX_MOMENTUM, add_conventional_tasks, pack,
    packed_pairs, unpack,
};
use fb_recipes_matrix_ao::add_harmonics;
use fb_recipes_symmetrize::symmetrize;
use wwr_sys::wwr_blas_operation::{WWR_BLAS_OP_N as OP_N, WWR_BLAS_OP_T as OP_T};

/// The conventional integrals of a plan, [`add_conventional`]'s product:
/// what [`add_conventional_jk`] reads.
#[derive(Clone, Copy, Debug)]
pub struct Conventional {
    /// `ERI_pp`: an output row of `n_pp²` doubles, the crate doc's.
    pub eri: RowId<Device, f64>,
    /// `X_pp`: an output row of `n_pp²` doubles, the crate doc's.
    pub x: RowId<Device, f64>,
    /// U, the basis's Cartesian-to-spherical transform: the plan's device
    /// row, `n_cart` × `n_sph`, column-major.
    pub u: RowId<Device, f64>,
    /// The basis's Cartesian AO count, which `pidx` runs over.
    pub n_cart: usize,
    /// The basis's spherical AO count, the order of D, J and K.
    pub n_sph: usize,
}

/// Add the conventional integrals of `basis` to `dag` on GPU worker
/// `worker`, and return them: `ERI_pp`, `X_pp` and U's row.
///
/// `classes` are the records of `basis`, added on `worker` by
/// [`fb_integrals_mmd::add_records`] earlier in `dag`, so one build serves
/// every reader of the plan. `u` is the plan's slot for U over `basis` on
/// `worker`, shared with every other reader (the one-electron matrices'
/// transforms): `harmonics` and `upload U` are added when it is `None`,
/// and U's device row left there ([`add_harmonics`]). The tasks are then
/// `conventional tables` (CPU), `conventional eri` and `exchange gather`
/// (stream), `fb_integrals_sys::add_conventional_tasks`'.
///
/// First, before anything is added, the kernel libraries' load checks
/// ([`fb_integrals_sys::check_library`], and
/// [`fb_recipes_sys::check_library`] for the average
/// [`add_conventional_jk`] launches), and then what the kernels can take,
/// both at planning.
///
/// # Errors
/// A `GpuRuntime` error with the check's message, if a loaded kernel
/// library is not the one the bindings describe; an `Unsupported` one
/// naming the shell, if a shell is above d
/// ([`FB_CONVENTIONAL_MAX_MOMENTUM`]), or naming the count, if the basis
/// has more than [`FB_CONVENTIONAL_MAX_AO`] Cartesian AOs. Nothing is added
/// to `dag` on an error.
///
/// # Panics
/// If `classes`, or the row `u` holds, are rows of another worker's or
/// another plan's table ([`Dag::add_gpu`]), or, in every build, if
/// `classes` are not the records of `basis`
/// ([`fb_integrals_sys::add_conventional_tasks`]' contract against its
/// tables).
#[debug_requires(basis.nshells() > 0, "a basis of at least one shell")]
#[track_caller]
pub fn add_conventional(
    dag: &mut Dag<'_>,
    worker: usize,
    basis: &BasisSet,
    classes: &[ClassRows],
    u: &mut Option<RowId<Device, f64>>,
) -> Result<Conventional> {
    // Not `map_err(Error::gpu)`: through a function pointer the error's
    // origin would be core's, not this call's caller.
    if let Err(e) = fb_integrals_sys::check_library() {
        return Err(Error::gpu(e));
    }
    if let Err(e) = fb_recipes_sys::check_library() {
        return Err(Error::gpu(e));
    }
    if let Some((s, shell)) = (0..)
        .zip(basis.shells())
        .find(|(_, shell)| shell.angular_momentum() > FB_CONVENTIONAL_MAX_MOMENTUM)
    {
        return Err(Error::unsupported(format!(
            "add_conventional: shell {s} has momentum {}, but the conventional ERI's kernels \
             cover shells through l = {FB_CONVENTIONAL_MAX_MOMENTUM} (d)",
            shell.angular_momentum()
        )));
    }
    let n_cart = basis.nao_cart() as usize;
    if n_cart > FB_CONVENTIONAL_MAX_AO as usize {
        return Err(Error::unsupported(format!(
            "add_conventional: the basis has {n_cart} Cartesian AOs, but the conventional ERI \
             stores n_pp² doubles for at most {FB_CONVENTIONAL_MAX_AO} (two tensors of {} \
             bytes there); use density fitting",
            8 * packed_pairs(FB_CONVENTIONAL_MAX_AO as usize).map_or(0, |n| n * n)
        )));
    }
    let tables = record_tables(basis);
    let u = add_harmonics(dag, worker, basis, u);
    let [eri, x] = add_conventional_tasks(dag, worker, classes, &tables);
    Ok(Conventional {
        eri,
        x,
        u,
        n_cart,
        n_sph: basis.nao_sphe() as usize,
    })
}

/// Add J and K of the spherical density `density` over the conventional
/// integrals `eri` to `dag` on GPU worker `worker`, and return them,
/// `(J, K)`: two output rows of `n_sph²` doubles named `"J"` and `"K"`,
/// column-major, `J_mn = Σ_ls (mn|ls) D_ls` and `K_mn = Σ_ls (ml|ns) D_ls`,
/// each exactly symmetric: `(m, n)` and `(n, m)` carry the same bits.
///
/// `density` is an earlier task's output of `n_sph²` doubles, symmetric
/// (the packing folds `D_ls` and `D_sl` together, so a non-symmetric D's
/// K is its symmetric part's), and is left as it was. The task added is
/// `conventional jk` (stream): `D_cart = U D Uᵀ` by two gemms, [`pack`]
/// into `Dp`, `J_pp = ERI_pp Dp` and `K_pp = X_pp Dp` by two
/// [`gemv`]s, [`unpack`] each into a Cartesian matrix, `Uᵀ · U` around
/// each by two gemms into J and K, and each averaged with its transpose
/// ([`symmetrize`]). Its scratch is the Cartesian D, J and K, the gemms'
/// intermediate and the three packed vectors, `3 n_cart² + n_cart n_sph +
/// 3 n_pp` doubles.
///
/// # Panics
/// If a row is of another worker's table ([`Dag::add_gpu`]).
#[track_caller]
pub fn add_conventional_jk(
    dag: &mut Dag<'_>,
    worker: usize,
    eri: &Conventional,
    density: RowId<Device, f64>,
) -> (RowId<Device, f64>, RowId<Device, f64>) {
    let rows = dag.add_gpu("conventional jk", worker, Jk { eri: *eri, density });
    (rows.j, rows.k)
}

/// GPU: one build of J and K over the stored tensors.
struct Jk {
    eri: Conventional,
    density: RowId<Device, f64>,
}

#[derive(Clone, Copy)]
struct JkRows {
    work: RowId<Device, f64>,
    d_cart: RowId<Device, f64>,
    j_cart: RowId<Device, f64>,
    k_cart: RowId<Device, f64>,
    dp: RowId<Device, f64>,
    jp: RowId<Device, f64>,
    kp: RowId<Device, f64>,
    j: RowId<Device, f64>,
    k: RowId<Device, f64>,
}

impl GpuTask for Jk {
    type Rows = JkRows;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> JkRows {
        let e = &self.eri;
        device.read(e.eri);
        device.read(e.x);
        device.read(e.u);
        device.read(self.density);
        let (nc, ns) = (e.n_cart, e.n_sph);
        // n_cart is within FB_CONVENTIONAL_MAX_AO (add_conventional).
        let n_pp = nc * (nc + 1) / 2;
        JkRows {
            work: device.scratch("jk U", nc * ns),
            d_cart: device.scratch("D cartesian", nc * nc),
            j_cart: device.scratch("J cartesian", nc * nc),
            k_cart: device.scratch("K cartesian", nc * nc),
            dp: device.scratch("Dp", n_pp),
            jp: device.scratch("Jp", n_pp),
            kp: device.scratch("Kp", n_pp),
            j: device.output("J", ns * ns),
            k: device.output("K", ns * ns),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n_cart", self.eri.n_cart), ("n_ao", self.eri.n_sph)]
    }

    fn run(
        &self,
        rows: &JkRows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        let e = &self.eri;
        let handle = stream.blas();
        let (u, d) = (device.get(e.u), device.get(self.density));
        let (eri, x) = (device.get(e.eri), device.get(e.x));
        let mut work = device.take_mut(rows.work);
        let mut d_cart = device.take_mut(rows.d_cart);
        let mut dp = device.take_mut(rows.dp);
        // In range: n_cart is within FB_CONVENTIONAL_MAX_AO
        // (add_conventional), so every count and n_pp is an i32.
        let (nc, ns) = (e.n_cart as i32, e.n_sph as i32);
        let n_pp = nc * (nc + 1) / 2;

        // D_cart = U D Uᵀ: work = U D (n_cart × n_sph), then work Uᵀ.
        gemm(
            handle, OP_N, OP_N, nc, ns, ns, 1.0, &u, nc, &d, ns, 0.0, &mut work, nc,
        )?;
        gemm(
            handle,
            OP_N,
            OP_T,
            nc,
            nc,
            ns,
            1.0,
            &work,
            nc,
            &u,
            nc,
            0.0,
            &mut d_cart,
            nc,
        )?;
        pack(stream, e.n_cart, &d_cart, &mut dp)?;

        for (tensor, packed, cart, out) in [
            (&eri, rows.jp, rows.j_cart, rows.j),
            (&x, rows.kp, rows.k_cart, rows.k),
        ] {
            let mut packed = device.take_mut(packed);
            let mut cart = device.take_mut(cart);
            let mut out = device.take_mut(out);
            // The tensor is exactly symmetric, so the product is the same
            // read either way; untransposed, a column of it is contiguous.
            gemv(
                handle,
                OP_N,
                n_pp,
                n_pp,
                1.0,
                tensor,
                n_pp,
                &dp,
                1,
                0.0,
                &mut packed,
                1,
            )?;
            unpack(stream, e.n_cart, &packed, &mut cart)?;
            // Uᵀ V U: work = V U (n_cart × n_sph), then Uᵀ work.
            gemm(
                handle, OP_N, OP_N, nc, ns, nc, 1.0, &cart, nc, &u, nc, 0.0, &mut work, nc,
            )?;
            gemm(
                handle, OP_T, OP_N, ns, ns, nc, 1.0, &u, nc, &work, nc, 0.0, &mut out, ns,
            )?;
            // The two gemms round (m, n) and (n, m) in different orders.
            symmetrize(stream, e.n_sph, &mut out, e.n_sph)?;
        }
        Ok(())
    }
}
