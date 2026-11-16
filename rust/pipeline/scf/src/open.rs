//! The open shell's nodes (ROHF, `n_alpha > n_beta`), what it changes in
//! the C++ loop's steps 1, 2 and 6: the spin densities, the spin Fock
//! matrices and their energy, the Roothaan effective Fock matrix, the
//! densities from the occupied orbitals, and the published orbitals with
//! the open ones made PySCF's.

use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId, memcpy_d2p};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::blas::{geam, gemm, gemv};
use fb_core_wwr::copy;
use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut, split_at_mut};
use fb_recipes_diagonalize::add_diagonalize;
use wwr_sys::wwr_blas_operation::{WWR_BLAS_OP_N as OP_N, WWR_BLAS_OP_T as OP_T};

use super::iteration::add_residual;

/// The C++ `kOpenBlockBetaWeight`: the open-open block of the Roothaan
/// matrix the loop iterates on is `Fo = (1 - w) Fa + w Fb` at this `w`,
/// not PySCF's `Fc = (Fa + Fb)/2`.
///
/// Where the open eigenvalues sit against the core and virtual ones is the
/// Roothaan step's estimate of the energy's curvature along a core-open
/// rotation (a beta electron moving, coupled through Fb) and an
/// open-virtual one (an alpha electron, through Fa). On a localized hole
/// `Fc` puts the open orbital below doubly occupied ones (glycylglycine⁺ in
/// STO-3G), so the minimum is no aufbau state of PySCF's matrix and the
/// iteration moves the hole from lone pair to lone pair (the C++ tree's
/// issue #476); `w = 1` inverts the open-virtual gap instead (NO,
/// CH₂/cc-pVTZ). Over twenty open shells in PySCF, 0.7 and 0.8 reach every
/// minimum; 3/4 is the middle. The minimum, the energy and the residual do
/// not read the diagonal blocks, and the published orbitals are made
/// PySCF's ([`add_publish_open`]).
pub(crate) const OPEN_BLOCK_BETA_WEIGHT: f64 = 0.75;

/// Add, before the loop, **`spin densities`** (stream): `Da = Db = D/2`
/// from the guess `d`, as PySCF's ROHF splits a density that carries no
/// spin, into the output rows `"Da"` and `"Db"` of `n²` doubles, which the
/// loop carries. And **`identity`** (CPU: the identity into the pinned row
/// `"I host"`) with its upload, the device row `"I"` the Roothaan matrix's
/// virtual projector is made from. Returns `[Da, Db, I]`.
pub(crate) fn add_spin_densities(
    dag: &mut Dag<'_>,
    worker: usize,
    n: usize,
    d: RowId<Device, f64>,
) -> [RowId<Device, f64>; 3] {
    let [da, db] = dag.add_gpu("spin densities", worker, SpinDensities { d, n });
    let host = dag.add_cpu("identity", worker, Identity { n });
    let identity = dag.upload(worker, host, "I", n * n);
    [da, db, identity]
}

/// Add an open shell's Fock build from J and the two exchange matrices, its
/// energy, its Roothaan matrix and its residual, and return
/// `(F, [Fa, Fb], e, scalars)`, the rows [`add_fock`](crate::add_fock)
/// returns with the spin Fock matrices beside them:
///
/// ```text
/// Fa = H + J - Ka/2        Fb = H + J - Kb/2
/// E  = tr(D H + Da Fa + Db Fb)/2 + E_nuc        scalars[0]
/// F  = A + Aᵀ, the Roothaan matrix below
/// e  = Xᵀ (F D S - S D F) X
/// ‖e‖_F                                         scalars[1]
/// ```
///
/// `ka` and `kb` are `df_exchange` of the alpha and beta factors, whose
/// `D = 2 C Cᵀ` convention makes each twice the spin density's K, so
/// `Ka/2 = K(Da)`; `d = da + db`. The tasks added are:
///
/// - **`open fock`** (stream): `H + J` into the scratch `"H + J"`, then Fa
///   and Fb by a geam each, the output rows `"Fa"` and `"Fb"`; the energy's
///   three traces by three gemvs into a device scratch and down to the
///   pinned `"E elec"`.
/// - **`roothaan fock`** (stream): with `Fc = (Fa + Fb)/2`,
///   `Fo = (1 - w) Fa + w Fb` at [`OPEN_BLOCK_BETA_WEIGHT`], and the
///   projectors `Pc = Db S`, `Po = (Da - Db) S`, `Pv = I - Da S`,
///   `A = Pcᵀ Fc Pc/2 + Poᵀ Fo Po/2 + Pvᵀ Fc Pv/2 + Poᵀ Fb Pc +
///   Poᵀ Fa Pv + Pvᵀ Fc Pc` and `F = A + Aᵀ` into the output row `"F"`,
///   symmetric bit for bit, which the residual's one product needs: the C++
///   `roothaan_fock`, PySCF's `get_roothaan_fock` but for the open-open
///   block. Made from the densities, so the first build needs no C.
/// - `residual` and `energy and norm`, as [`add_fock`](crate::add_fock)
///   adds them, on `(F, D)`.
#[allow(
    clippy::too_many_arguments,
    reason = "one iteration's operator rows, each its own"
)]
#[allow(
    clippy::type_complexity,
    reason = "the build's products as rows, in the crate's tuple style"
)]
pub(crate) fn add_open_fock(
    dag: &mut Dag<'_>,
    worker: usize,
    n: usize,
    e_nuc: f64,
    [h, j, ka, kb]: [RowId<Device, f64>; 4],
    [d, da, db]: [RowId<Device, f64>; 3],
    [s, x, identity]: [RowId<Device, f64>; 3],
) -> (
    RowId<Device, f64>,
    [RowId<Device, f64>; 2],
    RowId<Device, f64>,
    RowId<Pinned, f64>,
) {
    let fock = dag.add_gpu(
        "open fock",
        worker,
        OpenFock {
            h,
            j,
            ka,
            kb,
            d,
            da,
            db,
            n,
        },
    );
    let RoothaanRows { f, .. } = dag.add_gpu(
        "roothaan fock",
        worker,
        RoothaanFock {
            fa: fock.fa,
            fb: fock.fb,
            da,
            db,
            s,
            identity,
            n,
        },
    );
    let (e, scalars) = add_residual(dag, worker, n, e_nuc, f, fock.e_elec, [d, s, x]);
    (f, [fock.fa, fock.fb], e, scalars)
}

/// Add **`open density`** (stream) to the loop's body: C's first `n_alpha`
/// and `n_beta` columns copied into the carried factors `ca` and `cb` (a
/// geam each), `Da = Ca Caᵀ` and `Db = Cb Cbᵀ` (a gemm each) and
/// `D = Da + Db` (a geam), each row the loop carries written as its last
/// user in the body, as [`add_density`](crate::add_density) writes a closed
/// shell's.
pub(crate) fn add_open_density(
    dag: &mut Dag<'_>,
    worker: usize,
    n: usize,
    [n_alpha, n_beta]: [usize; 2],
    c: RowId<Device, f64>,
    [d, da, db]: [RowId<Device, f64>; 3],
    [ca, cb]: [RowId<Device, f64>; 2],
) {
    dag.add_gpu(
        "open density",
        worker,
        OpenDensity {
            c,
            d,
            da,
            db,
            ca,
            cb,
            n,
            n_alpha,
            n_beta,
        },
    );
}

/// Add, after the loop, the open shell's published orbitals and return
/// `[ε, C, D]`, output rows of `n`, `n²` and `n²` doubles: C the canonical
/// orbitals `c` of the converged Roothaan matrix as built, with its open
/// ones, columns `[n_beta, n_alpha)`, rotated among themselves to
/// diagonalize `Fc = (Fa + Fb)/2` there, and ε `eps` with those columns'
/// energies Fc's eigenvalues: PySCF's ROHF `mo_energy`, every one, by
/// occupation class, since the two matrices differ in the open block
/// alone. `D = Ca Caᵀ + Cb Cbᵀ` from `c`, which the rotation leaves as it
/// is. The C++ `canonicalize_open`. The tasks added are:
///
/// - **`open block`** (stream): the open columns copied into the output
///   `"C open"` (n × m, `m = n_alpha - n_beta`), Fc, and `C_oᵀ Fc C_o`
///   into the output `"open block"` (m × m).
/// - [`add_diagonalize`] on the block, `"open U"` and `"open w"`, at the
///   work `lwork` asked at `n`, which syevd at `m <= n` takes as enough.
/// - **`publish open`** (stream): C copied into `"C"` with its open columns
///   replaced by `C_o U`, ε into `"epsilon"` with theirs by `w`, and D by
///   two gemms over `c`'s leading columns into `"D"`.
#[allow(
    clippy::too_many_arguments,
    reason = "the converged rows, each its own"
)]
pub(crate) fn add_publish_open(
    dag: &mut Dag<'_>,
    worker: usize,
    n: usize,
    lwork: usize,
    [n_alpha, n_beta]: [usize; 2],
    [fa, fb]: [RowId<Device, f64>; 2],
    eps: RowId<Device, f64>,
    c: RowId<Device, f64>,
) -> [RowId<Device, f64>; 3] {
    let m = n_alpha - n_beta;
    let block = dag.add_gpu(
        "open block",
        worker,
        OpenBlock {
            c,
            fa,
            fb,
            n,
            n_beta,
            m,
        },
    );
    let (u, w) = add_diagonalize(dag, worker, "open", m, lwork, block.block);
    let out = dag.add_gpu(
        "publish open",
        worker,
        PublishOpen {
            c,
            eps,
            c_open: block.c_open,
            u,
            w,
            n,
            n_alpha,
            n_beta,
        },
    );
    [out.eps, out.c, out.d]
}

/// GPU: `Da = D/2`, `Db = Da`.
struct SpinDensities {
    d: RowId<Device, f64>,
    n: usize,
}

impl GpuTask for SpinDensities {
    type Rows = [RowId<Device, f64>; 2];

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.d);
        let n2 = self.n * self.n;
        [device.output("Da", n2), device.output("Db", n2)]
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n)]
    }

    fn run(
        &self,
        &[da, db]: &Self::Rows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        // add_scf checked that n² fits an i32.
        let n = self.n as i32;
        let d = device.get(self.d);
        let (mut da, mut db) = (device.take_mut(da), device.take_mut(db));
        // The copy follows the geam on the stream.
        geam(
            worker.blas(),
            OP_N,
            OP_N,
            n,
            n,
            0.5,
            &d,
            n,
            0.0,
            &d,
            n,
            &mut da,
            n,
        )?;
        copy::d2d(&mut db, &da, worker)
    }
}

/// CPU: the `n × n` identity into a pinned row.
struct Identity {
    n: usize,
}

impl CpuTask for Identity {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> Self::Rows {
        host.output("I host", self.n * self.n)
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n)]
    }

    fn run(&self, row: &Self::Rows, host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        let identity = host.take_mut(*row);
        identity.fill(0.0);
        identity
            .iter_mut()
            .step_by(self.n + 1)
            .for_each(|v| *v = 1.0);
        Ok(())
    }
}

/// GPU: `Fa = H + J - Ka/2`, `Fb = H + J - Kb/2`, and
/// `tr(D H + Da Fa + Db Fb)/2` down to a pinned row.
struct OpenFock {
    h: RowId<Device, f64>,
    j: RowId<Device, f64>,
    ka: RowId<Device, f64>,
    kb: RowId<Device, f64>,
    d: RowId<Device, f64>,
    da: RowId<Device, f64>,
    db: RowId<Device, f64>,
    n: usize,
}

/// The open Fock task's rows: Fa, Fb and the pinned electronic energy, and
/// the scratch `H + J` and the trace on the device.
#[derive(Clone, Copy)]
struct OpenFockRows {
    fa: RowId<Device, f64>,
    fb: RowId<Device, f64>,
    e_elec: RowId<Pinned, f64>,
    hj: RowId<Device, f64>,
    trace: RowId<Device, f64>,
}

impl GpuTask for OpenFock {
    type Rows = OpenFockRows;

    fn layout(
        &self,
        device: &mut Layout<'_, Device>,
        host: &mut Layout<'_, Pinned>,
    ) -> OpenFockRows {
        for row in [self.h, self.j, self.ka, self.kb, self.d, self.da, self.db] {
            device.read(row);
        }
        let n2 = self.n * self.n;
        OpenFockRows {
            hj: device.scratch("H + J", n2),
            trace: device.scratch("tr(D H + Da Fa + Db Fb)/2", 1),
            fa: device.output("Fa", n2),
            fb: device.output("Fb", n2),
            e_elec: host.output("E elec", 1),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n)]
    }

    fn run(
        &self,
        rows: &OpenFockRows,
        device: &Bufs<'_, Device>,
        host: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        let blas = worker.blas();
        // add_scf checked that n² fits an i32.
        let (n, n2) = (self.n as i32, (self.n * self.n) as i32);
        let [h, j, ka, kb, d, da, db] =
            [self.h, self.j, self.ka, self.kb, self.d, self.da, self.db].map(|r| device.get(r));
        let (mut hj, mut fa, mut fb, mut trace) = (
            device.take_mut(rows.hj),
            device.take_mut(rows.fa),
            device.take_mut(rows.fb),
            device.take_mut(rows.trace),
        );
        // Each call follows the one before on the handle's stream.
        geam(blas, OP_N, OP_N, n, n, 1.0, &h, n, 1.0, &j, n, &mut hj, n)?;
        geam(
            blas, OP_N, OP_N, n, n, 1.0, &hj, n, -0.5, &ka, n, &mut fa, n,
        )?;
        geam(
            blas, OP_N, OP_N, n, n, 1.0, &hj, n, -0.5, &kb, n, &mut fb, n,
        )?;
        // vec(D)ᵀ vec(M): each density as an n² × 1 matrix, transposed.
        gemv(blas, OP_T, n2, 1, 0.5, &d, n2, &h, 1, 0.0, &mut trace, 1)?;
        gemv(blas, OP_T, n2, 1, 0.5, &da, n2, &fa, 1, 1.0, &mut trace, 1)?;
        gemv(blas, OP_T, n2, 1, 0.5, &db, n2, &fb, 1, 1.0, &mut trace, 1)?;
        memcpy_d2p(host, rows.e_elec, &trace, worker)
    }
}

/// GPU: the Roothaan effective Fock matrix of Fa and Fb.
struct RoothaanFock {
    fa: RowId<Device, f64>,
    fb: RowId<Device, f64>,
    da: RowId<Device, f64>,
    db: RowId<Device, f64>,
    s: RowId<Device, f64>,
    identity: RowId<Device, f64>,
    n: usize,
}

/// The Roothaan task's rows: F, and its seven `n²` scratch matrices.
#[derive(Clone, Copy)]
struct RoothaanRows {
    f: RowId<Device, f64>,
    fc: RowId<Device, f64>,
    fo_half: RowId<Device, f64>,
    pc: RowId<Device, f64>,
    po: RowId<Device, f64>,
    pv: RowId<Device, f64>,
    a: RowId<Device, f64>,
    work: RowId<Device, f64>,
}

impl GpuTask for RoothaanFock {
    type Rows = RoothaanRows;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> RoothaanRows {
        for row in [self.fa, self.fb, self.da, self.db, self.s, self.identity] {
            device.read(row);
        }
        let n2 = self.n * self.n;
        RoothaanRows {
            fc: device.scratch("Fc", n2),
            fo_half: device.scratch("Fo/2", n2),
            pc: device.scratch("Pc", n2),
            po: device.scratch("Po", n2),
            pv: device.scratch("Pv", n2),
            a: device.scratch("A", n2),
            work: device.scratch("roothaan work", n2),
            f: device.output("F", n2),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n)]
    }

    fn run(
        &self,
        rows: &RoothaanRows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        let blas = worker.blas();
        // add_scf checked that n² fits an i32.
        let n = self.n as i32;
        let [fa, fb, da, db, s, identity] =
            [self.fa, self.fb, self.da, self.db, self.s, self.identity].map(|r| device.get(r));
        let [
            mut fc,
            mut fo_half,
            mut pc,
            mut po,
            mut pv,
            mut a,
            mut work,
            mut f,
        ] = [
            rows.fc,
            rows.fo_half,
            rows.pc,
            rows.po,
            rows.pv,
            rows.a,
            rows.work,
            rows.f,
        ]
        .map(|r| device.take_mut(r));
        let w = OPEN_BLOCK_BETA_WEIGHT;
        // Each call follows the one before on the handle's stream, so a
        // matrix written is read by the calls after it.
        geam(blas, OP_N, OP_N, n, n, 0.5, &fa, n, 0.5, &fb, n, &mut fc, n)?;
        // Fo/2, as A takes it.
        let (fa_weight, fb_weight) = (0.5 * (1.0 - w), 0.5 * w);
        geam(
            blas,
            OP_N,
            OP_N,
            n,
            n,
            fa_weight,
            &fa,
            n,
            fb_weight,
            &fb,
            n,
            &mut fo_half,
            n,
        )?;
        square(worker, OP_N, n, 1.0, &db, &s, 0.0, &mut pc)?;
        geam(
            blas, OP_N, OP_N, n, n, 1.0, &da, n, -1.0, &db, n, &mut work, n,
        )?;
        square(worker, OP_N, n, 1.0, &work, &s, 0.0, &mut po)?;
        copy::d2d(&mut pv, &identity, worker)?;
        square(worker, OP_N, n, -1.0, &da, &s, 1.0, &mut pv)?;
        // Fc Pc: A = Pcᵀ (Fc Pc)/2 + Pvᵀ (Fc Pc).
        square(worker, OP_N, n, 1.0, &fc, &pc, 0.0, &mut work)?;
        square(worker, OP_T, n, 0.5, &pc, &work, 0.0, &mut a)?;
        square(worker, OP_T, n, 1.0, &pv, &work, 1.0, &mut a)?;
        // Fc Pv: A += Pvᵀ (Fc Pv)/2.
        square(worker, OP_N, n, 1.0, &fc, &pv, 0.0, &mut work)?;
        square(worker, OP_T, n, 0.5, &pv, &work, 1.0, &mut a)?;
        // Every term with Po on the left: A += Poᵀ (Fo Po/2 + Fb Pc + Fa Pv).
        square(worker, OP_N, n, 1.0, &fo_half, &po, 0.0, &mut work)?;
        square(worker, OP_N, n, 1.0, &fb, &pc, 1.0, &mut work)?;
        square(worker, OP_N, n, 1.0, &fa, &pv, 1.0, &mut work)?;
        square(worker, OP_T, n, 1.0, &po, &work, 1.0, &mut a)?;
        // A + Aᵀ is symmetric bit for bit.
        geam(blas, OP_N, OP_T, n, n, 1.0, &a, n, 1.0, &a, n, &mut f, n)
    }
}

/// `C = alpha op(A) B + beta C` for `n × n` matrices on the worker's handle,
/// `op` the transpose or not: the Roothaan matrix's one shape of product.
#[allow(clippy::too_many_arguments, reason = "the gemm's own operands")]
fn square(
    worker: &DeviceStream,
    op: wwr_sys::wwr_blas_operation,
    n: i32,
    alpha: f64,
    a: &impl DeviceSlice<f64>,
    b: &impl DeviceSlice<f64>,
    beta: f64,
    c: &mut impl DeviceSliceMut<f64>,
) -> Status {
    gemm(
        worker.blas(),
        op,
        OP_N,
        n,
        n,
        n,
        alpha,
        a,
        n,
        b,
        n,
        beta,
        c,
        n,
    )
}

/// GPU: C's first `n_alpha` and `n_beta` columns into the carried factors,
/// their densities and `D = Da + Db`.
struct OpenDensity {
    c: RowId<Device, f64>,
    d: RowId<Device, f64>,
    da: RowId<Device, f64>,
    db: RowId<Device, f64>,
    ca: RowId<Device, f64>,
    cb: RowId<Device, f64>,
    n: usize,
    n_alpha: usize,
    n_beta: usize,
}

impl GpuTask for OpenDensity {
    type Rows = ();

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) {
        device.read(self.c);
        for row in [self.ca, self.cb, self.da, self.db, self.d] {
            device.write(row);
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("n", self.n),
            ("n_alpha", self.n_alpha),
            ("n_beta", self.n_beta),
        ]
    }

    fn run(
        &self,
        _: &(),
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        let blas = worker.blas();
        // add_scf checked that n² fits an i32, and n_beta < n_alpha <= n.
        let n = self.n as i32;
        let c = device.get(self.c);
        let [mut ca, mut cb, mut da, mut db, mut d] =
            [self.ca, self.cb, self.da, self.db, self.d].map(|r| device.take_mut(r));
        // Each call follows the one before on the handle's stream.
        for (cols, factor, density) in [
            (self.n_alpha as i32, &mut ca, &mut da),
            (self.n_beta as i32, &mut cb, &mut db),
        ] {
            geam(blas, OP_N, OP_N, n, cols, 1.0, &c, n, 0.0, &c, n, factor, n)?;
            let factor = &*factor;
            gemm(
                blas, OP_N, OP_T, n, n, cols, 1.0, factor, n, factor, n, 0.0, density, n,
            )?;
        }
        geam(blas, OP_N, OP_N, n, n, 1.0, &da, n, 1.0, &db, n, &mut d, n)
    }
}

/// GPU: the open columns of C, and `C_oᵀ Fc C_o`.
struct OpenBlock {
    c: RowId<Device, f64>,
    fa: RowId<Device, f64>,
    fb: RowId<Device, f64>,
    n: usize,
    n_beta: usize,
    m: usize,
}

/// The open block task's rows: the open columns and the block, and the
/// scratch copy of C they are cut from, Fc and `Fc C_o`.
#[derive(Clone, Copy)]
struct OpenBlockRows {
    c_open: RowId<Device, f64>,
    block: RowId<Device, f64>,
    c_copy: RowId<Device, f64>,
    fc: RowId<Device, f64>,
    fc_co: RowId<Device, f64>,
}

impl GpuTask for OpenBlock {
    type Rows = OpenBlockRows;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> OpenBlockRows {
        for row in [self.c, self.fa, self.fb] {
            device.read(row);
        }
        let (n, m) = (self.n, self.m);
        OpenBlockRows {
            c_copy: device.scratch("C copy", n * n),
            fc: device.scratch("Fc", n * n),
            fc_co: device.scratch("Fc C open", n * m),
            c_open: device.output("C open", n * m),
            block: device.output("open block", m * m),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![("n", self.n), ("n_beta", self.n_beta), ("m", self.m)]
    }

    fn run(
        &self,
        rows: &OpenBlockRows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        let blas = worker.blas();
        // add_scf checked that n² fits an i32, and m <= n.
        let (n, m) = (self.n as i32, self.m as i32);
        let [c, fa, fb] = [self.c, self.fa, self.fb].map(|r| device.get(r));
        let [mut c_copy, mut fc, mut fc_co, mut c_open, mut block] =
            [rows.c_copy, rows.fc, rows.fc_co, rows.c_open, rows.block].map(|r| device.take_mut(r));
        // A read row is not cut, so its open columns come from a copy. Each
        // call follows the one before on the stream.
        copy::d2d(&mut c_copy, &c, worker)?;
        {
            let (_, mut tail) = split_at_mut(&mut c_copy, self.n * self.n_beta);
            let (open, _) = split_at_mut(&mut tail, self.n * self.m);
            copy::d2d(&mut c_open, &open, worker)?;
        }
        geam(blas, OP_N, OP_N, n, n, 0.5, &fa, n, 0.5, &fb, n, &mut fc, n)?;
        gemm(
            blas, OP_N, OP_N, n, m, n, 1.0, &fc, n, &c_open, n, 0.0, &mut fc_co, n,
        )?;
        gemm(
            blas, OP_T, OP_N, m, m, n, 1.0, &c_open, n, &fc_co, n, 0.0, &mut block, m,
        )
    }
}

/// GPU: C and ε with the open columns rotated by U and their energies
/// replaced by w, and `D = Ca Caᵀ + Cb Cbᵀ`.
struct PublishOpen {
    c: RowId<Device, f64>,
    eps: RowId<Device, f64>,
    c_open: RowId<Device, f64>,
    u: RowId<Device, f64>,
    w: RowId<Device, f64>,
    n: usize,
    n_alpha: usize,
    n_beta: usize,
}

/// The published rows.
#[derive(Clone, Copy)]
struct PublishedRows {
    eps: RowId<Device, f64>,
    c: RowId<Device, f64>,
    d: RowId<Device, f64>,
}

impl GpuTask for PublishOpen {
    type Rows = PublishedRows;

    fn layout(&self, device: &mut Layout<'_, Device>, _: &mut Layout<'_, Pinned>) -> PublishedRows {
        for row in [self.c, self.eps, self.c_open, self.u, self.w] {
            device.read(row);
        }
        let n = self.n;
        PublishedRows {
            eps: device.output("epsilon", n),
            c: device.output("C", n * n),
            d: device.output("D", n * n),
        }
    }

    fn dims(&self) -> Vec<(&'static str, usize)> {
        vec![
            ("n", self.n),
            ("n_alpha", self.n_alpha),
            ("n_beta", self.n_beta),
        ]
    }

    fn run(
        &self,
        rows: &PublishedRows,
        device: &Bufs<'_, Device>,
        _: &CopyOnly<'_, Pinned>,
        worker: &DeviceStream,
    ) -> Status {
        let blas = worker.blas();
        // add_scf checked that n² fits an i32, and n_beta < n_alpha <= n.
        let (n, m) = (self.n as i32, (self.n_alpha - self.n_beta) as i32);
        let (open, width) = (self.n * self.n_beta, self.n_alpha - self.n_beta);
        let [c, eps, c_open, u, w] =
            [self.c, self.eps, self.c_open, self.u, self.w].map(|r| device.get(r));
        let [mut eps_out, mut c_out, mut d] =
            [rows.eps, rows.c, rows.d].map(|r| device.take_mut(r));
        // Each call follows the one before on the stream.
        copy::d2d(&mut c_out, &c, worker)?;
        {
            let (_, mut tail) = split_at_mut(&mut c_out, open);
            let (mut rotated, _) = split_at_mut(&mut tail, self.n * width);
            gemm(
                blas,
                OP_N,
                OP_N,
                n,
                m,
                m,
                1.0,
                &c_open,
                n,
                &u,
                m,
                0.0,
                &mut rotated,
                n,
            )?;
        }
        copy::d2d(&mut eps_out, &eps, worker)?;
        {
            let (_, mut tail) = split_at_mut(&mut eps_out, self.n_beta);
            let (mut energies, _) = split_at_mut(&mut tail, width);
            copy::d2d(&mut energies, &w, worker)?;
        }
        for (cols, beta) in [(self.n_alpha as i32, 0.0), (self.n_beta as i32, 1.0)] {
            gemm(
                blas, OP_N, OP_T, n, n, cols, 1.0, &c, n, &c, n, beta, &mut d, n,
            )?;
        }
        Ok(())
    }
}
