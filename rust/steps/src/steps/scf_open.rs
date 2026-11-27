//! The open shell's part of [`ScfStep`](crate::ScfStep): ROHF at
//! `n_alpha > n_beta`, what it changes in the closed shell's iteration
//! (the deleted `rust/pipeline/scf/src/open.rs`, the old path's nodes, and the C++
//! loop's open-shell steps before them).
//!
//! **Two densities, one J/K pass.** The carried orbitals C give the spin
//! densities `Da = Ca Caᵀ` and `Db = Cb Cbᵀ`, `Ca` C's first `n_alpha`
//! columns and `Cb` its first `n_beta`, and `D = Da + Db`. A pass is given
//! both factors at once ([`EriManager::build_jk`]), each as the
//! `D = 2 C Cᵀ` of [`Density::Occupied`], so its J and K are those of
//! `2 Da` and `2 Db`: `J(D) = (J₀ + J₁)/2` and `K(Da) = K₀/2`. With no beta
//! electron the pass is given `Ca` alone and `K(Db) = 0`. A fresh run's
//! first pass is the guess's factor once, the guess carrying no spin:
//! `Da = Db = D/2`, as PySCF's ROHF splits it, so both spin K are its K.
//!
//! **One iteration**, in place of the closed shell's F and energy:
//!
//! ```text
//! Fa = H + J - K(Da)        Fb = H + J - K(Db)
//! E  = tr(D H + Da Fa + Db Fb)/2 + E_nuc
//! F  = A + Aᵀ,  A = Pcᵀ Fc Pc/2 + Poᵀ Fo Po/2 + Pvᵀ Fc Pv/2
//!                 + Poᵀ Fb Pc + Poᵀ Fa Pv + Pvᵀ Fc Pc
//! ```
//!
//! with `Fc = (Fa + Fb)/2`, `Fo = (1 - w) Fa + w Fb` at
//! [`OPEN_BLOCK_BETA_WEIGHT`], and the projectors `Pc = Db S`,
//! `Po = (Da - Db) S`, `Pv = I - Da S`: the Roothaan effective Fock matrix,
//! PySCF's `get_roothaan_fock` but for its open-open block. The residual,
//! the DIIS step and the Roothaan step are the closed shell's on this F
//! and `D`, and the Roothaan step's orbitals give the next `Da`, `Db` and
//! `D`.
//!
//! **The published orbitals** are those of F as built, its open columns
//! `[n_beta, n_alpha)` rotated among themselves to diagonalize Fc there,
//! their energies Fc's eigenvalues in that block (the C++
//! `canonicalize_open`). At convergence F is block-diagonal across the
//! occupation classes to the residual, and PySCF's own Roothaan matrix
//! differs from it in the open-open block alone, so the energies are
//! PySCF's ROHF `mo_energy` class by class, to first order in the
//! residual; the rotation leaves `D`. The old path computed PySCF's own
//! matrix's orbitals beside them and kept those when the energies ascend,
//! a choice the step's one host read after its last enqueue cannot make;
//! the two sets differ by the convergence error alone. The effective Fock
//! matrix the step leaves ([`ScfOutput::fock`](crate::ScfOutput::fock)) is
//! PySCF's, `w = 1/2`, at the densities of the last build.

use fb_core_budget::{Layout, Lease, ScratchBuf, Span};
use fb_core_device::{
    DeviceSlice, DeviceSliceMut, GpuResources, part, part_mut, wwr_blas_operation,
};
use fb_core_error_handling::Status;

use crate::steps::GPU0;

const OP_N: wwr_blas_operation = wwr_blas_operation::WWR_BLAS_OP_N;
const OP_T: wwr_blas_operation = wwr_blas_operation::WWR_BLAS_OP_T;

/// The C++ `kOpenBlockBetaWeight`: the open-open block of the Roothaan
/// matrix the loop iterates on is `Fo = (1 - w) Fa + w Fb` at this `w`,
/// not PySCF's `Fc = (Fa + Fb)/2`. Where the open eigenvalues sit against
/// the core and virtual ones is the Roothaan step's estimate of the
/// energy's curvature along a core-open rotation (a beta electron moving,
/// coupled through Fb) and an open-virtual one (an alpha electron, through
/// Fa). On a localized hole `Fc` puts the open orbital below doubly
/// occupied ones (glycylglycine⁺ in STO-3G), so the minimum is no aufbau
/// state of PySCF's matrix and the iteration moves the hole (the C++
/// tree's issue #476); `w = 1` inverts the open-virtual gap instead (NO,
/// CH₂/cc-pVTZ). Over twenty open shells in PySCF, 0.7 and 0.8 reach
/// every minimum; 3/4 is the middle. The minimum, the energy and the
/// residual do not read the diagonal blocks.
const OPEN_BLOCK_BETA_WEIGHT: f64 = 0.75;

/// The open shell's tenants of the step's layout, after the closed
/// shell's: the spin Fock matrices and densities, the second density's J
/// and K, the identity and its pinned source, and the published
/// orbitals' open block.
pub(crate) struct OpenSpans {
    fa: Span<f64>,
    fb: Span<f64>,
    da: Span<f64>,
    db: Span<f64>,
    j2: Span<f64>,
    k2: Span<f64>,
    identity: Span<f64>,
    c_open: Span<f64>,
    fc_c_open: Span<f64>,
    block: Span<f64>,
    w: Span<f64>,
    info: Span<i32>,
    h_identity: Span<f64>,
    h_info: Span<i32>,
}

impl OpenSpans {
    /// The spans at `n` AOs and `m` singly occupied orbitals, placed after
    /// what `dev` and `pin` already hold.
    pub(crate) fn new(dev: &mut Layout, pin: &mut Layout, n: usize, m: usize) -> OpenSpans {
        let n2 = n * n;
        OpenSpans {
            fa: dev.span(n2),
            fb: dev.span(n2),
            da: dev.span(n2),
            db: dev.span(n2),
            j2: dev.span(n2),
            k2: dev.span(n2),
            identity: dev.span(n2),
            c_open: dev.span(n * m),
            fc_c_open: dev.span(n * m),
            block: dev.span(m * m),
            w: dev.span(m),
            info: dev.span(1),
            h_identity: pin.span(n2),
            h_info: pin.span(1),
        }
    }
}

/// The open shell's tenants, carved.
pub(crate) struct Open<'l> {
    pub(crate) fa: ScratchBuf<'l, f64>,
    pub(crate) fb: ScratchBuf<'l, f64>,
    pub(crate) da: ScratchBuf<'l, f64>,
    pub(crate) db: ScratchBuf<'l, f64>,
    pub(crate) j2: ScratchBuf<'l, f64>,
    pub(crate) k2: ScratchBuf<'l, f64>,
    pub(crate) identity: ScratchBuf<'l, f64>,
    pub(crate) c_open: ScratchBuf<'l, f64>,
    pub(crate) fc_c_open: ScratchBuf<'l, f64>,
    pub(crate) block: ScratchBuf<'l, f64>,
    pub(crate) w: ScratchBuf<'l, f64>,
    pub(crate) info: ScratchBuf<'l, i32>,
    /// The flag's pinned end, until the read that takes it.
    pub(crate) h_info: Option<fb_core_budget::HostPart<'l, i32>>,
}

impl<'l> Open<'l> {
    /// Carve `spans` from gpu0's lease and upload the `n × n` identity,
    /// written on the host into its pinned tenant, on gpu0's stream. The
    /// copy's token is dropped: its host end is a tenant of the lease,
    /// whose bytes are the step's until the wrapper's step-end sync
    /// (`docs/gpu_layer.md`, "The v0.0.4 runtime"), and every later read of
    /// the identity is on the same stream.
    ///
    /// # Errors
    /// The upload's enqueue.
    pub(crate) fn carve(
        gpu: &mut GpuResources,
        scratch: &'l Lease<'_>,
        spans: OpenSpans,
        n: usize,
    ) -> fb_core_error_handling::Result<Open<'l>> {
        let dev = |span| scratch.device(GPU0, span);
        let mut identity = dev(spans.identity);
        let mut h_identity = scratch.pinned(GPU0, spans.h_identity);
        h_identity.fill(0.0);
        h_identity.iter_mut().step_by(n + 1).for_each(|v| *v = 1.0);
        drop(gpu.stream().upload(&mut identity, h_identity)?);
        Ok(Open {
            fa: dev(spans.fa),
            fb: dev(spans.fb),
            da: dev(spans.da),
            db: dev(spans.db),
            j2: dev(spans.j2),
            k2: dev(spans.k2),
            identity,
            c_open: dev(spans.c_open),
            fc_c_open: dev(spans.fc_c_open),
            block: dev(spans.block),
            w: dev(spans.w),
            info: scratch.device(GPU0, spans.info),
            h_info: Some(scratch.pinned(GPU0, spans.h_info)),
        })
    }
}

/// What a J/K pass was given, and so what its J and K are.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Pass {
    /// One factor of the spinless D, both spins': `J = J₀`, `K(Da) = K(Db)
    /// = K₀/2`.
    Spinless,
    /// `Ca` alone, no beta electron: `J = J₀/2`, `K(Da) = K₀/2`, `K(Db) = 0`.
    Alpha,
    /// `Ca` and `Cb`: `J = (J₀ + J₁)/2`, `K(Da) = K₀/2`, `K(Db) = K₁/2`.
    Both,
}

/// `Fa` and `Fb` into `o.fa` and `o.fb` from the pass's J and K (`j`, `k`,
/// and `o.j2`, `o.k2` after [`Pass::Both`]), `t` scratch for `H + J`; and
/// `tr(D H + Da Fa + Db Fb)/2` into `energy[0]`, on gpu0's stream.
#[allow(
    clippy::too_many_arguments,
    reason = "the build's operands, each its own"
)]
pub(crate) fn spin_fock(
    gpu: &mut GpuResources,
    n: usize,
    pass: Pass,
    h: &impl DeviceSlice<f64>,
    (j, k): (&impl DeviceSlice<f64>, &impl DeviceSlice<f64>),
    d: &impl DeviceSlice<f64>,
    o: &mut Open<'_>,
    t: &mut impl DeviceSliceMut<f64>,
    energy: &mut impl DeviceSliceMut<f64>,
) -> Status {
    // In range: mem_budget refused an n² past i32.
    let (ni, n2i) = (n as i32, (n * n) as i32);
    // H + J into t; Fa = (H + J) - K₀/2.
    match pass {
        Pass::Spinless => gpu.geam(OP_N, OP_N, ni, ni, 1.0, h, ni, 1.0, j, ni, t, ni)?,
        Pass::Alpha => gpu.geam(OP_N, OP_N, ni, ni, 1.0, h, ni, 0.5, j, ni, t, ni)?,
        Pass::Both => {
            // H + J₀/2 through Fa, then + J₁/2.
            gpu.geam(OP_N, OP_N, ni, ni, 1.0, h, ni, 0.5, j, ni, &mut o.fa, ni)?;
            gpu.geam(OP_N, OP_N, ni, ni, 1.0, &o.fa, ni, 0.5, &o.j2, ni, t, ni)?;
        }
    }
    gpu.geam(OP_N, OP_N, ni, ni, 1.0, &*t, ni, -0.5, k, ni, &mut o.fa, ni)?;
    match pass {
        Pass::Spinless => gpu.geam(OP_N, OP_N, ni, ni, 1.0, &*t, ni, -0.5, k, ni, &mut o.fb, ni)?,
        Pass::Alpha => gpu.geam(
            OP_N, OP_N, ni, ni, 1.0, &*t, ni, 0.0, &*t, ni, &mut o.fb, ni,
        )?,
        Pass::Both => gpu.geam(
            OP_N, OP_N, ni, ni, 1.0, &*t, ni, -0.5, &o.k2, ni, &mut o.fb, ni,
        )?,
    }
    // vec(M)ᵀ vec(D) for each of the three, accumulated.
    gpu.gemv(OP_T, n2i, 1, 0.5, h, n2i, d, 0.0, energy)?;
    gpu.gemv(OP_T, n2i, 1, 0.5, &o.fa, n2i, &o.da, 1.0, energy)?;
    gpu.gemv(OP_T, n2i, 1, 0.5, &o.fb, n2i, &o.db, 1.0, energy)
}

/// The Roothaan matrix's seven `n²` scratch matrices, named for what they
/// hold once [`roothaan_fock`] has run.
pub(crate) struct RoothaanScratch<'a, M> {
    /// Fc, read again by the published orbitals' open block.
    pub(crate) fc: &'a mut M,
    pub(crate) fo_half: &'a mut M,
    pub(crate) pc: &'a mut M,
    /// Po, read again by PySCF's matrix.
    pub(crate) po: &'a mut M,
    pub(crate) pv: &'a mut M,
    pub(crate) a: &'a mut M,
    pub(crate) work: &'a mut M,
}

/// The Roothaan effective Fock matrix of `Fa` and `Fb` at the densities
/// `Da` and `Db` (`spin`, with the identity), its open block at
/// [`OPEN_BLOCK_BETA_WEIGHT`], into `f`, symmetric bit for bit (`A + Aᵀ`),
/// which the residual's one product needs: the old path's `roothaan fock`,
/// the same calls in the same order.
pub(crate) fn roothaan_fock<'a, M: DeviceSliceMut<f64> + 'a, S: DeviceSlice<f64>>(
    gpu: &mut GpuResources,
    n: usize,
    s: &impl DeviceSlice<f64>,
    [fa, fb, da, db, identity]: [&S; 5],
    r: RoothaanScratch<'a, M>,
    f: &mut impl DeviceSliceMut<f64>,
) -> Status {
    // In range: mem_budget refused an n² past i32.
    let ni = n as i32;
    let w = OPEN_BLOCK_BETA_WEIGHT;
    let RoothaanScratch {
        fc,
        fo_half,
        pc,
        po,
        pv,
        a,
        work,
    } = r;
    // Each call follows the one before on the stream.
    gpu.geam(OP_N, OP_N, ni, ni, 0.5, fa, ni, 0.5, fb, ni, fc, ni)?;
    gpu.geam(
        OP_N,
        OP_N,
        ni,
        ni,
        0.5 * (1.0 - w),
        fa,
        ni,
        0.5 * w,
        fb,
        ni,
        fo_half,
        ni,
    )?;
    square(gpu, ni, OP_N, 1.0, db, s, 0.0, pc)?;
    gpu.geam(OP_N, OP_N, ni, ni, 1.0, da, ni, -1.0, db, ni, work, ni)?;
    square(gpu, ni, OP_N, 1.0, &*work, s, 0.0, po)?;
    gpu.stream().copy_device(pv, identity)?;
    square(gpu, ni, OP_N, -1.0, da, s, 1.0, pv)?;
    // Fc Pc: A = Pcᵀ (Fc Pc)/2 + Pvᵀ (Fc Pc).
    square(gpu, ni, OP_N, 1.0, &*fc, &*pc, 0.0, work)?;
    square(gpu, ni, OP_T, 0.5, &*pc, &*work, 0.0, a)?;
    square(gpu, ni, OP_T, 1.0, &*pv, &*work, 1.0, a)?;
    // Fc Pv: A += Pvᵀ (Fc Pv)/2.
    square(gpu, ni, OP_N, 1.0, &*fc, &*pv, 0.0, work)?;
    square(gpu, ni, OP_T, 0.5, &*pv, &*work, 1.0, a)?;
    // Every term with Po on the left: A += Poᵀ (Fo Po/2 + Fb Pc + Fa Pv).
    square(gpu, ni, OP_N, 1.0, &*fo_half, &*po, 0.0, work)?;
    square(gpu, ni, OP_N, 1.0, fb, &*pc, 1.0, work)?;
    square(gpu, ni, OP_N, 1.0, fa, &*pv, 1.0, work)?;
    square(gpu, ni, OP_T, 1.0, &*po, &*work, 1.0, a)?;
    // A + Aᵀ is symmetric bit for bit.
    gpu.geam(OP_N, OP_T, ni, ni, 1.0, &*a, ni, 1.0, &*a, ni, f, ni)
}

/// `z = alpha op(x) y + beta z` for `n × n` matrices, `op` the transpose
/// or not: the Roothaan matrix's one shape of product.
#[allow(clippy::too_many_arguments, reason = "the gemm's own operands")]
fn square(
    gpu: &mut GpuResources,
    n: i32,
    op: wwr_blas_operation,
    alpha: f64,
    x: &impl DeviceSlice<f64>,
    y: &impl DeviceSlice<f64>,
    beta: f64,
    z: &mut impl DeviceSliceMut<f64>,
) -> Status {
    gpu.gemm(op, OP_N, n, n, n, alpha, x, n, y, n, beta, z, n)
}

/// PySCF's Roothaan matrix, `w = 1/2`, from the iterated one `f` built at
/// the same densities: they differ by `G + Gᵀ`, `G = Poᵀ (Fa - Fb)/8 Po`,
/// the open block's term at the two weights. Into `out`, symmetric bit for
/// bit; `t1` and `t2` scratch.
pub(crate) fn pyscf_fock(
    gpu: &mut GpuResources,
    n: usize,
    (f, po): (&impl DeviceSlice<f64>, &impl DeviceSlice<f64>),
    o: &Open<'_>,
    (t1, t2): (&mut impl DeviceSliceMut<f64>, &mut impl DeviceSliceMut<f64>),
    out: &mut impl DeviceSliceMut<f64>,
) -> Status {
    // In range: mem_budget refused an n² past i32.
    let ni = n as i32;
    let eighth = 0.5 * (0.5 - (1.0 - OPEN_BLOCK_BETA_WEIGHT));
    // (Fa - Fb)/8, its Po, Poᵀ of that: G; G + Gᵀ, then F + it.
    gpu.geam(
        OP_N, OP_N, ni, ni, eighth, &o.fa, ni, -eighth, &o.fb, ni, t1, ni,
    )?;
    gpu.gemm(OP_N, OP_N, ni, ni, ni, 1.0, &*t1, ni, po, ni, 0.0, t2, ni)?;
    gpu.gemm(OP_T, OP_N, ni, ni, ni, 1.0, po, ni, &*t2, ni, 0.0, t1, ni)?;
    gpu.geam(OP_N, OP_T, ni, ni, 1.0, &*t1, ni, 1.0, &*t1, ni, t2, ni)?;
    gpu.geam(OP_N, OP_N, ni, ni, 1.0, f, ni, 1.0, &*t2, ni, out, ni)
}

/// `Da = Ca Caᵀ` and `Db = Cb Cbᵀ` into `o.da` and `o.db` from C's first
/// `n_alpha` and `n_beta` columns, and `D = Da + Db` into `d`; with no
/// beta electron `Db = 0`.
pub(crate) fn spin_densities(
    gpu: &mut GpuResources,
    n: usize,
    (n_alpha, n_beta): (usize, usize),
    c: &impl DeviceSlice<f64>,
    o: &mut Open<'_>,
    d: &mut impl DeviceSliceMut<f64>,
) -> Status {
    // In range: mem_budget refused an n² past i32, and n_beta < n_alpha
    // <= n.
    let (ni, na, nb) = (n as i32, n_alpha as i32, n_beta as i32);
    gpu.gemm(
        OP_N, OP_T, ni, ni, na, 1.0, c, ni, c, ni, 0.0, &mut o.da, ni,
    )?;
    if n_beta == 0 {
        gpu.stream().zero(&mut o.db)?;
    } else {
        gpu.gemm(
            OP_N, OP_T, ni, ni, nb, 1.0, c, ni, c, ni, 0.0, &mut o.db, ni,
        )?;
    }
    gpu.geam(OP_N, OP_N, ni, ni, 1.0, &o.da, ni, 1.0, &o.db, ni, d, ni)
}

/// The open columns `[n_beta, n_alpha)` of `c` rotated among themselves to
/// diagonalize `fc` there, and their energies in `epsilon` replaced by its
/// eigenvalues: `C_oᵀ Fc C_o = U diag(w) Uᵀ` by syevd at `m` into
/// `o.block` and `o.w`, its flag into `o.info`, then `C_o U` and `w` into
/// place (the C++ `canonicalize_open`).
#[allow(
    clippy::too_many_arguments,
    reason = "the published orbitals' operands"
)]
pub(crate) fn canonical_open(
    gpu: &mut GpuResources,
    n: usize,
    (n_alpha, n_beta): (usize, usize),
    fc: &impl DeviceSlice<f64>,
    o: &mut Open<'_>,
    work: &mut impl DeviceSliceMut<f64>,
    c: &mut impl DeviceSliceMut<f64>,
    epsilon: &mut impl DeviceSliceMut<f64>,
) -> Status {
    let m = n_alpha - n_beta;
    // In range: mem_budget refused an n² past i32, and m <= n.
    let (ni, mi) = (n as i32, m as i32);
    gpu.stream()
        .copy_device(&mut o.c_open, &part(&*c, n * n_beta, n * m))?;
    gpu.gemm(
        OP_N,
        OP_N,
        ni,
        mi,
        ni,
        1.0,
        fc,
        ni,
        &o.c_open,
        ni,
        0.0,
        &mut o.fc_c_open,
        ni,
    )?;
    gpu.gemm(
        OP_T,
        OP_N,
        mi,
        mi,
        ni,
        1.0,
        &o.c_open,
        ni,
        &o.fc_c_open,
        ni,
        0.0,
        &mut o.block,
        mi,
    )?;
    gpu.syevd(m, &mut o.block, &mut o.w, work, &mut o.info)?;
    gpu.gemm(
        OP_N,
        OP_N,
        ni,
        mi,
        mi,
        1.0,
        &o.c_open,
        ni,
        &o.block,
        mi,
        0.0,
        &mut part_mut(c, n * n_beta, n * m),
        ni,
    )?;
    gpu.stream()
        .copy_device(&mut part_mut(epsilon, n_beta, m), &o.w)
}
