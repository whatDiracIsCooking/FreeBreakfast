//! The density-fitted integrals, as one step whose `Output` is a manager:
//! the [`EriManager`] owns the one fitted B and builds J and K over it for
//! the steps that read it.
//!
//! One step, "three-centre integrals, then the solve": the metric (P|Q),
//! its Cholesky factor, (mn|P) and the triangular solve are all inside it,
//! so what other steps see is B alone.
//!
//! **Crudest form.** B is held as the dense square, `n_ao² × n_aux`, not in
//! the packed pair layout the old tree streams: twice the bytes, and every
//! intermediate of the build stays allocated for the run. The route:
//!
//! 1. the Cartesian metric and the packed Cartesian (mn|P), by the class
//!    kernels on the step's own stream;
//! 2. on the library stream: the metric to spherical, averaged, factored
//!    `M = L Lᵀ` in place (the old tree's blocked Cholesky); (mn|P)'s aux
//!    index to spherical and `L⁻¹` applied (a trsm); the pair columns
//!    gathered into Cartesian squares and those sandwiched with U.

use std::sync::Mutex;

use fb_core_error_handling::{Error, Result, Status};
use fb_core_wwr::blas::{geam, gemm, gemm_strided_batched, gemv, syrk, trsm};
use fb_core_wwr::device_slice::{DeviceSlice, DeviceSliceMut, split_at_mut};
use fb_core_wwr::handle::DeviceHandleStream;
use fb_core_wwr::memset;
use fb_core_wwr::runtime::Stream;
use fb_integrals_sys::{fb_metric_class, fb_packed_gather, fb_three_center_packed_class};
use fb_recipes_sys::{FB_CHOLESKY_TILE, cholesky_diagonal, symmetrize};
use wwr_sys::{wwr_blas_diag_type, wwr_blas_fill_mode, wwr_blas_side_mode};

use crate::budget::{Budget, DeviceBuf, PinnedBuf};
use crate::bus::{Lane, MainBus, Shared};
use crate::device::{DeviceResources, download, launched, upload};
use crate::flow::{Next, Placement, Slot, Step, View};
use crate::steps::harmonics::HarmonicsStep;
use crate::steps::records::RecordsStep;
use crate::steps::{GPU0, OP_N, OP_T};

const LOWER: wwr_blas_fill_mode = wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER;
const NON_UNIT: wwr_blas_diag_type = wwr_blas_diag_type::WWR_BLAS_DIAG_NON_UNIT;
const BLOCK: usize = FB_CHOLESKY_TILE as usize;
/// The fields of an aux record, of a metric pair and of a packed cell.
const AUX_FIELDS: usize = 5;
const PAIR_FIELDS: usize = 6;
const CELL_FIELDS: usize = 6;

/// The step.
#[derive(Clone, Copy)]
pub struct EriStep;

/// The host tables the kernels' device tables are staged from: a function
/// of the two bases alone, so `mem_budget` sizes from them and `run` fills
/// from them.
struct Tables {
    /// `AUX_FIELDS` segments of one entry per aux primitive: x, y, z, the
    /// exponent, and `c (pi / alpha)^(3/2)`.
    aux_records: Vec<f64>,
    n_aux_rec: usize,
    /// `PAIR_FIELDS` segments per metric class, the classes concatenated.
    metric_pairs: Vec<i32>,
    /// Each metric class: `(lp, lq, first pair, pairs)`.
    metric_classes: Vec<(i32, i32, usize, usize)>,
    /// `CELL_FIELDS` segments per cell class, the classes concatenated.
    cells: Vec<i32>,
    /// Each cell class: `(orbital class, lp, first cell, cells)`.
    cell_classes: Vec<(usize, i32, usize, usize)>,
    /// The Cartesian pair columns' rows, then their columns.
    gather_at: Vec<i32>,
    /// The columns' weights: 1 on a shell's pair with itself, else 2.
    gather_weight: Vec<f64>,
}

fn n_cart_of(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

impl Tables {
    fn new(shared: &Shared) -> Tables {
        let aux = shared
            .aux
            .as_ref()
            .expect("the eri step runs with a fitting set");
        let d = shared.dims;

        // The aux shells and their records.
        let mut records: Vec<[f64; AUX_FIELDS]> = Vec::new();
        let (mut rec_begin, mut momentum, mut offset) = (vec![0i32], Vec::new(), Vec::new());
        for (s, shell) in (0u32..).zip(aux.shells()) {
            let c = shell.center();
            for p in shell.primitives() {
                let weight = p.coefficient * (std::f64::consts::PI / p.exponent).powf(1.5);
                records.push([c.x, c.y, c.z, p.exponent, weight]);
            }
            rec_begin.push(records.len() as i32);
            momentum.push(shell.angular_momentum());
            offset.push(aux.ao_offset_cart(s) as i32);
        }
        let n_aux_rec = records.len();
        let mut aux_records = vec![0.0; AUX_FIELDS * n_aux_rec];
        for (r, record) in records.iter().enumerate() {
            for (f, &v) in record.iter().enumerate() {
                aux_records[f * n_aux_rec + r] = v;
            }
        }
        let n_l = momentum.iter().copied().max().unwrap_or(0) as usize + 1;
        let field_major = |run: &[[i32; 6]], out: &mut Vec<i32>| {
            for f in 0..6 {
                out.extend(run.iter().map(|fields| fields[f]));
            }
        };

        // The metric's aux shell pairs P <= Q, by class lp <= lq.
        let n_shell = momentum.len();
        let mut by_class = vec![Vec::new(); n_l * n_l];
        for i in 0..n_shell {
            for j in i..n_shell {
                let (p, q) = if momentum[i] > momentum[j] {
                    (j, i)
                } else {
                    (i, j)
                };
                by_class[momentum[p] as usize * n_l + momentum[q] as usize].push([
                    rec_begin[p],
                    rec_begin[p + 1],
                    rec_begin[q],
                    rec_begin[q + 1],
                    offset[p],
                    offset[q],
                ]);
            }
        }
        let (mut metric_pairs, mut metric_classes) = (Vec::new(), Vec::new());
        for (class, run) in by_class.iter().enumerate() {
            if run.is_empty() {
                continue;
            }
            metric_classes.push((
                (class / n_l) as i32,
                (class % n_l) as i32,
                metric_pairs.len() / PAIR_FIELDS,
                run.len(),
            ));
            field_major(run, &mut metric_pairs);
        }

        // The packed cells: every orbital shell pair crossed with every aux
        // shell, by (orbital class, lp); and the pair columns' places.
        let mut aux_of_l = vec![Vec::new(); n_l];
        for (s, &l) in momentum.iter().enumerate() {
            aux_of_l[l as usize].push(s);
        }
        let shells = shared.tables.shells();
        let (mut cells, mut cell_classes) = (Vec::new(), Vec::new());
        let mut gather_rows = Vec::with_capacity(d.n_col);
        let mut gather_cols = Vec::with_capacity(d.n_col);
        let mut gather_weight = Vec::with_capacity(d.n_col);
        let mut column = 0usize;
        for (c, class) in shared.tables.classes().iter().enumerate() {
            let (n_a, n_b) = (n_cart_of(class.la), n_cart_of(class.lb));
            let first_column = column;
            for (&a, &b) in class.bra.iter().zip(&class.ket) {
                let (off_a, off_b) = (shells.ao_offset[a as usize], shells.ao_offset[b as usize]);
                for ia in 0..n_a as i32 {
                    for ib in 0..n_b as i32 {
                        gather_rows.push(off_a + ia);
                        gather_cols.push(off_b + ib);
                        gather_weight.push(if a == b { 1.0 } else { 2.0 });
                    }
                }
                column += n_a * n_b;
            }
            for (lp, aux_shells) in aux_of_l.iter().enumerate() {
                if aux_shells.is_empty() {
                    continue;
                }
                let mut run = Vec::with_capacity(class.bra.len() * aux_shells.len());
                for q in 0..class.bra.len() {
                    let pair_column = first_column + q * n_a * n_b;
                    for &sh in aux_shells {
                        run.push([
                            class.rec_begin[q],
                            class.rec_begin[q + 1],
                            rec_begin[sh],
                            rec_begin[sh + 1],
                            (pair_column * d.n_aux_cart) as i32 + offset[sh],
                            i32::from(class.bra[q] == class.ket[q]),
                        ]);
                    }
                }
                cell_classes.push((c, lp as i32, cells.len() / CELL_FIELDS, run.len()));
                field_major(&run, &mut cells);
            }
        }
        assert_eq!(column, d.n_col, "the columns are the dimensions' n_col");
        let mut gather_at = gather_rows;
        gather_at.extend(gather_cols);
        Tables {
            aux_records,
            n_aux_rec,
            metric_pairs,
            metric_classes,
            cells,
            cell_classes,
            gather_at,
            gather_weight,
        }
    }
}

/// A device table and the pinned copy it is staged in.
struct Staged<T> {
    device: DeviceBuf<T>,
    host: PinnedBuf<T>,
}

impl<T: Copy> Staged<T> {
    fn new(budget: &Budget<'_>, len: usize) -> Self {
        Staged {
            device: budget.device(GPU0, len),
            host: budget.pinned(GPU0, len),
        }
    }

    fn upload(&mut self, from: &[T], stream: &Stream) -> Status {
        self.host.copy_from_slice(from);
        upload(&mut self.device, &self.host, stream)
    }
}

/// What the build of B needs and nothing reads afterwards. It stays
/// allocated for the run: no bytes are reused between steps, and this step
/// reuses none within itself either.
struct Build {
    aux_records: Staged<f64>,
    metric_pairs: Staged<i32>,
    cells: Staged<i32>,
    gather_at: Staged<i32>,
    gather_weight: Staged<f64>,
    /// The fitting set's Cartesian-to-spherical transform.
    u_aux: Staged<f64>,
    metric_cart: DeviceBuf<f64>,
    metric_half: DeviceBuf<f64>,
    /// M, then its factor L in the lower triangle.
    factor: DeviceBuf<f64>,
    tile: DeviceBuf<f64>,
    flag: DeviceBuf<i32>,
    flag_host: PinnedBuf<i32>,
    /// (mn|P), `n_aux_cart × n_col`.
    three_center: DeviceBuf<f64>,
    /// `L⁻¹ U_auxᵀ (mn|P)`, `n_aux × n_col`.
    fitted: DeviceBuf<f64>,
    /// Its transpose, `n_col × n_aux`: the layout the gather takes.
    fitted_t: DeviceBuf<f64>,
    /// `n_aux` Cartesian squares.
    slab_cart: DeviceBuf<f64>,
    /// `n_aux` half-transformed `n_cart × n_ao` blocks.
    slab_half: DeviceBuf<f64>,
}

/// The scratch of one J and K build: the manager's own, behind a lock its
/// `&self` methods take.
struct JkScratch {
    gamma: DeviceBuf<f64>,
    /// `B_Q D` for every Q.
    bd: DeviceBuf<f64>,
}

/// The fitted integrals and the machinery over them. One manager, one B.
pub struct EriManager {
    n_ao: usize,
    n_aux: usize,
    /// B: `n_aux` symmetric `n_ao × n_ao` blocks, block Q at `n_ao² Q`,
    /// with `(mn|ls) ≈ Σ_Q B_Q[m, n] B_Q[l, s]`.
    b: DeviceBuf<f64>,
    scratch: Mutex<JkScratch>,
    build: Build,
}

impl EriManager {
    /// The fitting functions.
    pub fn n_aux(&self) -> usize {
        self.n_aux
    }

    /// B on the device.
    pub fn b(&self) -> &DeviceBuf<f64> {
        &self.b
    }

    /// `J[m, n] = Σ_ls (mn|ls) D[l, s]` and `K[m, n] = Σ_ls (ml|ns) D[l, s]`
    /// of the symmetric `n_ao × n_ao` density `d`, into the caller's `j` and
    /// `k`, enqueued on the caller's library handle. The manager writes its
    /// own scratch and what it is passed, nothing else.
    ///
    /// The scratch's lock is held while the calls are enqueued, not while
    /// they run: what keeps two callers' builds apart on the device is that
    /// both enqueue on the device's one library stream, in the order they
    /// took that stream's lock.
    ///
    /// # Errors
    /// A BLAS launch error.
    pub fn build_jk(
        &self,
        library: &DeviceResources,
        d: &impl DeviceSlice<f64>,
        j: &mut impl DeviceSliceMut<f64>,
        k: &mut impl DeviceSliceMut<f64>,
    ) -> Status {
        let mut scratch = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        let scratch = &mut *scratch;
        let blas = library.blas();
        let (n, q) = (self.n_ao as i32, self.n_aux as i32);
        let n2 = n * n;
        // gamma_Q = B_Q · D, J = Σ_Q gamma_Q B_Q.
        gemv(
            blas,
            OP_T,
            n2,
            q,
            1.0,
            &self.b,
            n2,
            d,
            1,
            0.0,
            &mut scratch.gamma,
            1,
        )?;
        gemv(
            blas,
            OP_N,
            n2,
            q,
            1.0,
            &self.b,
            n2,
            &scratch.gamma,
            1,
            0.0,
            j,
            1,
        )?;
        // K = Σ_Q (B_Q D) B_Qᵀ: every B_Q D in one batched gemm, then the
        // blocks side by side as one n × (n n_aux) matrix times B's.
        gemm_strided_batched(
            blas,
            OP_N,
            OP_N,
            n,
            n,
            n,
            1.0,
            &self.b,
            n,
            i64::from(n2),
            d,
            n,
            0,
            0.0,
            &mut scratch.bd,
            n,
            i64::from(n2),
            q,
        )?;
        gemm(
            blas,
            OP_N,
            OP_T,
            n,
            n,
            n * q,
            1.0,
            &scratch.bd,
            n,
            &self.b,
            n,
            0.0,
            k,
            n,
        )
    }
}

impl Step<MainBus> for EriStep {
    const NAME: &'static str = "eri";
    type Output = EriManager;
    type Reads = (RecordsStep, HarmonicsStep);

    fn slot(bus: &MainBus) -> &Slot<EriManager, Lane> {
        &bus.eri
    }

    fn placement(_: &Shared) -> Placement {
        Placement::Gpu(GPU0)
    }

    fn mem_budget(shared: &Shared, _: &Lane, budget: &Budget<'_>) -> Result<EriManager> {
        let d = shared.dims;
        let (n, n_cart, q, q_cart) = (d.n_ao, d.n_cart, d.n_aux, d.n_aux_cart);
        if (n * n * q) as u64 > i32::MAX as u64 || (d.n_col * q_cart) as u64 > i32::MAX as u64 {
            return Err(Error::unsupported(format!(
                "eri: {n} AOs with {q} fitting functions is past the dense prototype's 32-bit \
                 index; the batched path is not built"
            )));
        }
        let t = Tables::new(shared);
        let device = |len: usize| budget.device::<f64>(GPU0, len);
        let tile = q.min(BLOCK);
        Ok(EriManager {
            n_ao: n,
            n_aux: q,
            b: device(n * n * q),
            scratch: Mutex::new(JkScratch {
                gamma: device(q),
                bd: device(n * n * q),
            }),
            build: Build {
                aux_records: Staged::new(budget, t.aux_records.len()),
                metric_pairs: Staged::new(budget, t.metric_pairs.len()),
                cells: Staged::new(budget, t.cells.len()),
                gather_at: Staged::new(budget, t.gather_at.len()),
                gather_weight: Staged::new(budget, t.gather_weight.len()),
                u_aux: Staged::new(budget, q_cart * q),
                metric_cart: device(q_cart * q_cart),
                metric_half: device(q_cart * q),
                factor: device(q * q),
                tile: device(tile * tile),
                flag: budget.device(GPU0, 1),
                flag_host: budget.pinned(GPU0, 1),
                three_center: device(q_cart * d.n_col),
                fitted: device(q * d.n_col),
                fitted_t: device(d.n_col * q),
                slab_cart: device(q * n_cart * n_cart),
                slab_half: device(q * n_cart * n),
            },
        })
    }

    fn run(out: &mut EriManager, view: &View<'_, MainBus, Self>) -> Result<Next> {
        let shared = view.shared();
        let d = shared.dims;
        let lane = view.lane().gpu();
        let stream = lane.stream();
        let records = view.read(RecordsStep);
        let harmonics = view.read(HarmonicsStep);
        let aux = shared
            .aux
            .as_ref()
            .expect("the eri step runs with a fitting set");
        let t = Tables::new(shared);
        let build = &mut out.build;

        // The tables, staged and copied up on the step's own stream.
        build.aux_records.upload(&t.aux_records, stream)?;
        build.metric_pairs.upload(&t.metric_pairs, stream)?;
        build.cells.upload(&t.cells, stream)?;
        build.gather_at.upload(&t.gather_at, stream)?;
        build.gather_weight.upload(&t.gather_weight, stream)?;
        fb_basis::spherical_transform_matrix(aux, &mut build.u_aux.host);
        upload(&mut build.u_aux.device, &build.u_aux.host, stream)?;

        // The Cartesian metric (P|Q): plain stores, each with its mirror.
        let (n_rec, q_cart) = (t.n_aux_rec as i32, d.n_aux_cart as i32);
        for &(lp, lq, first, n_pair) in &t.metric_classes {
            // SAFETY: the records and pairs are this step's uploads, laid
            // out as the kernel's header says by `Tables`, every pair's
            // block inside the n_aux_cart² metric; the metric is this
            // step's, n_aux_cart² doubles by `mem_budget`.
            let status = unsafe {
                fb_metric_class(
                    stream.stream(),
                    lp,
                    lq,
                    n_rec,
                    build.aux_records.device.as_ptr(),
                    n_pair as i32,
                    build
                        .metric_pairs
                        .device
                        .as_ptr()
                        .wrapping_add(PAIR_FIELDS * first),
                    q_cart,
                    build.metric_cart.as_mut_ptr(),
                )
            };
            launched(&format!("fb_metric_class ({lp}, {lq})"), status)?;
        }

        // The packed Cartesian (mn|P): every cell stores its whole block.
        for &(class, lp, first, n_cell) in &t.cell_classes {
            let records = &records.classes()[class];
            let cls = records.view();
            // SAFETY: the class's records are the records step's, finished
            // before this step ran; the aux records and cells are this
            // step's uploads, each cell's destination inside the
            // n_aux_cart × n_col tensor by `Tables`; the tensor is this
            // step's, of that size by `mem_budget`.
            let status = unsafe {
                fb_three_center_packed_class(
                    stream.stream(),
                    &cls,
                    lp,
                    n_rec,
                    build.aux_records.device.as_ptr(),
                    n_cell as i32,
                    build
                        .cells
                        .device
                        .as_ptr()
                        .wrapping_add(CELL_FIELDS * first),
                    d.n_col as i32,
                    q_cart,
                    build.three_center.as_mut_ptr(),
                )
            };
            launched(
                &format!(
                    "fb_three_center_packed_class ({}, {}, {lp})",
                    records.la(),
                    records.lb()
                ),
                status,
            )?;
        }

        let (n, n_cart, q) = (d.n_ao as i32, d.n_cart as i32, d.n_aux as i32);
        let n_col = d.n_col as i32;
        let b = &mut out.b;
        lane.library(shared.library(GPU0), |library| {
            let blas = library.blas();
            let on = library.stream();
            // M = U_auxᵀ (P|Q) U_aux, averaged with its transpose.
            gemm(
                blas,
                OP_N,
                OP_N,
                q_cart,
                q,
                q_cart,
                1.0,
                &build.metric_cart,
                q_cart,
                &build.u_aux.device,
                q_cart,
                0.0,
                &mut build.metric_half,
                q_cart,
            )?;
            gemm(
                blas,
                OP_T,
                OP_N,
                q,
                q,
                q_cart,
                1.0,
                &build.u_aux.device,
                q_cart,
                &build.metric_half,
                q_cart,
                0.0,
                &mut build.factor,
                q,
            )?;
            symmetrize(on, d.n_aux, &mut build.factor, d.n_aux)?;

            // M = L Lᵀ in place: the old tree's blocked right-looking
            // Cholesky, its diagonal tile a kernel.
            memset::zero(&mut build.flag, on)?;
            let size = d.n_aux;
            for at in (0..size).step_by(BLOCK) {
                let jb = BLOCK.min(size - at);
                {
                    let (_, mut diagonal) = split_at_mut(&mut build.factor, at * (size + 1));
                    cholesky_diagonal(
                        on,
                        jb,
                        at,
                        &mut diagonal,
                        size,
                        &mut build.tile,
                        &mut build.flag,
                    )?;
                }
                let rest = size - at - jb;
                if rest == 0 {
                    break;
                }
                let (mut head, mut trailing) =
                    split_at_mut(&mut build.factor, (at + jb) * (size + 1));
                let (_, mut panel) = split_at_mut(&mut head, at * (size + 1) + jb);
                let (jb, rest) = (jb as i32, rest as i32);
                trsm(
                    blas,
                    wwr_blas_side_mode::WWR_BLAS_SIDE_RIGHT,
                    LOWER,
                    OP_T,
                    NON_UNIT,
                    rest,
                    jb,
                    1.0,
                    &build.tile,
                    jb,
                    &mut panel,
                    q,
                )?;
                syrk(
                    blas,
                    LOWER,
                    OP_N,
                    rest,
                    jb,
                    -1.0,
                    &panel,
                    q,
                    1.0,
                    &mut trailing,
                    q,
                )?;
            }
            download(&mut build.flag_host, &build.flag, on)?;

            // The fit: the aux index to spherical, then L⁻¹, in place.
            gemm(
                blas,
                OP_T,
                OP_N,
                q,
                n_col,
                q_cart,
                1.0,
                &build.u_aux.device,
                q_cart,
                &build.three_center,
                q_cart,
                0.0,
                &mut build.fitted,
                q,
            )?;
            trsm(
                blas,
                wwr_blas_side_mode::WWR_BLAS_SIDE_LEFT,
                LOWER,
                OP_N,
                NON_UNIT,
                q,
                n_col,
                1.0,
                &build.factor,
                q,
                &mut build.fitted,
                q,
            )?;

            // The pair columns into Cartesian squares, one per fitting
            // function, then each square sandwiched with U.
            geam(
                blas,
                OP_T,
                OP_T,
                n_col,
                q,
                1.0,
                &build.fitted,
                q,
                0.0,
                &build.fitted,
                q,
                &mut build.fitted_t,
                n_col,
            )?;
            memset::zero(&mut build.slab_cart, on)?;
            // SAFETY: the gather tables are this step's uploads, n_col rows,
            // n_col columns and n_col weights, every index inside n_cart by
            // `Tables`; the packed batch is n_col × n_aux and the slab
            // n_aux × n_cart², both this step's, by `mem_budget`.
            let status = unsafe {
                fb_packed_gather(
                    on.stream(),
                    n_cart,
                    n_col,
                    q,
                    build.gather_at.device.as_ptr(),
                    build.gather_at.device.as_ptr().wrapping_add(d.n_col),
                    build.gather_weight.device.as_ptr(),
                    build.fitted_t.as_ptr(),
                    build.slab_cart.as_mut_ptr(),
                )
            };
            launched("fb_packed_gather", status)?;
            let (cart2, half) = (i64::from(n_cart * n_cart), i64::from(n_cart * n));
            gemm_strided_batched(
                blas,
                OP_N,
                OP_N,
                n_cart,
                n,
                n_cart,
                1.0,
                &build.slab_cart,
                n_cart,
                cart2,
                harmonics.u(),
                n_cart,
                0,
                0.0,
                &mut build.slab_half,
                n_cart,
                half,
                q,
            )?;
            gemm_strided_batched(
                blas,
                OP_T,
                OP_N,
                n,
                n,
                n_cart,
                1.0,
                harmonics.u(),
                n_cart,
                0,
                &build.slab_half,
                n_cart,
                half,
                0.0,
                b,
                n,
                i64::from(n * n),
                q,
            )
        })?;

        // The factorization's flag is read on the host: the step waits.
        lane.sync()?;
        match build.flag_host[0] {
            0 => Ok(Next::Done),
            minor => Err(Error::numerical(format!(
                "eri: the metric's leading minor {minor} is not positive"
            ))),
        }
    }
}
