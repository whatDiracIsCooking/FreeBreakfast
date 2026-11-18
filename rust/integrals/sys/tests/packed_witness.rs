//! The packed Cartesian (mn|P), [`add_packed_three_center_tasks`] over
//! `fb_three_center_packed_class`, against libcint's `int3c2e_cart`
//! (`tests/data/README.md`), so the packed kernels are witnessed by an
//! independent code rather than by another of the same arithmetic. Needs a
//! GPU (tier 4).
//!
//! The basis has an orbital shell of every momentum through f on each of
//! two centres, A's contracted, and an aux shell of every momentum through
//! i, the even ones on A and the odd ones on a third centre, so every one
//! of the kernels' 70 classes is launched, each with a one-centre and a
//! two-centre pair, the Boys argument zero among them. Both sides use the
//! coefficients as given on raw Cartesian Gaussians, the reference reaching
//! that normalisation through the overlap.
//!
//! The bar is `1e-11` of each class `(la, lb, lP)`'s largest
//! `|(mn|P)_ref|`, element by element. Its analysis: at L = 12, the
//! kernels' Hermite expansion and R recursion run a few hundred operations
//! deep, each near an ulp (1.1e-16), so a few times 1e-14 of the largest
//! term; the sum over the Hermite terms of mixed sign cancels by up to two
//! decades at the class's highest momenta, and libcint's Rys quadrature
//! carries its own rounding of the same order. Per class, so a small
//! high-momentum class is not measured against a large (ss|s). One run.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId, memcpy_d2p};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_integrals_sys::{
    AuxTable, ClassPairs, RecordTables, ShellTable, add_packed_three_center_tasks, add_record_tasks,
};

/// The bar on `|got - want|`, relative to the class's largest `|want|`.
const BAR: f64 = 1e-11;

/// A shell: its momentum, its centre (bohr) and its primitives, each an
/// (exponent, coefficient) pair.
type Shell = (i32, [f64; 3], Vec<(f64, f64)>);

const A: [f64; 3] = [0.0, 0.1, -0.2];
const B: [f64; 3] = [0.4, -0.9, 1.3];
const C: [f64; 3] = [-0.7, 0.5, 0.8];

/// Cartesian components of momentum `l`.
fn n_cart(l: i32) -> usize {
    ((l + 1) * (l + 2) / 2) as usize
}

/// The orbital shells: A's s, p, d, f, two primitives each, then B's, one
/// each (`tests/data/reference.py`'s `shells`).
fn shells() -> Vec<Shell> {
    let mut out = Vec::new();
    for l in 0..=3 {
        let e = 0.6 + 0.35 * f64::from(l);
        out.push((l, A, vec![(2.5 * e, 0.7), (e, 0.4)]));
    }
    for l in 0..=3 {
        let e = 0.6 + 0.35 * f64::from(l);
        out.push((l, B, vec![(1.3 * e, 0.9)]));
    }
    out
}

/// The aux shells: A's s, d, g, i, one primitive each, then C's p, f, h,
/// two each (`reference.py`'s `aux_shells`).
fn aux_shells() -> Vec<Shell> {
    let mut out = Vec::new();
    for (centre, ls) in [(A, [0, 2, 4, 6].as_slice()), (C, [1, 3, 5].as_slice())] {
        for &l in ls {
            let e = 0.9 + 0.2 * f64::from(l);
            let prims = if l % 2 == 0 {
                vec![(e, 1.0)]
            } else {
                vec![(3.0 * e, 0.6), (e, 0.5)]
            };
            out.push((l, centre, prims));
        }
    }
    out
}

/// The tables of [`shells`]: every pair `i <= j` once, the lower momentum
/// the bra (`i` on a tie), in classes `(la, lb)` ascending.
fn tables() -> RecordTables {
    let shells = shells();
    let mut t = ShellTable {
        prim_begin: vec![0],
        ..ShellTable::default()
    };
    let mut at = 0;
    let mut rank = [0; 4];
    for (l, c, prims) in &shells {
        for &(e, k) in prims {
            t.exponent.push(e);
            t.coefficient.push(k);
        }
        t.prim_begin.push(t.exponent.len() as i32);
        t.c_x.push(c[0]);
        t.c_y.push(c[1]);
        t.c_z.push(c[2]);
        t.ao_offset.push(at as i32);
        t.rank.push(rank[*l as usize]);
        rank[*l as usize] += 1;
        at += n_cart(*l);
    }
    let mut classes = Vec::new();
    for la in 0..=3 {
        for lb in la..=3 {
            let mut c = ClassPairs {
                la,
                lb,
                bra: Vec::new(),
                ket: Vec::new(),
                rec_begin: vec![0],
            };
            for i in 0..shells.len() {
                for j in i..shells.len() {
                    let (bra, ket) = if shells[i].0 <= shells[j].0 {
                        (i, j)
                    } else {
                        (j, i)
                    };
                    if (shells[bra].0, shells[ket].0) != (la, lb) {
                        continue;
                    }
                    c.bra.push(bra as u32);
                    c.ket.push(ket as u32);
                    let n = shells[bra].2.len() * shells[ket].2.len();
                    c.rec_begin.push(c.rec_begin.last().unwrap() + n as i32);
                }
            }
            classes.push(c);
        }
    }
    RecordTables::new(t, classes, at)
}

/// The aux table of [`aux_shells`], its functions in the shells' order.
fn aux() -> AuxTable {
    let shells = aux_shells();
    let mut aux = AuxTable::new(shells.iter().map(|s| n_cart(s.0)).sum());
    let mut at = 0;
    for (l, c, prims) in shells {
        aux.push_shell(l, c, at, prims);
        at += n_cart(l);
    }
    aux
}

/// GPU: a device row copied into a pinned one.
struct Download(RowId<Device, f64>, usize);
/// CPU: that pinned row read out.
struct Collect(RowId<Pinned, f64>, Arc<Mutex<Vec<f64>>>);

impl GpuTask for Download {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.0);
        host.output("host copy", self.1)
    }

    fn run(
        &self,
        pinned: &Self::Rows,
        device: &Bufs<'_, Device>,
        host: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        memcpy_d2p(host, *pinned, device.get(self.0), stream)
    }
}

impl CpuTask for Collect {
    type Rows = ();

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) {
        host.read(self.0);
    }

    fn run(&self, _: &(), host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        *self.1.lock().unwrap() = host.get(self.0).to_vec();
        Ok(())
    }
}

/// `row`, `len` doubles, brought to the host after the plan runs.
fn download(dag: &mut Dag<'_>, row: RowId<Device, f64>, len: usize) -> Arc<Mutex<Vec<f64>>> {
    let pinned = dag.add_gpu("download", 0, Download(row, len));
    let out = Arc::new(Mutex::new(Vec::new()));
    dag.add_cpu("collect", 0, Collect(pinned, Arc::clone(&out)));
    out
}

/// Every pair of `pairs`' block in `packed`, laid out from column 0,
/// against the reference's columns of that pair, each element within the
/// bar of its class's scale; the count of elements compared and the
/// largest error relative to its class's scale.
fn compare(packed: &[f64], want: &[f64], pairs: &[usize]) -> (usize, f64) {
    let t = tables();
    let aux_l: Vec<i32> = aux_shells()
        .iter()
        .flat_map(|s| std::iter::repeat_n(s.0, n_cart(s.0)))
        .collect();
    let n_aux = aux_l.len();
    // Each pair's class (la, lb) and its first column in the reference.
    let mut where_ = Vec::new();
    let mut column = 0;
    for c in t.classes() {
        for _ in 0..c.bra.len() {
            where_.push((c.la, c.lb, column));
            column += n_cart(c.la) * n_cart(c.lb);
        }
    }
    assert_eq!(want.len(), column * n_aux, "the reference's length");
    // Each class (la, lb, lP)'s largest |want|, over the whole reference.
    let mut scale = [[[0.0_f64; 7]; 4]; 4];
    for &(la, lb, first) in &where_ {
        for col in first..first + n_cart(la) * n_cart(lb) {
            for (q, &lp) in aux_l.iter().enumerate() {
                let s = &mut scale[la as usize][lb as usize][lp as usize];
                *s = s.max(want[col * n_aux + q].abs());
            }
        }
    }
    let mut at = 0;
    let (mut worst, mut misses) = (0.0_f64, Vec::new());
    for &pair in pairs {
        let (la, lb, first) = where_[pair];
        for k in 0..n_cart(la) * n_cart(lb) {
            for (q, &lp) in aux_l.iter().enumerate() {
                let got = packed[(at + k) * n_aux + q];
                let w = want[(first + k) * n_aux + q];
                let rel = (got - w).abs() / scale[la as usize][lb as usize][lp as usize];
                // Not f64::max, which would drop a NaN.
                if rel.is_nan() || rel > worst {
                    worst = rel;
                }
                if rel.is_nan() || rel > BAR {
                    misses.push(format!(
                        "pair {pair} ({la}, {lb}|{lp}), column {k}, aux {q}: {got:e} against {w:e}"
                    ));
                }
            }
        }
        at += n_cart(la) * n_cart(lb);
    }
    assert_eq!(packed.len(), at * n_aux, "the packed tensor's length");
    assert!(
        misses.is_empty(),
        "{} of {} elements past the bar {BAR:e}:\n{}",
        misses.len(),
        packed.len(),
        misses[..misses.len().min(20)].join("\n")
    );
    (packed.len(), worst)
}

/// Every pair as one batch, and a sub-range with pairs screened out (every
/// other pair of 7..31, its columns relative to its first), each element
/// within the bar of libcint's. Catches a Hermite coefficient, an R
/// recursion or a Boys order wrong in any class through (ff|i), a
/// contraction summed wrong, a block stored with `b` slower or at the
/// wrong destination, and a diagonal pair's block not stored whole. Needs
/// a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn packed_blocks_match_libcint_through_ff_i() {
    let aux = aux();
    let n_aux = aux.n_cart();
    let t = tables();
    let every: Vec<usize> = (0..t.n_pairs()).collect();
    let some: Vec<usize> = (7..31).step_by(2).collect();
    let mut sizes = Vec::new();
    for c in t.classes() {
        sizes.extend(std::iter::repeat_n(
            n_cart(c.la) * n_cart(c.lb),
            c.bra.len(),
        ));
    }
    let cols = |pairs: &[usize]| -> usize { pairs.iter().map(|&p| sizes[p]).sum() };

    let mut dag = Dag::new();
    let classes = add_record_tasks(&mut dag, 0, tables());
    let all = add_packed_three_center_tasks(&mut dag, 0, &classes, &t, &aux, &every, every.len());
    let part = add_packed_three_center_tasks(&mut dag, 0, &classes, &t, &aux, &some, some.len());
    let all = download(&mut dag, all, cols(&every) * n_aux);
    let part = download(&mut dag, part, cols(&some) * n_aux);
    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = instance.backend();
    instance.run(&mut backend, 1).result.unwrap();

    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data/packed_witness.npy");
    let want = fb_io_npy::read_npy(&path).expect("the reference reads");
    assert_eq!(want.shape, [n_aux, cols(&every)], "the reference's shape");
    let (checked, worst) = compare(&all.lock().unwrap(), &want.data, &every);
    let (part_checked, part_worst) = compare(&part.lock().unwrap(), &want.data, &some);
    println!(
        "packed against libcint: {checked} elements of every pair, max {worst:e} of the class's \
         scale; {part_checked} of the sub-range, max {part_worst:e}; bar {BAR:e}"
    );
}
