//! J and K over the dense conventional ERI on cards: water in cc-pVDZ (an O
//! d shell, 25 Cartesian AOs to 24 spherical), the committed SAD guess D
//! staged and uploaded, the row blocks planned by
//! [`add_dense`](fb_integrals_eri_conventional::add_dense) over one, two or
//! four GPU workers (worker `w` on device `w`), J and K of D by
//! [`add_dense_jk`](fb_integrals_eri_conventional::add_dense_jk), and both
//! brought back from worker 0. The references are PySCF's exact `get_jk`
//! of the same D, the packed path's (`tests/data/reference.py`).
//!
//! **The bar, by analysis, one run: `max |Δ| <= 1e-10 max |ref|`,** the
//! packed path's (`tests/jk.rs` has the analysis): the same exact
//! integrals, now each element of J or K a sum of `n_cart² = 625` products
//! rather than `n_pp = 325`, `n_cart² ε` of the sum's scale at worst,
//! 1.4e-13. Beside it, J and K exactly symmetric and no element NaN.
//!
//! **Across workers, a tolerance by analysis: `max |Δ| <= 2 n_cart² ε max
//! |one worker's|`.** D is broadcast exactly and the gathers are exact, so
//! what can differ between worker counts is each entry's row product: a
//! GEMV row of `n_cart² = 625` terms carries a relative error of about
//! `n_cart² ε` of its scale, on each side of the comparison, so twice that,
//! 2.8e-13 (the spherical transform and the symmetrization after it are
//! the same operations on both sides, each a combination of a few
//! elements, inside it). Bitwise across worker counts is not promised
//! (`docs/testing.md`, "Determinism and placements"). Four workers over
//! water's twelve shells leave none empty; the split's evenness is not
//! what is tested.

mod common;

use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId, memcpy_p2d};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_wwr::runtime::device_count;
use fb_integrals_eri_conventional::{add_dense, add_dense_jk};

/// The bar's fraction of `max |ref|`.
const REL_TOL: f64 = 1e-10;

/// Water's Cartesian AO count in cc-pVDZ: 3 s, 2 p and 1 d on O, 2 s and 1
/// p on each H.
const N_CART: usize = 25;

/// The across-workers bar's fraction of one worker's `max |·|`: twice a
/// GEMV row's `n_cart² ε` (module docs).
const WORKERS_REL_TOL: f64 = 2.0 * (N_CART * N_CART) as f64 * f64::EPSILON;

/// CPU: an input written into a pinned row.
struct Stage {
    data: Vec<f64>,
}

impl CpuTask for Stage {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> Self::Rows {
        host.output("D staged", self.data.len())
    }

    fn run(&self, row: &Self::Rows, host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        host.take_mut(*row).copy_from_slice(&self.data);
        Ok(())
    }
}

/// GPU: the staged D copied up into a device output.
struct Upload {
    staged: RowId<Pinned, f64>,
    len: usize,
}

impl GpuTask for Upload {
    type Rows = RowId<Device, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        host.read(self.staged);
        device.output("D", self.len)
    }

    fn run(
        &self,
        row: &Self::Rows,
        device: &Bufs<'_, Device>,
        host: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        memcpy_p2d(&mut device.take_mut(*row), host, self.staged, stream)
    }
}

/// CPU: the pinned copies read out.
struct Collect {
    rows: [RowId<Pinned, f64>; 2],
    out: Arc<Mutex<Vec<Vec<f64>>>>,
}

impl CpuTask for Collect {
    type Rows = ();

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) {
        for row in self.rows {
            host.read(row);
        }
    }

    fn run(&self, _: &(), host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        *self.out.lock().unwrap() = self.rows.iter().map(|&r| host.get(r).to_vec()).collect();
        Ok(())
    }
}

/// The fixture `file` of `tests/data/`: its shape and its elements,
/// column-major.
fn fixture(file: &str) -> (Vec<usize>, Vec<f64>) {
    let a = fb_io_npy::read_npy(&common::data(file)).unwrap();
    (a.shape, a.data)
}

/// J and K of water's cc-pVDZ SAD guess over the dense ERI on `workers`
/// GPU workers, worker `w` on device `w`, and the plan's split.
fn run(workers: usize) -> (Vec<Vec<f64>>, Vec<std::ops::Range<usize>>) {
    let set = common::basis_on(&common::data("h2o.xyz"), "cc-pvdz");
    let n = set.nao_sphe() as usize;
    let (shape, d) = fixture("h2o_cc-pvdz_d.npy");
    assert_eq!(shape, [n, n], "D: not nao × nao");
    let mut dag = Dag::new();
    let eri = add_dense(&mut dag, workers, &set, &mut None).expect("the basis plans");
    let split = (0..workers).map(|w| eri.split.range(w)).collect();
    let staged = dag.add_cpu("stage D", 0, Stage { data: d });
    let d = dag.add_gpu("upload D", 0, Upload { staged, len: n * n });
    let (j, k) = add_dense_jk(&mut dag, &eri, d);
    let rows = [
        (j, "J host".to_string(), n * n),
        (k, "K host".to_string(), n * n),
    ];
    let down = dag.download_rows(0, "J and K", rows);
    let out = Arc::new(Mutex::new(Vec::new()));
    dag.add_cpu(
        "collect",
        0,
        Collect {
            rows: [down[0], down[1]],
            out: Arc::clone(&out),
        },
    );
    let planned = dag.plan();
    assert_eq!(planned.n_workers(), workers);
    let devices: Vec<i32> = (0..workers as i32).collect();
    let mut instance = planned.instantiate(&devices);
    let mut backend = instance.backend();
    instance.run(&mut backend, 1).result.unwrap();
    let got = std::mem::take(&mut *out.lock().unwrap());
    (got, split)
}

/// Each of J and K against PySCF's, to the bar, and exactly symmetric.
fn check(got: &[Vec<f64>], workers: usize) {
    let n = 24;
    for (what, (got, file)) in ["J", "K"]
        .iter()
        .zip(got.iter().zip(["h2o_cc-pvdz_j.npy", "h2o_cc-pvdz_k.npy"]))
    {
        let (_, want) = fixture(file);
        assert_eq!(got.len(), n * n, "{what}: the download did not run");
        let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let mut miss = 0.0_f64;
        for (g, w) in got.iter().zip(&want) {
            let d = (g - w).abs();
            if d.is_nan() || d > miss {
                miss = d;
            }
        }
        println!(
            "{workers} worker(s): max |Δ{what}| = {miss:e}, {:e} of max {scale:e}",
            miss / scale
        );
        assert!(
            !miss.is_nan() && miss <= REL_TOL * scale,
            "{workers} worker(s): max |Δ{what}| = {miss:e} > {REL_TOL:e} · {scale:e}"
        );
        for m in 0..n {
            for k in 0..m {
                assert_eq!(
                    got[m + n * k].to_bits(),
                    got[k + n * m].to_bits(),
                    "{what}({m}, {k}) and ({k}, {m}) differ"
                );
            }
        }
    }
}

/// Water in cc-pVDZ on one worker: every ordered class through (dd|dd) in
/// both layouts, D in and J and K out through U, and both collectives a
/// clique of one rank. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn h2o_cc_pvdz_on_one_device_matches_pyscf() {
    let (got, _) = run(1);
    check(&got, 1);
}

/// Water in cc-pVDZ on two workers, and on four where four devices are
/// visible: each against PySCF's, and one worker's J and K within
/// `2 n_cart² ε` of their scale (module docs). Needs two GPUs.
#[test]
#[ignore = "tier 5: needs two GPUs"]
fn tier5_h2o_cc_pvdz_across_devices_is_one_devices() {
    let visible = device_count().unwrap();
    assert!(visible >= 2, "{visible} devices");
    let (one, _) = run(1);
    check(&one, 1);
    let counts: &[usize] = if visible >= 4 { &[2, 4] } else { &[2] };
    for &workers in counts {
        let (got, split) = run(workers);
        println!("{workers} workers over the rows {split:?}");
        check(&got, workers);
        for (what, (g, o)) in ["J", "K"].iter().zip(got.iter().zip(&one)) {
            let scale = o.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
            let miss = g.iter().zip(o).fold(0.0_f64, |m, (a, b)| {
                if (a - b).abs() > m || (a - b).is_nan() {
                    (a - b).abs()
                } else {
                    m
                }
            });
            println!(
                "{workers} workers: max |Δ{what}| against one worker's = {miss:e} of max {scale:e}"
            );
            assert!(
                !miss.is_nan() && miss <= WORKERS_REL_TOL * scale,
                "{workers} workers: max |Δ{what}| = {miss:e} > {WORKERS_REL_TOL:e} · {scale:e}"
            );
        }
    }
}
