//! J and K over the stored tensors on a card (tier 4, device 0, one run per
//! case): H₂ and water, each in STO-3G and cc-pVDZ (water's with an O d
//! shell, 25 Cartesian AOs to 24 spherical), the committed SAD guess D
//! staged and uploaded, the tensors planned by
//! [`add_conventional`](fb_integrals_eri_conventional::add_conventional),
//! J and K of D by
//! [`add_conventional_jk`](fb_integrals_eri_conventional::add_conventional_jk),
//! and both brought back. The references are PySCF's exact `get_jk` of the
//! same D over the same orbital documents, no fit (`tests/data/reference.py`,
//! which says how each file is made).
//!
//! **The bar, by analysis, one run: `max |Δ| <= 1e-10 max |ref|`.** Both
//! sides compute the same exact integrals. FelizJueves's kernels agree with
//! its host reference within 16 ε of each element's absolute-value scale
//! (its `test_kernel_eri_full.py`), libcint's with the exact integral to
//! about the same, and each element of J or K is a sum of at most
//! `n_pp = 325` products (water in cc-pVDZ) with a positive semidefinite D,
//! whose rounding is `n_pp ε` of the sum's scale at worst, 4e-14; the two
//! transforms by U add `n_cart ε` each. So what is left is about 1e-13 of
//! `max |ref|`, three decades under the bar, which is the tree's for a
//! whole path against the same approximation (the density-fitted ERI
//! step's, `fb-pipeline-eri`'s `tests/jk.rs`). Beside it, J and K exactly
//! symmetric, as the task averages them, and no element NaN.

mod common;

use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId, memcpy_d2p, memcpy_p2d};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_integrals_eri_conventional::add_conventional_jk;

/// The bar's fraction of `max |ref|`.
const REL_TOL: f64 = 1e-10;

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

/// GPU: a result copied down into a pinned output.
struct Download {
    name: &'static str,
    row: RowId<Device, f64>,
    len: usize,
}

impl GpuTask for Download {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.row);
        host.output(self.name, self.len)
    }

    fn run(
        &self,
        down: &Self::Rows,
        device: &Bufs<'_, Device>,
        host: &CopyOnly<'_, Pinned>,
        stream: &DeviceStream,
    ) -> Status {
        memcpy_d2p(host, *down, device.get(self.row), stream)
    }
}

/// CPU: the pinned copy read out.
struct Collect {
    row: RowId<Pinned, f64>,
    out: Arc<Mutex<Vec<f64>>>,
}

impl CpuTask for Collect {
    type Rows = ();

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) {
        host.read(self.row);
    }

    fn run(&self, _: &(), host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        *self.out.lock().unwrap() = host.get(self.row).to_vec();
        Ok(())
    }
}

/// The fixture `file` of `tests/data/`: its shape and its elements,
/// column-major.
fn fixture(file: &str) -> (Vec<usize>, Vec<f64>) {
    let a = fb_io_npy::read_npy(&common::data(file)).unwrap();
    (a.shape, a.data)
}

/// `max |got - want|`, a NaN anywhere its result, and `max |want|`.
fn miss_and_scale(got: &[f64], want: &[f64]) -> (f64, f64) {
    let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut miss = 0.0_f64;
    for (g, w) in got.iter().zip(want) {
        // Not f64::max, which would drop a NaN.
        let d = (g - w).abs();
        if d.is_nan() || d > miss {
            miss = d;
        }
    }
    (miss, scale)
}

/// J and K of the SAD guess of `molecule` in `basis` over the stored
/// tensors, against PySCF's exact `get_jk`, to the bar; each exactly
/// symmetric.
fn j_and_k_match_pyscf(molecule: &str, basis: &str) {
    let set = common::basis_on(&common::data(&format!("{molecule}.xyz")), basis);
    let n = set.nao_sphe() as usize;
    let stem = format!("{molecule}_{basis}");
    let (shape, d) = fixture(&format!("{stem}_d.npy"));
    assert_eq!(shape, [n, n], "D: not nao × nao");
    let (_, want_j) = fixture(&format!("{stem}_j.npy"));
    let (_, want_k) = fixture(&format!("{stem}_k.npy"));

    let mut dag = Dag::new();
    let eri = common::add_step(&mut dag, &set).expect("the basis plans");
    let staged = dag.add_cpu("stage D", 0, Stage { data: d });
    let d = dag.add_gpu("upload D", 0, Upload { staged, len: n * n });
    let (j, k) = add_conventional_jk(&mut dag, 0, &eri, d);
    let outs: Vec<_> = [("J host", j), ("K host", k)]
        .into_iter()
        .map(|(name, row)| {
            let down = dag.add_gpu(
                format!("download {name}"),
                0,
                Download {
                    name,
                    row,
                    len: n * n,
                },
            );
            let out = Arc::new(Mutex::new(Vec::new()));
            dag.add_cpu(
                format!("collect {name}"),
                0,
                Collect {
                    row: down,
                    out: Arc::clone(&out),
                },
            );
            out
        })
        .collect();
    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = instance.backend();
    instance.run(&mut backend, 1).result.unwrap();

    for ((what, out), want) in ["J", "K"].iter().zip(&outs).zip([&want_j, &want_k]) {
        let got = std::mem::take(&mut *out.lock().unwrap());
        assert_eq!(got.len(), n * n, "{what}: the download did not run");
        let (miss, scale) = miss_and_scale(&got, want);
        println!(
            "{stem} (nao {n}, cartesian {}): max |Δ{what}| = {miss:e}, {:e} of max {scale:e}",
            eri.n_cart,
            miss / scale
        );
        assert!(
            !miss.is_nan() && miss <= REL_TOL * scale,
            "{stem}: max |Δ{what}| = {miss:e} > {REL_TOL:e} · {scale:e}"
        );
        for m in 0..n {
            for k in 0..m {
                assert_eq!(
                    got[m + n * k].to_bits(),
                    got[k + n * m].to_bits(),
                    "{stem}: {what}({m}, {k}) and ({k}, {m}) differ"
                );
            }
        }
    }
}

/// H₂ in STO-3G: one (ss|ss) class, two AOs. Catches the packing of D or
/// the unpacking of J and K, the smallest case the kernels' scatter has.
/// Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn h2_sto3g_matches_pyscf() {
    j_and_k_match_pyscf("h2", "sto-3g");
}

/// H₂ in cc-pVDZ: s and p shells, every class through (pp|pp), and the
/// diagonal shell pairs whose lower triangle alone is stored. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn h2_cc_pvdz_matches_pyscf() {
    j_and_k_match_pyscf("h2", "cc-pvdz");
}

/// Water in STO-3G: two centres' shells across the tasks of one class and
/// between classes. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn h2o_sto3g_matches_pyscf() {
    j_and_k_match_pyscf("h2o", "sto-3g");
}

/// Water in cc-pVDZ, the d case: the classes through (dd|dd), 25 Cartesian
/// AOs, so D enters and J and K leave through U. Catches a Cartesian
/// order the kernels and U disagree on, and the spherical transform's
/// direction. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn h2o_cc_pvdz_matches_pyscf() {
    j_and_k_match_pyscf("h2o", "cc-pvdz");
}
