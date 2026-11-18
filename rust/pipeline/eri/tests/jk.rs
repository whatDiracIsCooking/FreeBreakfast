//! J and K over the step's B on a card (tier 4, device 0, one run each):
//! water in cc-pVDZ / cc-pVDZ-JKFIT at `schwarz_threshold = 0`, the
//! committed SAD guess D and its factor C (`D = 2 C Cᵀ`) staged and
//! uploaded, J of D and K of C built by one streamed pass of the jk
//! crate's over the packed B [`add_df_eri`](fb_pipeline_eri::add_df_eri)
//! planned, and both brought back. The references are PySCF's DF
//! `get_jk` of the same D over the same orbital and fitting documents
//! (`tests/data/reference.py`, which says how each file is made).
//!
//! And the screen at a positive threshold, on two waters 12 Å apart
//! (`tests/data/h2o_pair.xyz`), where the pairs across the gap are
//! dropped: the dropped pairs are those whose host Schwarz bound is below
//! the threshold, and their columns of the unscreened B, built on the card,
//! have 2-norm below it (the projection bound, `Σ_Q B_Q,mn² = (mn|mn)_DF
//! <= (mn|mn)`).
//!
//! **The bar** is the C++ tree's for DF against the same approximation,
//! `max |Δ| <= 1e-10 max |ref|` (`test/gtest/df_jk_derivative_gpu.cpp`,
//! `kRelTol`; `test/gtest/df_mo_gpu.cpp` holds the same). Both sides fit
//! with the same aux set and a Cholesky of the same metric, so what is left
//! is the integrals' own agreement, 1e-12 of their maximum
//! (`test/gtest/pipeline_eri_gpu.cpp` holds `BᵀB` there), carried through
//! two contractions with a positive semidefinite D: two decades of margin.
//! B itself is not compared: it is defined only up to how M is factored,
//! and J and K fix `BᵀB`.

mod common;

use std::path::Path;
use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId, memcpy_d2p, memcpy_p2d};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_integrals_eri_density_fitting_jk::{Contraction, add_streamed_jk};
use fb_integrals_eri_density_fitting_pair_map::PairMap;
use fb_integrals_mmd::record_tables;

/// The bar's fraction of `max |ref|`.
const REL_TOL: f64 = 1e-10;

/// CPU: an input written into a pinned row.
struct Stage {
    name: &'static str,
    data: Vec<f64>,
}

impl CpuTask for Stage {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, host: &mut Layout<'_, Pinned>, _: &mut Layout<'_, Heap>) -> Self::Rows {
        host.output(self.name, self.data.len())
    }

    fn run(&self, row: &Self::Rows, host: &Bufs<'_, Pinned>, _: &Bufs<'_, Heap>) -> Status {
        host.take_mut(*row).copy_from_slice(&self.data);
        Ok(())
    }
}

/// GPU: a staged input copied up into a device output.
struct Upload {
    name: &'static str,
    staged: RowId<Pinned, f64>,
    len: usize,
}

impl GpuTask for Upload {
    type Rows = RowId<Device, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        host.read(self.staged);
        device.output(self.name, self.len)
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
    let a = fb_io_npy::read_npy(Path::new(&common::data(file))).unwrap();
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

/// J of the SAD guess D and K of its factor over the step's B at
/// threshold 0, one streamed pass, against PySCF's DF `get_jk` over the
/// same documents, to 1e-10 of each one's maximum. Catches the step wiring
/// the wrong tensor or metric into the fit, the aux basis placed
/// unnormalized or off the molecule, a B laid out other than J and K read
/// it, a pair of the triangle missing, and a reader of B ordered before
/// the build. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn j_and_k_over_b_match_pyscf_df() {
    let (molecule, basis) = common::water();
    let nao = basis.nao_sphe() as usize;
    let (shape, d) = fixture("h2o_cc-pvdz_d.npy");
    assert_eq!(shape, [nao, nao], "D: not nao × nao");
    let (shape, c) = fixture("h2o_cc-pvdz_c.npy");
    assert_eq!(shape[0], nao, "C: not nao rows");
    let nocc = shape[1];
    let (_, want_j) = fixture("h2o_cc-pvdz_cc-pvdz-jkfit_j.npy");
    let (_, want_k) = fixture("h2o_cc-pvdz_cc-pvdz-jkfit_k.npy");

    let mut dag = Dag::new();
    let fitted = common::add_step(&mut dag, &molecule, &basis, common::AUX, 0.0).unwrap();
    let naux = fitted.aux.nao_sphe() as usize;
    let mut upload = |name, data: Vec<f64>| {
        let len = data.len();
        let staged = dag.add_cpu(format!("stage {name}"), 0, Stage { name, data });
        dag.add_gpu(format!("upload {name}"), 0, Upload { name, staged, len })
    };
    let (d, c) = (upload("D", d), upload("C", c));
    let set = [Contraction::Coulomb(d), Contraction::Exchange { c, nocc }];
    let jk = add_streamed_jk(
        &mut dag,
        0,
        fitted.map.sphe(),
        fitted.tables,
        naux,
        fitted.b,
        &set,
    )
    .unwrap();
    let outs: Vec<_> = [("J host", jk[0]), ("K host", jk[1])]
        .into_iter()
        .map(|(name, row)| {
            let down = dag.add_gpu(
                format!("download {name}"),
                0,
                Download {
                    name,
                    row,
                    len: nao * nao,
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
        assert_eq!(got.len(), nao * nao, "{what}: the download did not run");
        let (miss, scale) = miss_and_scale(&got, want);
        println!(
            "water cc-pVDZ / cc-pVDZ-JKFIT (nao {nao}, naux {naux}, nocc {nocc}): \
             max |Δ{what}| = {miss:e}, {:e} of max {scale:e}",
            miss / scale
        );
        assert!(
            !miss.is_nan() && miss <= REL_TOL * scale,
            "max |Δ{what}| = {miss:e} > {REL_TOL:e} · {scale:e}"
        );
    }
}

/// CPU: the heap row of B copied out, after the build's last write.
struct Keep(RowId<Heap, f64>, Arc<Mutex<Vec<f64>>>);

impl CpuTask for Keep {
    type Rows = ();

    fn layout(&self, _: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) {
        heap.read(self.0);
    }

    fn run(&self, _: &(), _: &Bufs<'_, Pinned>, heap: &Bufs<'_, Heap>) -> Status {
        *self.1.lock().unwrap() = heap.get(self.0).to_vec();
        Ok(())
    }
}

/// The screen's threshold for two waters 12 Å apart: between the pairs
/// within a molecule (bounds of 1e-3 and up) and those across the gap
/// (below 1e-15), so no pair lies near it.
const THRESHOLD: f64 = 1e-8;

/// At a positive threshold over two distant waters: the step keeps
/// exactly the pairs whose largest host Schwarz element is not below it,
/// some are dropped, and each dropped pair's columns of the unscreened B
/// (threshold 0, built on the card) have 2-norm below the threshold.
/// Catches a screen that drops a pair whose B is not negligible (the
/// bound read from the wrong block, or of the wrong pair), and a map
/// built over other bounds than the step's. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn dropped_pairs_columns_are_below_the_threshold() {
    let (molecule, basis) = common::molecule("h2o_pair.xyz");
    let tables = record_tables(&basis);
    let pairs: Vec<(u32, u32)> = tables
        .classes()
        .iter()
        .flat_map(|c| c.bra.iter().copied().zip(c.ket.iter().copied()))
        .collect();
    let q = fb_integrals_schwarz::schwarz_diagonal(&basis, &pairs);
    let shells = basis.shells();
    let mut at = 0;
    let bounds: Vec<f64> = pairs
        .iter()
        .map(|&(a, b)| {
            let n = (shells[a as usize].function_count_cart()
                * shells[b as usize].function_count_cart()) as usize;
            at += n;
            q[at - n..at].iter().copied().fold(0.0, f64::max)
        })
        .collect();

    let mut screened = Dag::new();
    let kept = common::add_step(&mut screened, &molecule, &basis, common::AUX, THRESHOLD)
        .unwrap()
        .map;
    let want: Vec<usize> = (0..pairs.len())
        .filter(|&p| bounds[p] >= THRESHOLD)
        .collect();
    let got: Vec<usize> = kept.pairs().iter().map(|k| k.pair).collect();
    assert_eq!(got, want, "the step's kept pairs are the bound's");
    assert!(got.len() < pairs.len(), "the screen drops a pair");

    let mut dag = Dag::new();
    let fitted = common::add_step(&mut dag, &molecule, &basis, common::AUX, 0.0).unwrap();
    assert_eq!(
        fitted.map,
        PairMap::new(&tables, None, 0.0),
        "every pair kept"
    );
    let out = Arc::new(Mutex::new(Vec::new()));
    dag.add_cpu("keep B", 0, Keep(fitted.b, Arc::clone(&out)));
    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = instance.backend();
    instance.run(&mut backend, 1).result.unwrap();

    let b = std::mem::take(&mut *out.lock().unwrap());
    let n_columns = fitted.map.sphe().n_columns();
    let naux = fitted.aux.nao_sphe() as usize;
    assert_eq!(b.len(), naux * n_columns, "the build did not run");
    let mut worst = 0.0_f64;
    let dropped: Vec<usize> = (0..pairs.len())
        .filter(|&p| bounds[p] < THRESHOLD)
        .collect();
    for &p in &dropped {
        let block = &fitted.map.sphe().blocks()[p];
        for column in block.column..block.end() {
            let norm = (0..naux)
                .map(|k| b[column + n_columns * k].powi(2))
                .sum::<f64>()
                .sqrt();
            assert!(
                norm < THRESHOLD,
                "pair {p}: a column of norm {norm:e} >= {THRESHOLD:e}"
            );
            worst = worst.max(norm);
        }
    }
    println!(
        "two waters 12 Å apart: {} of {} pairs dropped at {THRESHOLD:e}, largest dropped \
         column norm {worst:e}",
        dropped.len(),
        pairs.len()
    );
}
