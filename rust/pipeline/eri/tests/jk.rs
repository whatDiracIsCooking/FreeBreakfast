//! J and K over the step's B on a card (tier 4, device 0, one run): water
//! in cc-pVDZ / cc-pVDZ-JKFIT, the committed SAD guess D and its factor C
//! (`D = 2 C Cᵀ`) staged and uploaded, J of D and K of C built by the jk
//! crate's tasks over the B [`add_df_eri`](fb_pipeline_eri::add_df_eri)
//! planned, and both brought back. The references are PySCF's DF
//! `get_jk` of the same D over the same orbital and fitting documents
//! (`tests/data/reference.py`, which says how each file is made).
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

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_d2p, memcpy_p2d};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_integrals_eri_density_fitting_jk::{add_df_coulomb, add_df_exchange};

/// The bar's fraction of `max |ref|`.
const REL_TOL: f64 = 1e-10;

/// CPU: an input written into a pinned row.
struct Stage {
    name: &'static str,
    data: Vec<f64>,
}

impl Task for Stage {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        host.output(self.name, self.data.len())
    }

    fn run(
        &self,
        row: &Self::Rows,
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
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

impl Task for Upload {
    type Rows = RowId<Device, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        host.read(self.staged);
        device.output(self.name, self.len)
    }

    fn run(
        &self,
        row: &Self::Rows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        memcpy_p2d(&device.take_mut(*row), host, self.staged, stream)
    }
}

/// GPU: a result copied down into a pinned output.
struct Download {
    name: &'static str,
    row: RowId<Device, f64>,
    len: usize,
}

impl Task for Download {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.row);
        host.output(self.name, self.len)
    }

    fn run(
        &self,
        down: &Self::Rows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        memcpy_d2p(host, *down, device.get(self.row), stream)
    }
}

/// CPU: the pinned copy read out.
struct Collect {
    row: RowId<Pinned, f64>,
    out: Arc<Mutex<Vec<f64>>>,
}

impl Task for Collect {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) {
        host.read(self.row);
    }

    fn run(
        &self,
        _: &(),
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
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

/// J of the SAD guess D and K of its factor over the step's B, against
/// PySCF's DF `get_jk` over the same documents, to 1e-10 of each one's
/// maximum. Catches the step wiring the wrong tensor or metric into the
/// fit, the aux basis placed unnormalized or off the molecule, a B laid
/// out other than J and K read it, and a reader of B ordered before the
/// fit. Needs a GPU.
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
    let fitted = common::add_step(&mut dag, &molecule, &basis, common::AUX).unwrap();
    let naux = fitted.aux.nao_sphe() as usize;
    let mut upload = |name, data: Vec<f64>| {
        let len = data.len();
        let staged = dag.add(format!("stage {name}"), 0, Stage { name, data });
        dag.add(format!("upload {name}"), 0, Upload { name, staged, len })
    };
    let (d, c) = (upload("D", d), upload("C", c));
    let j = add_df_coulomb(&mut dag, 0, nao, naux, fitted.b, d);
    let k = add_df_exchange(&mut dag, 0, nao, naux, nocc, fitted.b, c).unwrap();
    let outs: Vec<_> = [("J host", j), ("K host", k)]
        .into_iter()
        .map(|(name, row)| {
            let down = dag.add(
                format!("download {name}"),
                0,
                Download {
                    name,
                    row,
                    len: nao * nao,
                },
            );
            let out = Arc::new(Mutex::new(Vec::new()));
            dag.add(
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
