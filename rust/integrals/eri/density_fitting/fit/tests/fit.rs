//! The fit on a card (tier 4, device 0), one run per case, on the committed
//! fixtures `tests/data/*.npy` (written by `tests/data/inputs.py`, which
//! says how each is built): the metric and the tensor are staged, uploaded,
//! fitted by [`add_density_fit`], and B and L brought back by a task after
//! the fit.
//!
//! **The bar, by analysis, one run.** The triangular solve is backward
//! stable column by column: each computed column x of B satisfies
//! `(L + ΔL) x = b` with `|ΔL| <= γₙ |L|` elementwise (Higham, *Accuracy and
//! Stability of Numerical Algorithms*, 2nd ed., Theorem 8.5), so
//! `‖L x - b‖₂ <= γₙ ‖|L|‖₂ ‖x‖₂` and, with `‖x‖₂ <= ‖L⁻¹‖₂ ‖b‖₂` and
//! `‖|L|‖₂ <= √n ‖L‖₂`, the residual is at most about `n ε κ₂(L)` times b's
//! size. The test holds `max |L B - b| <= 100 n ε κ₂(L) max |b|`, n = naux:
//! two decades of margin, the norms' factors of `√n` and the host product's
//! own rounding inside it. κ₂(L) is the fixture's committed singular values'
//! ratio (100 here, cond(M) = 1e4), not the run's L: the bar is fixed before
//! the run. L itself is the device's, so the check is of the solve, whatever
//! rounding the factorization left; the factorization's own bar is
//! `fb-recipes-cholesky`'s test.
//!
//! B is not compared with a reference B: it is defined only up to how M is
//! factored, and J and K, which fix `BᵀB`, are the ERI step's test (#1516).

use std::path::Path;
use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_d2p, memcpy_p2d};
use fb_core_error_handling::{ErrorKind, Status};
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane, Record};
use fb_integrals_eri_density_fitting_fit::add_density_fit;

/// The tensor's orbital count: b is naux × nao².
const NAO: usize = 5;

/// The bar's multiple of `n ε κ₂(L) max |b|`.
const MARGIN: f64 = 100.0;

/// The tasks a failed check must leave unrun: the fit and its readers.
const AFTER_THE_CHECK: [&str; 3] = ["density fit", "download", "collect"];

/// CPU: one input written into a pinned row.
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

/// GPU: B and L copied down into pinned outputs, after the fit.
struct Download {
    b: RowId<Device, f64>,
    l: RowId<Device, f64>,
    b_len: usize,
    l_len: usize,
}

impl Task for Download {
    type Rows = (RowId<Pinned, f64>, RowId<Pinned, f64>);
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.b);
        device.read(self.l);
        (
            host.output("B down", self.b_len),
            host.output("L down", self.l_len),
        )
    }

    fn run(
        &self,
        &(b, l): &Self::Rows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        memcpy_d2p(host, b, device.get(self.b), stream)?;
        memcpy_d2p(host, l, device.get(self.l), stream)
    }
}

/// B and L as the host read them, if the download ran.
type Fitted = Option<(Vec<f64>, Vec<f64>)>;

/// Where `collect` leaves them.
type Factors = Arc<Mutex<Fitted>>;

/// CPU: the pinned copies of B and L read out.
struct Collect {
    rows: (RowId<Pinned, f64>, RowId<Pinned, f64>),
    out: Factors,
}

impl Task for Collect {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) {
        host.read(self.rows.0);
        host.read(self.rows.1);
    }

    fn run(
        &self,
        _: &(),
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        *self.out.lock().unwrap() = Some((
            host.get(self.rows.0).to_vec(),
            host.get(self.rows.1).to_vec(),
        ));
        Ok(())
    }
}

/// The fixture `stem`: its shape and its elements, column-major.
fn fixture(stem: &str) -> (Vec<usize>, Vec<f64>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/data/{stem}.npy"));
    let a = fb_io_npy::read_npy(&path).unwrap();
    (a.shape, a.data)
}

/// The fit of the metric `m` (order `naux`) and the tensor `b`, planned on
/// device 0 and run once: the run's record, and B and L if the download
/// after the fit ran.
fn fit(naux: usize, m: Vec<f64>, b: Vec<f64>) -> (Record, Fitted) {
    let b_len = b.len();
    let mut dag = Dag::new();
    let staged_m = dag.add("stage M", 0, Stage { name: "M", data: m });
    let staged_b = dag.add("stage b", 0, Stage { name: "b", data: b });
    let metric = dag.add(
        "upload M",
        0,
        Upload {
            name: "M",
            staged: staged_m,
            len: naux * naux,
        },
    );
    let tensor = dag.add(
        "upload b",
        0,
        Upload {
            name: "b",
            staged: staged_b,
            len: b_len,
        },
    );
    let (b, l) = add_density_fit(&mut dag, 0, NAO, naux, metric, tensor).unwrap();
    let rows = dag.add(
        "download",
        0,
        Download {
            b,
            l,
            b_len,
            l_len: naux * naux,
        },
    );
    let out = Arc::new(Mutex::new(None));
    dag.add(
        "collect",
        0,
        Collect {
            rows,
            out: Arc::clone(&out),
        },
    );
    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = instance.backend();
    let record = instance.run(&mut backend, 1);
    let factors = out.lock().unwrap().take();
    (record, factors)
}

/// The committed SPD metric of order 41 and random b (41 × 5²) fitted once
/// on device 0: `max |L B - b|` within `100 n ε κ₂(L) max |b|`, the solve's
/// backward error with two decades of margin, L's lower triangle alone
/// read. Catches a wrong side, triangle, transpose or leading dimension in
/// the trsm, B not copied from the tensor, or the fit run on M rather than
/// L. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn solves_within_the_backward_error() {
    let (shape, m) = fixture("m41");
    let naux = shape[0];
    assert_eq!(shape, [naux, naux], "m41: not square");
    let (shape, b) = fixture("b41x25");
    assert_eq!(shape, [naux, NAO * NAO], "b41x25: not naux × nao²");
    let (_, sigma) = fixture("sigma_l41");
    let kappa = sigma[0] / sigma[sigma.len() - 1];

    let (record, factors) = fit(naux, m, b.clone());
    record.result.unwrap();
    let (fitted, l) = factors.expect("the download ran after a passing fit");

    let max_b = b.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let bar = MARGIN * naux as f64 * f64::EPSILON * kappa * max_b;
    let mut residual = 0.0_f64;
    for col in 0..NAO * NAO {
        let x = &fitted[col * naux..(col + 1) * naux];
        for i in 0..naux {
            let lx: f64 = (0..=i).map(|k| l[i + k * naux] * x[k]).sum();
            // Not f64::max, which would drop a NaN.
            let miss = (lx - b[i + col * naux]).abs();
            if miss.is_nan() || miss > residual {
                residual = miss;
            }
        }
    }
    println!(
        "naux {naux}, nao {NAO}: κ₂(L) {kappa:e}, max |b| {max_b:e}, \
         max |L B - b| = {residual:e} (bar {bar:e})"
    );
    assert!(
        !residual.is_nan() && residual <= bar,
        "max |L B - b| = {residual:e} > {bar:e}"
    );
}

/// A metric whose first non-positive leading minor is 17 fails the run with
/// the check's `Numerical` naming it, and no task after the check runs: not
/// the fit, nor the download that reads B. Catches a fit that reads L
/// without `checked`, and so runs on a failed factor. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_metric_that_is_not_positive_definite_runs_nothing_after_the_check() {
    let (shape, m) = fixture("minor_17");
    let naux = shape[0];
    let (_, b) = fixture("b41x25");
    let (record, factors) = fit(naux, m, b);
    let e = record
        .result
        .expect_err("a metric that is not positive definite");
    assert_eq!(e.kind, ErrorKind::Numerical, "{e}");
    assert_eq!(e.message, "cholesky: the leading minor 17 is not positive");
    assert_eq!(e.context, ["cholesky check@cpu0"]);
    assert!(factors.is_none(), "a reader of B ran");
    let ran: Vec<&str> = record.tasks.iter().map(|t| t.name.as_str()).collect();
    for name in AFTER_THE_CHECK {
        assert!(
            !ran.contains(&name),
            "{name} ran after a failed check: {ran:?}"
        );
    }
}
