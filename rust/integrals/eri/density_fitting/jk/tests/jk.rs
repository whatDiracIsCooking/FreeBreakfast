//! J and K on a card (tier 4, device 0), one run per case, on the committed
//! fixtures `tests/data/*.npy` (written by `tests/data/inputs.py`, which
//! says how each is built): a random fitted factor B of naux = 23 by
//! nao² = 7², each `B_Q` symmetric, an occupied factor C of 7 × 3, and
//! `D = 2 C Cᵀ`. They are staged, uploaded, J or K built by its task, and
//! the result brought back by a task after it.
//!
//! **The bar, by analysis, one run.** Each element of J and K is a sum of
//! `naux nao²` products of B, B and D (or C, C) evaluated in two stages,
//! each a dot product of at most `max(naux, nao²)` terms, whose forward
//! error is at most `γ_k Σ|terms|` (Higham, *Accuracy and Stability of
//! Numerical Algorithms*, 2nd ed., §3.1). The test holds
//! `max |Δ| <= 100 naux nao ε max |ref|`, the reference the host's
//! `Bᵀ (B vec(D))` and `Σ_Q B_Q D B_Q`: the products' rounding, with two
//! decades of margin over the host's own, and over `Σ|terms|` against
//! `max |ref|` on a positive semidefinite D, which has no systematic
//! cancellation.

use std::path::Path;
use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_d2p, memcpy_p2d};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_integrals_eri_density_fitting_jk::{add_df_coulomb, add_df_exchange};

/// The bar's multiple of `naux nao ε max |ref|`.
const MARGIN: f64 = 100.0;

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

/// GPU: a result copied down into a pinned output.
struct Download {
    row: RowId<Device, f64>,
    len: usize,
}

impl Task for Download {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.row);
        host.output("down", self.len)
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

/// The fixture `stem`: its shape and its elements, column-major.
fn fixture(stem: &str) -> (Vec<usize>, Vec<f64>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/data/{stem}.npy"));
    let a = fb_io_npy::read_npy(&path).unwrap();
    (a.shape, a.data)
}

/// The committed fixtures and their dimensions.
struct Fixtures {
    nao: usize,
    naux: usize,
    nocc: usize,
    b: Vec<f64>,
    c: Vec<f64>,
    d: Vec<f64>,
}

/// The fixtures, their shapes checked.
fn fixtures() -> Fixtures {
    let (shape, b) = fixture("b23x49");
    let naux = shape[0];
    let (c_shape, c) = fixture("c7x3");
    let (nao, nocc) = (c_shape[0], c_shape[1]);
    assert_eq!(shape, [naux, nao * nao], "b23x49: not naux × nao²");
    let (shape, d) = fixture("d7");
    assert_eq!(shape, [nao, nao], "d7: not nao × nao");
    Fixtures {
        nao,
        naux,
        nocc,
        b,
        c,
        d,
    }
}

/// Upload `inputs` (each a name and its data) on device 0, add what `build`
/// adds over their rows, download its `len`-double result after it, run
/// the plan once and return that result.
fn run(
    inputs: Vec<(&'static str, Vec<f64>)>,
    len: usize,
    build: impl FnOnce(&mut Dag<'_>, &[RowId<Device, f64>]) -> RowId<Device, f64>,
) -> Vec<f64> {
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let rows: Vec<RowId<Device, f64>> = inputs
        .into_iter()
        .map(|(name, data)| {
            let len = data.len();
            let staged = dag.add(format!("stage {name}"), 0, Stage { name, data });
            dag.add(format!("upload {name}"), 0, Upload { name, staged, len })
        })
        .collect();
    let result = build(&mut dag, &rows);
    let down = dag.add("download", 0, Download { row: result, len });
    let out = Arc::new(Mutex::new(Vec::new()));
    dag.add(
        "collect",
        0,
        Collect {
            row: down,
            out: Arc::clone(&out),
        },
    );
    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = instance.backend();
    instance.run(&mut backend, 1).result.unwrap();
    let got = std::mem::take(&mut *out.lock().unwrap());
    assert_eq!(got.len(), len, "the download did not run");
    got
}

/// `max |got - want|`, a NaN anywhere its result, and the bar
/// `MARGIN naux nao ε max |want|`.
fn miss_and_bar(got: &[f64], want: &[f64], naux: usize, nao: usize) -> (f64, f64) {
    let max_want = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let bar = MARGIN * (naux * nao) as f64 * f64::EPSILON * max_want;
    let mut miss = 0.0_f64;
    for (g, w) in got.iter().zip(want) {
        // Not f64::max, which would drop a NaN.
        let d = (g - w).abs();
        if d.is_nan() || d > miss {
            miss = d;
        }
    }
    (miss, bar)
}

/// J of the committed D over the committed B on device 0, one run: within
/// `100 naux nao ε max |J|` of the host's `Bᵀ (B vec(D))`. Catches a gemv
/// with the wrong op, count or leading dimension, J and γ swapped, or D
/// read transposed against B's pair order. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn coulomb_is_the_hosts_within_the_products_rounding() {
    let Fixtures {
        nao, naux, b, d, ..
    } = fixtures();
    let pairs = nao * nao;
    let gamma: Vec<f64> = (0..naux)
        .map(|q| (0..pairs).map(|p| b[q + naux * p] * d[p]).sum())
        .collect();
    let want: Vec<f64> = (0..pairs)
        .map(|p| (0..naux).map(|q| b[q + naux * p] * gamma[q]).sum())
        .collect();

    let j = run(vec![("B", b), ("D", d)], pairs, |dag, rows| {
        add_df_coulomb(dag, 0, nao, naux, rows[0], rows[1])
    });
    let (miss, bar) = miss_and_bar(&j, &want, naux, nao);
    println!("naux {naux}, nao {nao}: max |ΔJ| = {miss:e} (bar {bar:e})");
    assert!(
        !miss.is_nan() && miss <= bar,
        "max |ΔJ| = {miss:e} > {bar:e}"
    );
}

/// K of `D = 2 C Cᵀ` from the committed C over the committed B on device 0,
/// one run: within `100 naux nao ε max |K|` of the host's `Σ_Q B_Q D B_Q`,
/// and `(m, n)` and `(n, m)` the same bits. Catches a batch at the wrong
/// stride (a `B_m` that is not B's block m), C not shared by the batch, a
/// syrk with the wrong op, scale or triangle, and a missing or reversed
/// mirror. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn exchange_is_the_hosts_within_the_products_rounding_and_exactly_symmetric() {
    let Fixtures {
        nao,
        naux,
        nocc,
        b,
        c,
        d,
    } = fixtures();
    // B_Q's (l, m) is B's (Q, l + nao m).
    let bq = |q: usize, l: usize, m: usize| b[q + naux * (l + nao * m)];
    let mut want = vec![0.0; nao * nao];
    for n in 0..nao {
        for m in 0..nao {
            want[m + nao * n] = (0..naux)
                .map(|q| {
                    (0..nao)
                        .map(|l| {
                            (0..nao)
                                .map(|s| bq(q, m, l) * d[l + nao * s] * bq(q, s, n))
                                .sum::<f64>()
                        })
                        .sum::<f64>()
                })
                .sum();
        }
    }

    let k = run(vec![("B", b), ("C", c)], nao * nao, |dag, rows| {
        add_df_exchange(dag, 0, nao, naux, nocc, rows[0], rows[1]).unwrap()
    });
    let (miss, bar) = miss_and_bar(&k, &want, naux, nao);
    println!("naux {naux}, nao {nao}, nocc {nocc}: max |ΔK| = {miss:e} (bar {bar:e})");
    assert!(
        !miss.is_nan() && miss <= bar,
        "max |ΔK| = {miss:e} > {bar:e}"
    );
    for n in 0..nao {
        for m in 0..n {
            let (lower, upper) = (k[n + nao * m], k[m + nao * n]);
            assert_eq!(
                lower.to_bits(),
                upper.to_bits(),
                "K({n}, {m}) = {lower:e} but K({m}, {n}) = {upper:e}"
            );
        }
    }
}
