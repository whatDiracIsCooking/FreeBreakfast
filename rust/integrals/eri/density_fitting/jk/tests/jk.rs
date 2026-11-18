//! J and K on a card (tier 4, device 0), one run per case, on the committed
//! fixtures `tests/data/*.npy` (written by `tests/data/inputs.py`, which
//! says how each is built): a random fitted factor B of naux = 23 by
//! nao² = 7², each `B_Q` symmetric, an occupied factor C of 7 × 3, and
//! `D = 2 C Cᵀ`. B is packed over the spherical packing of an s and two p
//! shells (7 functions, every pair kept), losslessly since each `B_Q` is
//! symmetric, and written into a heap row; D and C are staged and uploaded;
//! J and K are built by one streamed pass ([`add_streamed_jk`]), and
//! brought back by a task after it.
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

use fb_basis::{BasisSet, Shell};
use fb_core_arena::{Bufs, CopyOnly, Device, Heap, Layout, Pinned, RowId, memcpy_d2p, memcpy_p2d};
use fb_core_error_handling::Status;
use fb_core_planner::{CpuTask, Dag, GpuTask};
use fb_core_thread_pool::DeviceStream;
use fb_core_types::Double3;
use fb_integrals_eri_density_fitting_jk::{Contraction, add_streamed_jk};
use fb_integrals_eri_density_fitting_pair_map::{Packing, PairMap};
use fb_integrals_mmd::record_tables;
use fb_integrals_sys::add_packed_tables;

/// The bar's multiple of `naux nao ε max |ref|`.
const MARGIN: f64 = 100.0;

/// CPU: the packed B written into a heap row.
struct Source(Vec<f64>);

impl CpuTask for Source {
    type Rows = RowId<Heap, f64>;

    fn layout(&self, _: &mut Layout<'_, Pinned>, heap: &mut Layout<'_, Heap>) -> Self::Rows {
        heap.output("B", self.0.len())
    }

    fn run(&self, row: &Self::Rows, _: &Bufs<'_, Pinned>, heap: &Bufs<'_, Heap>) -> Status {
        heap.take_mut(*row).copy_from_slice(&self.0);
        Ok(())
    }
}

/// CPU: one input written into a pinned row.
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
    row: RowId<Device, f64>,
    len: usize,
}

impl GpuTask for Download {
    type Rows = RowId<Pinned, f64>;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.row);
        host.output("down", self.len)
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

/// The spherical packing of an s and two p shells, 7 functions, every
/// pair kept: the fixtures' `nao`.
fn packing() -> Packing {
    let at = |z| Double3 { x: 0.0, y: 0.0, z };
    let basis = BasisSet::new(vec![
        Shell::zipped(0, &[1.0], &[1.0], at(0.0)),
        Shell::zipped(1, &[0.7], &[1.0], at(0.0)),
        Shell::zipped(1, &[0.5], &[1.0], at(1.5)),
    ]);
    PairMap::new(&record_tables(&basis), None, 0.0)
        .sphe()
        .clone()
}

/// The fixtures' B packed over `packing` into B's heap layout, element
/// `(column, Q)` at `column + n_columns Q`: each `B_Q` is symmetric, so
/// the pack keeps every element.
fn packed(f: &Fixtures, packing: &Packing) -> Vec<f64> {
    let (nao, nc) = (f.nao, packing.n_columns());
    let mut out = vec![0.0; f.naux * nc];
    let mut bq = vec![0.0; nao * nao];
    for (q, column) in out.chunks_mut(nc).enumerate() {
        for (mn, v) in bq.iter_mut().enumerate() {
            *v = f.b[q + f.naux * mn];
        }
        packing.pack(&bq, column);
    }
    out
}

/// J of the fixtures' D and K of their C by one streamed pass over their
/// B, packed into a heap row, on device 0, run once: `[J, K]`, `nao²`
/// doubles each.
fn run(f: &Fixtures) -> [Vec<f64>; 2] {
    let packing = packing();
    let (nao, nc) = (packing.nao(), packing.n_columns());
    assert_eq!(nao, f.nao, "the packing is the fixtures'");
    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let b = dag.add_cpu("source", 0, Source(packed(f, &packing)));
    let p = packing.clone();
    let tables = add_packed_tables(&mut dag, 0, nao, nc, move |r, c, w| p.write_tables(r, c, w));
    let mut upload = |name: &'static str, data: &[f64]| {
        let (len, data) = (data.len(), data.to_vec());
        let staged = dag.add_cpu(format!("stage {name}"), 0, Stage { name, data });
        dag.add_gpu(format!("upload {name}"), 0, Upload { name, staged, len })
    };
    let (d, c) = (upload("D", &f.d), upload("C", &f.c));
    let set = [
        Contraction::Coulomb(d),
        Contraction::Exchange { c, nocc: f.nocc },
    ];
    let jk = add_streamed_jk(&mut dag, 0, &packing, tables, f.naux, b, &set).unwrap();
    let outs: [_; 2] = [jk[0], jk[1]].map(|row| {
        let down = dag.add_gpu(
            "download",
            0,
            Download {
                row,
                len: nao * nao,
            },
        );
        let out = Arc::new(Mutex::new(Vec::new()));
        dag.add_cpu(
            "collect",
            0,
            Collect {
                row: down,
                out: Arc::clone(&out),
            },
        );
        out
    });
    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = instance.backend();
    instance.run(&mut backend, 1).result.unwrap();
    outs.map(|out| {
        let got = std::mem::take(&mut *out.lock().unwrap());
        assert_eq!(got.len(), nao * nao, "the download did not run");
        got
    })
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

/// J of the committed D over the committed B on device 0, one streamed
/// pass: within `100 naux nao ε max |J|` of the host's `Bᵀ (B vec(D))`.
/// Catches a gemv with the wrong op, count or leading dimension, D packed
/// against other weights than B's columns, or J unpacked off its pairs.
/// Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn coulomb_is_the_hosts_within_the_products_rounding() {
    let f = fixtures();
    let Fixtures {
        nao, naux, b, d, ..
    } = &f;
    let (nao, naux) = (*nao, *naux);
    let pairs = nao * nao;
    let gamma: Vec<f64> = (0..naux)
        .map(|q| (0..pairs).map(|p| b[q + naux * p] * d[p]).sum())
        .collect();
    let want: Vec<f64> = (0..pairs)
        .map(|p| (0..naux).map(|q| b[q + naux * p] * gamma[q]).sum())
        .collect();

    let [j, _] = run(&f);
    let (miss, bar) = miss_and_bar(&j, &want, naux, nao);
    println!("naux {naux}, nao {nao}: max |ΔJ| = {miss:e} (bar {bar:e})");
    assert!(
        !miss.is_nan() && miss <= bar,
        "max |ΔJ| = {miss:e} > {bar:e}"
    );
}

/// K of `D = 2 C Cᵀ` from the committed C over the committed B on device 0,
/// one streamed pass: within `100 naux nao ε max |K|` of the host's
/// `Σ_Q B_Q D B_Q`, and `(m, n)` and `(n, m)` the same bits. Catches the
/// gather's slab at the wrong stride, C not shared by the batched gemm, a
/// syrk with the wrong op, scale or triangle, and a missing or reversed
/// mirror. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn exchange_is_the_hosts_within_the_products_rounding_and_exactly_symmetric() {
    let f = fixtures();
    let Fixtures {
        nao,
        naux,
        nocc,
        b,
        d,
        ..
    } = &f;
    let (nao, naux, nocc) = (*nao, *naux, *nocc);
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

    let [_, k] = run(&f);
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
