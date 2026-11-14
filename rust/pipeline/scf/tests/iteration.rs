//! One SCF iteration's dense nodes on a card (tier 4, device 0, one run per
//! case), on water in cc-pVDZ (n = 24, nocc = 5): the committed fixtures
//! `tests/data/iteration_*.npy`, PySCF's DF-RHF matrices over cc-pVDZ-JKFIT
//! at two points, the core Hamiltonian's orbitals (`start`) and the
//! converged ones, over the tree's basis documents and in its
//! normalization, written by `tests/data/iteration.py`, which says how each
//! is made. Each case stages
//! and uploads its inputs, runs the nodes and brings the results back.
//!
//! **The bars, by analysis, one run.**
//!
//! - **D** against the host's `2 C_occ C_occᵀ`: each element a dot of
//!   `nocc <= n` products, whose rounding is at most `γ_n Σ|terms|`
//!   (Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd ed.,
//!   §3.1) on both sides; `100 n ε max|D|`, two decades over it.
//! - **F** bit for bit against the host's `(H + J) - K/2`: each sum is one
//!   rounding of an exact value (`K/2` is exact), the same on either side.
//! - **E** against the fixture's, `1e-12` relative: a trace of `n²`
//!   products, `γ_{n²}` of the sum of their magnitudes, which on these
//!   fixtures is within a small multiple of `|E|`; `n² ε` is 1.3e-13.
//! - **‖e‖** against the fixture's, `1e-12 ‖F‖ ‖D‖ ‖S‖` (Frobenius norms):
//!   three products' rounding, each `γ_n` of its operands' norms, carried
//!   through `X`'s two (‖X‖₂² = 57 here): `3 n ε ‖X‖₂²` is 9.1e-13.
//! - **Converged**: at the converged fixture the norm is under ScfStep's
//!   default `commutator_tolerance`, 1e-7 (PySCF converged to
//!   `conv_tol_grad = 1e-9`; the fixture's norm is 4.9e-10).

use std::ops::ControlFlow;
use std::path::Path;
use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_d2p, memcpy_p2d};
use fb_core_error_handling::{Result, Status};
use fb_core_planner::{Dag, Decision, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_pipeline_scf::{add_density, add_fock};

/// The fixtures' basis size and occupied count.
const N: usize = 24;
const NOCC: usize = 5;
/// ScfStep's default `commutator_tolerance`.
const COMMUTATOR_TOLERANCE: f64 = 1e-7;

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

/// CPU: pinned rows read out, in order.
struct Collect {
    rows: Vec<RowId<Pinned, f64>>,
    out: Arc<Mutex<Vec<Vec<f64>>>>,
}

impl Task for Collect {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) {
        for &row in &self.rows {
            host.read(row);
        }
    }

    fn run(
        &self,
        _: &(),
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        *self.out.lock().unwrap() = self.rows.iter().map(|&r| host.get(r).to_vec()).collect();
        Ok(())
    }
}

/// The decision of a loop run once.
struct Once;

impl Decision for Once {
    type Rows = ();

    fn layout(&self, _: &mut Layout<'_, Pinned>) {}

    fn decide(&self, _: &(), _: &Bufs<'_, Pinned>, _: usize) -> Result<ControlFlow<()>> {
        Ok(ControlFlow::Break(()))
    }
}

/// The fixture `iteration_<name>.npy`, its length checked.
fn fixture(name: &str, len: usize) -> Vec<f64> {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/data/iteration_{name}.npy"));
    let data = fb_io_npy::read_npy(&path).unwrap().data;
    assert_eq!(data.len(), len, "{name}");
    data
}

/// `data` staged and uploaded into a device row named `name`.
fn upload(dag: &mut Dag<'_>, name: &'static str, data: Vec<f64>) -> RowId<Device, f64> {
    let len = data.len();
    let staged = dag.add(format!("stage {name}"), 0, Stage { name, data });
    dag.add(format!("upload {name}"), 0, Upload { name, staged, len })
}

/// Download each of `device` (with its length), collect them beside
/// `pinned`, run the plan once on device 0, and return them in that order.
fn run(
    mut dag: Dag<'_>,
    device: &[(RowId<Device, f64>, usize)],
    pinned: &[RowId<Pinned, f64>],
) -> Vec<Vec<f64>> {
    let mut rows: Vec<RowId<Pinned, f64>> = device
        .iter()
        .map(|&(row, len)| dag.add("download", 0, Download { row, len }))
        .collect();
    rows.extend_from_slice(pinned);
    let out = Arc::new(Mutex::new(Vec::new()));
    let count = rows.len();
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
    instance.run(&mut backend, 1).result.unwrap();
    let got = std::mem::take(&mut *out.lock().unwrap());
    assert_eq!(got.len(), count, "the collect did not run");
    got
}

/// `max |got - want|`, a NaN anywhere its result.
fn miss(got: &[f64], want: &[f64]) -> f64 {
    got.iter().zip(want).fold(0.0_f64, |m, (g, w)| {
        // Not f64::max, which would drop a NaN.
        let d = (g - w).abs();
        if d.is_nan() || d > m { d } else { m }
    })
}

fn frobenius(a: &[f64]) -> f64 {
    a.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// The density of the start fixture's orbitals, in a loop run once whose
/// carried D and factor start as zeros: the factor is C's first `nocc`
/// columns bit for bit, and D is within `100 n ε max|D|` of the host's
/// `2 C_occ C_occᵀ`. Catches a gemm with the wrong op or count, the factor
/// taken from the wrong columns, and a write the loop does not carry out.
/// Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn density_is_twice_the_occupied_product() {
    let c = fixture("start_c", N * N);
    let c_occ_want = c[..N * NOCC].to_vec();
    let mut want = vec![0.0; N * N];
    for col in 0..N {
        for row in 0..N {
            want[row + N * col] = 2.0
                * (0..NOCC)
                    .map(|i| c_occ_want[row + N * i] * c_occ_want[col + N * i])
                    .sum::<f64>();
        }
    }

    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let c_row = upload(&mut dag, "C", c);
    let d = upload(&mut dag, "D", vec![0.0; N * N]);
    let c_occ = upload(&mut dag, "C occ", vec![0.0; N * NOCC]);
    let body = dag.len();
    add_density(&mut dag, 0, N, NOCC, c_row, d, c_occ);
    dag.add_loop("once", 0, body, 1, Once);
    let got = run(dag, &[(d, N * N), (c_occ, N * NOCC)], &[]);

    assert_eq!(got[1], c_occ_want, "the factor is C's first columns");
    let max_d = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let bar = 100.0 * N as f64 * f64::EPSILON * max_d;
    let miss_d = miss(&got[0], &want);
    assert!(miss_d <= bar, "max |ΔD| = {miss_d:e} over the bar {bar:e}");
}

/// F, E and ‖e‖ at the fixture `point`: F bit for bit the host's
/// `(H + J) - K/2`, E within 1e-12 of the fixture's, relative, and ‖e‖
/// within `1e-12 ‖F‖ ‖D‖ ‖S‖` of it. Returns the norm.
fn check_fock(point: &str) -> f64 {
    let n2 = N * N;
    let [h, s, x] = ["h", "s", "x"].map(|m| fixture(m, n2));
    let [j, k, d] = ["j", "k", "d"].map(|m| fixture(&format!("{point}_{m}"), n2));
    let scalars = fixture(&format!("{point}_scalars"), 3);
    let (e_want, norm_want, e_nuc) = (scalars[0], scalars[1], scalars[2]);
    let f_want: Vec<f64> = (0..n2).map(|i| (h[i] + j[i]) - 0.5 * k[i]).collect();
    let bar_norm = 1e-12 * frobenius(&f_want) * frobenius(&d) * frobenius(&s);

    let mut dag = Dag::new();
    dag.blas_workspace(0);
    let rows = [("H", h), ("J", j), ("K", k), ("D", d), ("S", s), ("X", x)]
        .map(|(name, data)| upload(&mut dag, name, data));
    let (f, e, out) = add_fock(
        &mut dag, 0, N, e_nuc, rows[0], rows[1], rows[2], rows[3], rows[4], rows[5],
    );
    let got = run(dag, &[(f, n2), (e, n2)], &[out]);

    assert_eq!(got[0], f_want, "{point}: F is (H + J) - K/2");
    let (energy, norm) = (got[2][0], got[2][1]);
    assert!(
        (energy - e_want).abs() <= 1e-12 * e_want.abs(),
        "{point}: E = {energy:.15} against {e_want:.15}"
    );
    assert!(
        (norm - norm_want).abs() <= bar_norm,
        "{point}: ‖e‖ = {norm:e} against {norm_want:e}, bar {bar_norm:e}"
    );
    let norm_e = frobenius(&got[1]);
    assert!(
        (norm - norm_e).abs() <= bar_norm,
        "{point}: ‖e‖ = {norm:e}, but the e row's norm is {norm_e:e}"
    );
    norm
}

/// At the core Hamiltonian's orbitals, far from convergence (‖e‖ = 6.2):
/// catches J or K at the wrong sign or weight, the trace's halves, a
/// transposed product in the residual, or a missing X. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn fock_energy_and_residual_at_the_start() {
    check_fock("start");
}

/// At PySCF's converged orbitals: the same bars, and the norm under the
/// commutator tolerance. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_residual_vanishes_at_convergence() {
    let norm = check_fock("converged");
    assert!(
        norm < COMMUTATOR_TOLERANCE,
        "‖e‖ = {norm:e} at the converged fixture"
    );
}
