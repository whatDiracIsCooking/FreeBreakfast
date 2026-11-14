//! The orthogonalizer on a card (tier 4, device 0), one run per matrix.
//!
//! The overlap fixtures' S of order 7, 24 and 60, PySCF's, committed in
//! `integrals/one_electron/overlap/tests/data/` and read from there, each
//! staged, uploaded, orthogonalized by [`add_orthogonalize`] at the default
//! floor with [`syevd_lwork`] asked of a solver handle on the device first,
//! and X brought back. The bar is fixed by analysis, one run: the
//! eigendecomposition's backward error is a small multiple of `n ε ‖S‖₂`,
//! and `X S X = Λ^(-1/2) (Uᵀ S U) Λ^(-1/2)` amplifies it by `1 / λ_min`, so
//! `max |X S X - I| <= 100 n ε κ₂(S)` holds two decades of margin, the host
//! product's own rounding (`n ε κ₂` again) inside it. κ₂(S) is
//! `λ_max / λ_min` of LAPACK's eigenvalues, committed beside this test
//! (`tests/data/<stem>_eigenvalues.npy`, written by `tests/data/reference.py`),
//! which also show that no eigenvalue is at or below the floor, so nothing
//! is dropped.
//!
//! Then a rank-deficient S, the path graph's Laplacian of order 3, whose
//! eigenvalues are 0, 1 and 3 and whose null vector is `(1, 1, 1) / √3`:
//! its dropped mode contributes 0, so `X S X` is the projector
//! `I - J / 3` onto the kept modes, to `100 n ε` (κ₂ of the kept modes is 3,
//! inside the margin).

use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_d2p, memcpy_p2d};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{self as thread_pool, DeviceStream, Lane};
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::solver::SolverHandle;
use fb_recipes_diagonalize::syevd_lwork;
use fb_recipes_orthogonalize::{ORTHOGONALIZE_EIGENVALUE_FLOOR, add_orthogonalize};

/// The overlap's fixtures, by stem: S of order 7, 24 and 60.
const FIXTURES: [&str; 3] = ["h2o_sto-3g", "h2o_cc-pvdz", "co_cc-pvtz"];

/// CPU: S written into a pinned row.
struct Stage(Vec<f64>);

impl Task for Stage {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        host.output("S", self.0.len())
    }

    fn run(
        &self,
        s: &Self::Rows,
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        host.take_mut(*s).copy_from_slice(&self.0);
        Ok(())
    }
}

/// GPU: S, `self.1` doubles, copied up from its pinned row into a device
/// output.
struct Upload(RowId<Pinned, f64>, usize);

impl Task for Upload {
    type Rows = RowId<Device, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        host.read(self.0);
        device.output("S", self.1)
    }

    fn run(
        &self,
        s: &Self::Rows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        memcpy_p2d(&device.take_mut(*s), host, self.0, stream)
    }
}

/// GPU: X copied down into a pinned output.
struct Download(RowId<Device, f64>, usize);

impl Task for Download {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.0);
        host.output("X", self.1)
    }

    fn run(
        &self,
        x: &Self::Rows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        memcpy_d2p(host, *x, device.get(self.0), stream)
    }
}

/// CPU: the pinned copy of X read out.
struct Collect {
    x: RowId<Pinned, f64>,
    out: Arc<Mutex<Vec<f64>>>,
}

impl Task for Collect {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) {
        host.read(self.x);
    }

    fn run(
        &self,
        _: &(),
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        *self.out.lock().unwrap() = host.get(self.x).to_vec();
        Ok(())
    }
}

/// A float64 `.npy`: `(shape, data)`.
fn npy(path: &Path) -> (Vec<usize>, Vec<f64>) {
    let a = fb_io_npy::read_npy(path).unwrap();
    (a.shape, a.data)
}

/// X of the `n` × `n` S, column-major, planned on device 0 with `lwork`
/// asked of a handle there and the default floor, run once.
fn orthogonalize(n: usize, s: Vec<f64>) -> Vec<f64> {
    let lwork = {
        let solver = SolverHandle::new(Rc::new(Stream::new(0)));
        syevd_lwork(&solver, n).unwrap()
    };
    let mut dag = Dag::new();
    let staged = dag.add("stage", 0, Stage(s));
    let s = dag.add("upload", 0, Upload(staged, n * n));
    let x = add_orthogonalize(
        &mut dag,
        0,
        "X",
        n,
        lwork,
        s,
        ORTHOGONALIZE_EIGENVALUE_FLOOR,
    )
    .unwrap();
    let x = dag.add("download", 0, Download(x, n * n));
    let out = Arc::new(Mutex::new(Vec::new()));
    dag.add(
        "collect",
        0,
        Collect {
            x,
            out: Arc::clone(&out),
        },
    );
    let instance = dag.plan().instantiate(&[0]);
    let backend = thread_pool::Device::new(&[0], instance.plan());
    instance.run(&backend, 1).result.unwrap();
    std::mem::take(&mut *out.lock().unwrap())
}

/// `max |X S X - target|` on the host, both `n` × `n` column-major.
fn deviation(n: usize, x: &[f64], s: &[f64], target: impl Fn(usize, usize) -> f64) -> f64 {
    let sx: Vec<f64> = (0..n * n)
        .map(|ij| {
            let (i, j) = (ij % n, ij / n);
            (0..n).map(|k| s[i + k * n] * x[k + j * n]).sum()
        })
        .collect();
    let mut worst = 0.0_f64;
    for j in 0..n {
        for i in 0..n {
            let xsx: f64 = (0..n).map(|k| x[i + k * n] * sx[k + j * n]).sum();
            worst = worst.max((xsx - target(i, j)).abs());
        }
    }
    worst
}

/// Each fixture orthogonalized once on device 0 and held to its bar; its
/// measures are printed, and every miss named before the test fails.
/// Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn orthogonalizes_the_overlaps() {
    let data = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut misses = Vec::new();
    for stem in FIXTURES {
        let (shape, s) = npy(&data
            .join("../../integrals/one_electron/overlap/tests/data")
            .join(format!("{stem}.npy")));
        let [n, m] = shape[..] else {
            panic!("{stem}: not 2-D")
        };
        assert_eq!(n, m, "{stem}: not square");
        let (shape, w) = npy(&data.join(format!("tests/data/{stem}_eigenvalues.npy")));
        assert_eq!(shape, [n], "{stem}: the reference eigenvalues");
        assert!(
            w[0] > ORTHOGONALIZE_EIGENVALUE_FLOOR,
            "{stem}: λ_min {:e} is at or below the floor",
            w[0]
        );
        let kappa = w[n - 1] / w[0];
        let bar = 100.0 * n as f64 * f64::EPSILON * kappa;

        let x = orthogonalize(n, s.clone());
        assert_eq!(x.len(), n * n, "{stem}: X");
        let got = deviation(n, &x, &s, |i, j| if i == j { 1.0 } else { 0.0 });
        println!("{stem}: n {n}, κ₂(S) {kappa:e}, max |XSX - I| = {got:e} (bar {bar:e})");
        if got.is_nan() || got > bar {
            misses.push(format!("{stem}: max |XSX - I| = {got:e} > {bar:e}"));
        }
    }
    assert!(misses.is_empty(), "past the bar:\n{}", misses.join("\n"));
}

/// A rank-deficient S: its null mode is dropped, and `X S X` is the
/// projector onto the other two. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn drops_the_null_mode_of_a_singular_overlap() {
    const N: usize = 3;
    // The path graph's Laplacian, symmetric, so column-major is row-major.
    #[rustfmt::skip]
    let s = vec![
        1.0, -1.0, 0.0,
        -1.0, 2.0, -1.0,
        0.0, -1.0, 1.0,
    ];
    let x = orthogonalize(N, s.clone());
    assert!(x.iter().all(|v| v.is_finite()), "X is not finite: {x:?}");
    let bar = 100.0 * N as f64 * f64::EPSILON;
    let got = deviation(N, &x, &s, |i, j| f64::from(u8::from(i == j)) - 1.0 / 3.0);
    println!("singular: max |XSX - (I - J/3)| = {got:e} (bar {bar:e})");
    assert!(
        !got.is_nan() && got <= bar,
        "max |XSX - (I - J/3)| = {got:e} > {bar:e}"
    );
}
