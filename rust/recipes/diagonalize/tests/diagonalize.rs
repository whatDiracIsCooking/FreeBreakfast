//! The eigendecomposition on a card (tier 4, device 0), one run per matrix:
//! the overlap fixtures' S of order 7, 24 and 60, PySCF's, committed in
//! `integrals/one_electron/overlap/tests/data/` and read from there.
//!
//! Each is staged, uploaded, diagonalized by [`add_diagonalize`] with
//! [`syevd_lwork`] asked of a solver handle on the device first, and U and
//! w brought back. The bars are fixed by analysis, one run: syevd is
//! backward stable, its residual and its loss of orthogonality a small
//! multiple of `n ε ‖A‖₂` and `n ε` (1.3e-14 at n = 60), so
//! `max |A U - U diag(w)| <= 1e-12 ‖A‖₂` and `max |Uᵀ U - I| <= 1e-12` hold
//! two decades of margin, the host products' own rounding (`n ε` again)
//! inside it. `‖A‖₂` is `max |w|`, A being symmetric. w must be ascending.
//! The eigenvalues are not compared with LAPACK's: the residual, the
//! orthogonality and the order together imply them.

use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_d2p, memcpy_p2d};
use fb_core_error_handling::Status;
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{self as thread_pool, DeviceStream, Lane};
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::solver::SolverHandle;
use fb_recipes_diagonalize::{add_diagonalize, syevd_lwork};

/// The residual's bar, relative to `‖A‖₂`, and the orthogonality's.
const BAR: f64 = 1e-12;

/// The overlap's fixtures, by stem: S of order 7, 24 and 60.
const FIXTURES: [&str; 3] = ["h2o_sto-3g", "h2o_cc-pvdz", "co_cc-pvtz"];

/// CPU: A written into a pinned row.
struct Stage(Vec<f64>);

impl Task for Stage {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        host.output("A", self.0.len())
    }

    fn run(
        &self,
        a: &Self::Rows,
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        host.take_mut(*a).copy_from_slice(&self.0);
        Ok(())
    }
}

/// GPU: one copy between a pinned and a device row of the same length, on
/// the stream: up when `up`, into a device output, else down, into a
/// pinned output.
struct Transfer {
    pinned: Option<RowId<Pinned, f64>>,
    device: Option<RowId<Device, f64>>,
    len: usize,
    name: &'static str,
}

#[derive(Clone, Copy)]
enum Copied {
    Up(RowId<Device, f64>),
    Down(RowId<Pinned, f64>),
}

impl Task for Transfer {
    type Rows = Copied;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Copied {
        match (self.pinned, self.device) {
            (Some(p), None) => {
                host.read(p);
                Copied::Up(device.output(self.name, self.len))
            }
            (None, Some(d)) => {
                device.read(d);
                Copied::Down(host.output(self.name, self.len))
            }
            _ => unreachable!("one source"),
        }
    }

    fn run(
        &self,
        rows: &Copied,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        match *rows {
            Copied::Up(d) => memcpy_p2d(&device.take_mut(d), host, self.pinned.unwrap(), stream),
            Copied::Down(p) => memcpy_d2p(host, p, device.get(self.device.unwrap()), stream),
        }
    }
}

/// CPU: the pinned copies of U and w read out.
struct Collect {
    u: RowId<Pinned, f64>,
    w: RowId<Pinned, f64>,
    out: Arc<Mutex<(Vec<f64>, Vec<f64>)>>,
}

impl Task for Collect {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) {
        host.read(self.u);
        host.read(self.w);
    }

    fn run(
        &self,
        _: &(),
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        *self.out.lock().unwrap() = (host.get(self.u).to_vec(), host.get(self.w).to_vec());
        Ok(())
    }
}

/// `(n, A)` of a fixture, column-major.
fn fixture(stem: &str) -> (usize, Vec<f64>) {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../integrals/one_electron/overlap/tests/data")
        .join(format!("{stem}.npy"));
    let a = fb_io_npy::read_npy(&path).unwrap();
    let [n, m] = a.shape[..] else {
        panic!("{stem}: not 2-D")
    };
    assert_eq!(n, m, "{stem}: not square");
    (n, a.data)
}

/// A planned on device 0 with `lwork` asked of a handle there, run once:
/// `(lwork, U, w)`.
fn diagonalize(n: usize, a: Vec<f64>) -> (usize, Vec<f64>, Vec<f64>) {
    let lwork = {
        let solver = SolverHandle::new(Rc::new(Stream::new(0)));
        syevd_lwork(&solver, n).unwrap()
    };
    let mut dag = Dag::new();
    let staged = dag.add("stage", 0, Stage(a));
    let Copied::Up(a) = dag.add(
        "upload",
        0,
        Transfer {
            pinned: Some(staged),
            device: None,
            len: n * n,
            name: "A",
        },
    ) else {
        unreachable!()
    };
    let (u, w) = add_diagonalize(&mut dag, 0, "A", n, lwork, a);
    let mut down = |device, len, name| {
        let Copied::Down(p) = dag.add(
            format!("download {name}"),
            0,
            Transfer {
                pinned: None,
                device: Some(device),
                len,
                name,
            },
        ) else {
            unreachable!()
        };
        p
    };
    let (u, w) = (down(u, n * n, "U"), down(w, n, "w"));
    let out = Arc::new(Mutex::new((Vec::new(), Vec::new())));
    dag.add(
        "collect",
        0,
        Collect {
            u,
            w,
            out: Arc::clone(&out),
        },
    );
    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = thread_pool::Device::new(&[0], instance.plan());
    instance.run(&mut backend, 1).result.unwrap();
    let (u, w) = std::mem::take(&mut *out.lock().unwrap());
    (lwork, u, w)
}

/// Each fixture diagonalized once on device 0 and held to the bars; its
/// measures are printed, and every miss named before the test fails.
/// Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn diagonalizes_the_overlaps() {
    let mut misses = Vec::new();
    for stem in FIXTURES {
        let (n, a) = fixture(stem);
        let (lwork, u, w) = diagonalize(n, a.clone());
        assert_eq!((u.len(), w.len()), (n * n, n), "{stem}: the outputs");

        let norm = w.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        // max |A U - U diag(w)|, and max |Uᵀ U - I|, on the host.
        let (mut residual, mut orthogonality) = (0.0_f64, 0.0_f64);
        for j in 0..n {
            for i in 0..n {
                let au: f64 = (0..n).map(|k| a[i + k * n] * u[k + j * n]).sum();
                residual = residual.max((au - u[i + j * n] * w[j]).abs());
                let utu: f64 = (0..n).map(|k| u[k + i * n] * u[k + j * n]).sum();
                let delta = if i == j { 1.0 } else { 0.0 };
                orthogonality = orthogonality.max((utu - delta).abs());
            }
        }
        let ascending = w.windows(2).all(|p| p[0] <= p[1]);
        println!(
            "{stem}: n {n}, lwork {lwork}, w in [{:e}, {:e}], \
             max |AU - Uw| = {residual:e} (bar {:e}), max |UᵀU - I| = {orthogonality:e} (bar {BAR:e})",
            w[0],
            w[n - 1],
            BAR * norm
        );
        if residual > BAR * norm {
            misses.push(format!(
                "{stem}: max |AU - Uw| = {residual:e} > {:e}",
                BAR * norm
            ));
        }
        if orthogonality > BAR {
            misses.push(format!(
                "{stem}: max |UᵀU - I| = {orthogonality:e} > {BAR:e}"
            ));
        }
        if !ascending {
            misses.push(format!("{stem}: w is not ascending: {w:?}"));
        }
    }
    assert!(misses.is_empty(), "past the bar:\n{}", misses.join("\n"));
}
