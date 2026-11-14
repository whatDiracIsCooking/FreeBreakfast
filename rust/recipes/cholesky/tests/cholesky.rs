//! The factorization on a card (tier 4, device 0), one run per matrix, on
//! the committed fixtures `tests/data/*.npy` (written by
//! `tests/data/matrices.py`, which says how each is built).
//!
//! Each matrix is staged, uploaded, factored by [`add_cholesky`] and L
//! brought back by a task that reads the check's `checked` row too, so it
//! runs only after a check that passed.
//!
//! **The bar, by analysis, one run.** Cholesky is backward stable:
//! `L Lᵀ = A + ΔA` with `|ΔA| <= γ₍ₙ₊₁₎ |L| |Lᵀ|` elementwise (Higham,
//! *Accuracy and Stability of Numerical Algorithms*, 2nd ed., Theorem 10.3),
//! and `(|L| |Lᵀ|)ᵢⱼ <= ‖Lᵢ‖ ‖Lⱼ‖ = √(aᵢᵢ aⱼⱼ) <= max |A|` by
//! Cauchy-Schwarz, row i of L having norm `√aᵢᵢ`. So
//! `max |L Lᵀ - A| <= (n + 1) ε max |A|` to first order, whatever the
//! blocking or the order the BLAS sums in, and the bar
//! `100 n ε max |A|` holds two decades of margin, the host product's own
//! rounding (`n ε` again) inside it. Only the lower triangles are compared:
//! the upper triangle of the output row is A's, which must come back bit
//! for bit.
//!
//! **The failures.** A matrix whose first non-positive leading minor is k
//! returns `Numerical` naming k, from `cholesky check`, and the task that
//! reads `checked` does not run: k = 3, the acceptance's, in the first
//! tile, and k = 67, in the second, so the flag's `first + k + 1` and the
//! later tiles' skip are both exercised.

use std::path::Path;
use std::sync::{Arc, Mutex};

use fb_core_arena::{Bufs, Device, Layout, Pinned, RowId, memcpy_d2p, memcpy_p2d};
use fb_core_error_handling::{ErrorKind, Status};
use fb_core_planner::{Dag, Task};
use fb_core_thread_pool::{DeviceStream, Lane};
use fb_recipes_cholesky::add_cholesky;

/// The SPD fixtures' orders: around the 64-column tile.
const ORDERS: [usize; 6] = [1, 63, 64, 65, 128, 200];

/// The bar's multiple of `n ε max |A|`.
const MARGIN: f64 = 100.0;

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

/// GPU: the staged A copied up into a device output.
struct Upload {
    staged: RowId<Pinned, f64>,
    len: usize,
}

impl Task for Upload {
    type Rows = RowId<Device, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        host.read(self.staged);
        device.output("A", self.len)
    }

    fn run(
        &self,
        a: &Self::Rows,
        device: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        lane: Lane<'_, DeviceStream>,
    ) -> Status {
        let Lane::Stream { stream, .. } = lane else {
            unreachable!("a GPU task runs on a stream")
        };
        memcpy_p2d(&device.take_mut(*a), host, self.staged, stream)
    }
}

/// GPU: L copied down into a pinned output, after the check: it reads
/// `checked`, which the check writes only for a good factor.
struct Download {
    l: RowId<Device, f64>,
    checked: RowId<Pinned, i32>,
    len: usize,
}

impl Task for Download {
    type Rows = RowId<Pinned, f64>;
    const GPU: bool = true;

    fn layout(&self, device: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) -> Self::Rows {
        device.read(self.l);
        host.read(self.checked);
        host.output("L down", self.len)
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
        memcpy_d2p(host, *down, device.get(self.l), stream)
    }
}

/// CPU: the pinned copy of L read out.
struct Collect {
    l: RowId<Pinned, f64>,
    out: Arc<Mutex<Option<Vec<f64>>>>,
}

impl Task for Collect {
    type Rows = ();
    const GPU: bool = false;

    fn layout(&self, _: &mut Layout<'_, Device>, host: &mut Layout<'_, Pinned>) {
        host.read(self.l);
    }

    fn run(
        &self,
        _: &(),
        _: &Bufs<'_, Device>,
        host: &Bufs<'_, Pinned>,
        _: Lane<'_, DeviceStream>,
    ) -> Status {
        *self.out.lock().unwrap() = Some(host.get(self.l).to_vec());
        Ok(())
    }
}

/// `(n, A)` of the fixture `stem`, column-major (A is symmetric, so the
/// file's row-major order reads the same).
fn fixture(stem: &str) -> (usize, Vec<f64>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("tests/data/{stem}.npy"));
    let a = fb_io_npy::read_npy(&path).unwrap();
    let [n, m] = a.shape[..] else {
        panic!("{stem}: not 2-D")
    };
    assert_eq!(n, m, "{stem}: not square");
    (n, a.data)
}

/// A planned on device 0 and run once: the run's status, and L if the
/// download after the check ran.
fn factor(n: usize, a: Vec<f64>) -> (Status, Option<Vec<f64>>) {
    let mut dag = Dag::new();
    let staged = dag.add("stage", 0, Stage(a));
    let a = dag.add("upload", 0, Upload { staged, len: n * n });
    let (l, checked) = add_cholesky(&mut dag, 0, "L", n, a).unwrap();
    let down = dag.add(
        "download",
        0,
        Download {
            l,
            checked,
            len: n * n,
        },
    );
    let out = Arc::new(Mutex::new(None));
    dag.add(
        "collect",
        0,
        Collect {
            l: down,
            out: Arc::clone(&out),
        },
    );
    let mut instance = dag.plan().instantiate(&[0]);
    let mut backend = instance.backend();
    let result = instance.run(&mut backend, 1).result;
    let l = out.lock().unwrap().take();
    (result, l)
}

/// Each SPD fixture factored once on device 0 and held to the bar, its
/// upper triangle unchanged; its measures are printed, and every miss
/// named before the test fails. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn factors_within_the_backward_error() {
    let mut misses = Vec::new();
    for n in ORDERS {
        let stem = format!("spd_{n}");
        let (_, a) = fixture(&stem);
        let (result, l) = factor(n, a.clone());
        result.unwrap_or_else(|e| panic!("{stem}: {e}"));
        let l = l.expect("the download ran after a passing check");
        let max_a = a.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let bar = MARGIN * n as f64 * f64::EPSILON * max_a;
        // max |L Lᵀ - A| over the lower triangle, L's lower triangle alone.
        let mut residual = 0.0_f64;
        let mut upper_changed = 0;
        for j in 0..n {
            for i in 0..n {
                if i < j {
                    upper_changed += usize::from(l[i + j * n].to_bits() != a[i + j * n].to_bits());
                    continue;
                }
                let llt: f64 = (0..=j).map(|k| l[i + k * n] * l[j + k * n]).sum();
                // Not f64::max, which would drop a NaN.
                let miss = (llt - a[i + j * n]).abs();
                if miss.is_nan() || miss > residual {
                    residual = miss;
                }
            }
        }
        println!("{stem}: max |A| {max_a:e}, max |LLᵀ - A| = {residual:e} (bar {bar:e})");
        if residual.is_nan() || residual > bar {
            misses.push(format!("{stem}: max |LLᵀ - A| = {residual:e} > {bar:e}"));
        }
        if upper_changed != 0 {
            misses.push(format!(
                "{stem}: {upper_changed} elements of the strict upper triangle changed"
            ));
        }
    }
    assert!(misses.is_empty(), "past the bar:\n{}", misses.join("\n"));
}

/// Each matrix whose first non-positive leading minor is k fails the run
/// with `Numerical` naming k, from the check, and the download that reads
/// `checked` never runs. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn names_the_first_minor_that_is_not_positive() {
    for k in [3, 67] {
        let (n, a) = fixture(&format!("minor_{k}"));
        let (result, l) = factor(n, a);
        let e = result.expect_err("a matrix that is not positive definite");
        assert_eq!(e.kind, ErrorKind::Numerical, "minor {k}: {e}");
        assert_eq!(
            e.message,
            format!("cholesky: the leading minor {k} is not positive")
        );
        assert_eq!(e.context, ["cholesky check@cpu0"]);
        assert!(l.is_none(), "minor {k}: a reader of checked ran");
    }
}
