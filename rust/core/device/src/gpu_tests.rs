//! [`GpuResources`]'s tests, all on a card (tier 4): the workspace bound
//! once, the solver's work held to its query at the caller's line with
//! nothing enqueued, and the trsm and Cholesky recipes against host
//! references.

#![allow(
    clippy::disallowed_methods,
    reason = "a test reads its result back on the host: the harness waits, not a task"
)]

use std::panic::{AssertUnwindSafe, catch_unwind};

use wwr_sys::{wwr_blas_diag_type, wwr_blas_fill_mode, wwr_blas_operation, wwr_blas_side_mode};

use super::*;
use crate::tests::{DeviceBuf, Pinned, as_child};

// SAFETY: a test's buffer is device memory no host thread dereferences;
// moving it moves the address alone, freed once by its drop.
unsafe impl<T> Send for DeviceBuf<T> {}

const N: wwr_blas_operation = wwr_blas_operation::WWR_BLAS_OP_N;
const T: wwr_blas_operation = wwr_blas_operation::WWR_BLAS_OP_T;
const LOWER: wwr_blas_fill_mode = wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER;
const UPPER: wwr_blas_fill_mode = wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_UPPER;
const LEFT: wwr_blas_side_mode = wwr_blas_side_mode::WWR_BLAS_SIDE_LEFT;
const RIGHT: wwr_blas_side_mode = wwr_blas_side_mode::WWR_BLAS_SIDE_RIGHT;

/// The order of the recipes' acceptance cases.
const ORDER: usize = 512;

/// The recipes' bar: a triangular solve's backward error is a small multiple
/// of `n u |L|` (Higham, *Accuracy and Stability of Numerical Algorithms*,
/// 2nd ed., Thm. 8.5), `512 × 1.1e-16 ≈ 6e-14`, and a diagonally dominant
/// factor's condition number is O(1), so the forward error relative to the
/// solution is under 1e-12 with a margin; Cholesky's backward error has the
/// same form (Thm. 10.3).
const BAR: f64 = 1e-12;

/// `values` on the device, through `stream`, synced.
fn device(stream: &Stream, values: &[f64]) -> DeviceBuf<f64> {
    let mut buf = DeviceBuf::new(values.len());
    let staged = stream.upload(&mut buf, Pinned::from(values)).unwrap();
    drop(staged.sync().unwrap());
    buf
}

/// The first `len` elements of `buf`, read after `stream` drains.
fn host<E: Copy + Default>(stream: &Stream, buf: &DeviceBuf<E>, len: usize) -> Vec<E> {
    let out = Pinned::from(&vec![E::default(); len]);
    let out = stream.download(out, &part(buf, 0, len)).unwrap();
    out.sync().unwrap().read()
}

/// Uniform in [-1, 1): splitmix64, so each case's numbers are its own.
fn uniform(state: &mut u64) -> f64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    (z >> 11) as f64 / (1u64 << 52) as f64 - 1.0
}

/// `‖x - want‖_F / ‖want‖_F`.
fn relative(x: &[f64], want: &[f64]) -> f64 {
    let diff: f64 = x.iter().zip(want).map(|(a, b)| (a - b).powi(2)).sum();
    let norm: f64 = want.iter().map(|b| b * b).sum();
    (diff / norm).sqrt()
}

/// Whether the gates' fast mode is on: `FREE_BREAKFAST_FAST_TESTS` set,
/// non-empty and not starting with `0`, as `FBF_SKIP_IF_FAST_TESTS` reads it.
fn fast_tests() -> bool {
    std::env::var("FREE_BREAKFAST_FAST_TESTS").is_ok_and(|v| !v.is_empty() && !v.starts_with('0'))
}

/// GPU 0's resources, its recipes library checked first.
fn gpu0() -> std::sync::Mutex<GpuResources> {
    fb_recipes_sys::check_library().unwrap();
    GpuResources::new(0, 0)
}

/// The workspace is bound once, and a gemm then runs on it.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_gemm_runs_on_the_bound_workspace() {
    let gpu = gpu0();
    let mut held = gpu.lock().unwrap();
    held.bind_workspace(DeviceBuf::<u8>::new(HANDLE_WORKSPACE_BYTES))
        .unwrap();
    let a = device(held.stream(), &[1.0, 2.0, 3.0, 4.0]);
    let b = device(held.stream(), &[5.0, 6.0, 7.0, 8.0]);
    let mut c = device(held.stream(), &[0.0; 4]);
    held.gemm(N, N, 2, 2, 2, 1.0, &a, 2, &b, 2, 0.0, &mut c, 2)
        .unwrap();
    // [[1, 3], [2, 4]] [[5, 7], [6, 8]], column-major.
    assert_eq!(host(held.stream(), &c, 4), [23.0, 34.0, 31.0, 46.0]);
}

/// A second workspace is refused: the library would keep the first's
/// pointer for calls in flight.
#[test]
#[ignore = "tier 4: needs a GPU"]
#[should_panic(expected = "Pre-condition of bind_workspace violated: the workspace is bound once")]
fn a_second_workspace_panics() {
    let gpu = gpu0();
    let mut held = gpu.lock().unwrap();
    held.bind_workspace(DeviceBuf::<u8>::new(HANDLE_WORKSPACE_BYTES))
        .unwrap();
    let _ = held.bind_workspace(DeviceBuf::<u8>::new(HANDLE_WORKSPACE_BYTES));
}

/// A gemm whose C is one element short panics naming it.
#[test]
#[ignore = "tier 4: needs a GPU"]
#[should_panic(expected = "gemm: C is 2 × 2 at leading dimension 2, 4 elements")]
fn a_short_gemm_operand_panics_before_anything_is_enqueued() {
    let gpu = gpu0();
    let mut held = gpu.lock().unwrap();
    let a = DeviceBuf::<f64>::new(4);
    let mut c = DeviceBuf::<f64>::new(3);
    let _ = held.gemm(N, N, 2, 2, 2, 1.0, &a, 2, &a, 2, 0.0, &mut c, 2);
}

/// syevd with its queried work: the eigenvalues of `[[2, -1, 0], [-1, 2,
/// -1], [0, -1, 2]]`, `2 - √2`, 2 and `2 + √2`, and a zero flag.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn syevd_with_its_queried_work_diagonalizes() {
    let gpu = gpu0();
    let mut held = gpu.lock().unwrap();
    let lwork = held.syevd_lwork(3).unwrap();
    let mut a = device(
        held.stream(),
        &[2.0, -1.0, 0.0, -1.0, 2.0, -1.0, 0.0, -1.0, 2.0],
    );
    let mut w = DeviceBuf::<f64>::new(3);
    let mut work = DeviceBuf::<f64>::new(lwork);
    let mut info = DeviceBuf::<i32>::new(1);
    held.syevd(3, &mut a, &mut w, &mut work, &mut info).unwrap();
    assert_eq!(host(held.stream(), &info, 1), [0]);
    let s = 2f64.sqrt();
    for (got, want) in host(held.stream(), &w, 3)
        .iter()
        .zip([2.0 - s, 2.0, 2.0 + s])
    {
        assert!((got - want).abs() < 1e-14, "{got} against {want}");
    }
}

/// The order of the short-work cases.
const SHORT: usize = 64;

/// A work buffer one element under the query panics, naming the call and
/// the query. [`a_short_syevd_work_buffer_panics_at_its_caller`] runs it as
/// a child for the line.
#[test]
#[ignore = "tier 4: needs a GPU"]
#[should_panic(expected = "syevd at n = 64: the work buffer holds")]
fn a_short_syevd_work_buffer_panics() {
    as_child();
    let gpu = gpu0();
    let mut held = gpu.lock().unwrap();
    let need = held.syevd_lwork(SHORT).unwrap();
    let (mut a, mut w) = (DeviceBuf::new(SHORT * SHORT), DeviceBuf::new(SHORT));
    let (mut work, mut info) = (DeviceBuf::new(need - 1), DeviceBuf::new(1));
    let _ = held.syevd(SHORT, &mut a, &mut w, &mut work, &mut info); // MARK syevd
}

/// The panic is located at the caller's line, `#[track_caller]` reaching
/// through the check: the hook's line, read from the child's stderr.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_short_syevd_work_buffer_panics_at_its_caller() {
    let name = "gpu_tests::a_short_syevd_work_buffer_panics";
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            name,
            "--nocapture",
            "--test-threads=1",
            "--ignored",
        ])
        .env("FB_CONTRACT_CHILD", "1")
        .output()
        .unwrap();
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{:?}\n{stderr}", out.status);
    let line = include_str!("gpu_tests.rs")
        .lines()
        .position(|l| l.contains("// MARK syevd") && !l.contains("contains"))
        .unwrap()
        + 1;
    let want = format!("thread '{name}' panicked at {}:{line}:", file!());
    assert!(
        stderr
            .lines()
            .any(|l| l.starts_with(&want) && l.contains("syevd at n = 64")),
        "{stderr}\n  wanted a line starting {want}"
    );
}

/// With nothing enqueued: after the panic, A, w and the flag are bitwise
/// what they were.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn a_short_syevd_work_buffer_enqueues_nothing() {
    let gpu = gpu0();
    let mut held = gpu.lock().unwrap();
    let need = held.syevd_lwork(SHORT).unwrap();
    let mut state = 7;
    let before: Vec<f64> = (0..SHORT * SHORT).map(|_| uniform(&mut state)).collect();
    let mut a = device(held.stream(), &before);
    let mut w = device(held.stream(), &[-3.5; SHORT]);
    let mut info = DeviceBuf::<i32>::new(1);
    held.stream().fill(&mut info, 0x5A).unwrap();
    let mut work = DeviceBuf::<f64>::new(need - 1);
    let refused = catch_unwind(AssertUnwindSafe(|| {
        held.syevd(SHORT, &mut a, &mut w, &mut work, &mut info)
    }));
    assert!(refused.is_err());
    let bits = |v: Vec<f64>| v.into_iter().map(f64::to_bits).collect::<Vec<_>>();
    let stream = held.stream();
    assert_eq!(
        bits(host(stream, &a, SHORT * SHORT)),
        bits(before),
        "A was written"
    );
    assert_eq!(
        bits(host(stream, &w, SHORT)),
        bits(vec![-3.5; SHORT]),
        "w was written"
    );
    assert_eq!(host(stream, &info, 1), [0x5A5A_5A5A], "info was written");
}

/// A diagonally dominant lower-triangular `n × n` L, column-major: the
/// diagonal in [1.5, 2.5), each off-diagonal entry under `1/n`, so each
/// row's off-diagonal sum is under 1.
fn dominant_lower(n: usize, state: &mut u64) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        l[j + j * n] = 2.0 + 0.5 * uniform(state);
        for i in j + 1..n {
            l[i + j * n] = uniform(state) / n as f64;
        }
    }
    l
}

/// Solve `M y = rhs` in place for a triangular `k × k` column-major M.
fn substitute(m: &[f64], k: usize, lower: bool, rhs: &mut [f64]) {
    let rows: Vec<usize> = if lower {
        (0..k).collect()
    } else {
        (0..k).rev().collect()
    };
    for &i in &rows {
        let mut s = rhs[i];
        for j in 0..k {
            let below = if lower { j < i } else { j > i };
            if below {
                s -= m[i + j * k] * rhs[j];
            }
        }
        rhs[i] = s / m[i + i * k];
    }
}

/// trsm against substitution on the host, at order 512 with 77 right-hand
/// sides and `alpha` 1.5, for each side, each triangle of storage and each
/// operation among the four cases below; the triangle A does not store is
/// NaN, so a read of it would show.
///
/// A slow case (`docs/testing.md`, "Slow cases"): about 4.5 s on the
/// gfx1200, most of it the library loading a gemm for each operation pair
/// on first use; [`cholesky_matches_a_host_factor`], which solves on the
/// right through this same code, is the family's representative.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn trsm_matches_a_host_solve() {
    if fast_tests() {
        return;
    }
    let (k, rhs, alpha) = (ORDER, 77, 1.5);
    let gpu = gpu0();
    let mut held = gpu.lock().unwrap();
    let mut state = 1808;
    let l = dominant_lower(k, &mut state);
    let lt = |i: usize, j: usize| l[j + i * k];
    for (side, fill, op) in [
        (LEFT, LOWER, N),
        (LEFT, UPPER, T),
        (RIGHT, LOWER, T),
        (RIGHT, UPPER, N),
    ] {
        let upper = fill == UPPER;
        // A holds L (lower) or Lᵀ (upper), NaN elsewhere.
        let mut stored = vec![f64::NAN; k * k];
        for j in 0..k {
            for i in 0..k {
                if (i >= j) != upper || i == j {
                    stored[i + j * k] = if upper { lt(i, j) } else { l[i + j * k] };
                }
            }
        }
        // op(A) dense, then M: op(A) on the left, op(A)ᵀ on the right.
        let a_at = |i: usize, j: usize| if upper { lt(i, j) } else { l[i + j * k] };
        let op_a = |i: usize, j: usize| if op == N { a_at(i, j) } else { a_at(j, i) };
        let left = side == LEFT;
        let mut m = vec![0.0; k * k];
        for j in 0..k {
            for i in 0..k {
                m[i + j * k] = if left { op_a(i, j) } else { op_a(j, i) };
            }
        }
        // op(A) is lower for (lower, N) and (upper, T); M is its transpose
        // on the right.
        let op_lower = (fill == LOWER) == (op == N);
        let lower_m = op_lower == left;
        let (rows, cols) = if left { (k, rhs) } else { (rhs, k) };
        let b: Vec<f64> = (0..rows * cols).map(|_| uniform(&mut state)).collect();
        let mut want = vec![0.0; rows * cols];
        if left {
            for c in 0..cols {
                let mut y: Vec<f64> = (0..k).map(|r| alpha * b[r + c * rows]).collect();
                substitute(&m, k, lower_m, &mut y);
                want[c * rows..(c + 1) * rows].copy_from_slice(&y);
            }
        } else {
            for r in 0..rows {
                let mut y: Vec<f64> = (0..k).map(|c| alpha * b[r + c * rows]).collect();
                substitute(&m, k, lower_m, &mut y);
                for (c, v) in y.into_iter().enumerate() {
                    want[r + c * rows] = v;
                }
            }
        }
        let a = device(held.stream(), &stored);
        let mut x = device(held.stream(), &b);
        let diag = wwr_blas_diag_type::WWR_BLAS_DIAG_NON_UNIT;
        held.trsm(
            side,
            fill,
            op,
            diag,
            rows as i32,
            cols as i32,
            alpha,
            &a,
            k as i32,
            &mut x,
            rows as i32,
        )
        .unwrap();
        let got = host(held.stream(), &x, rows * cols);
        let error = relative(&got, &want);
        assert!(
            error <= BAR,
            "{side:?} {fill:?} {op:?}: relative error {error:e} past {BAR:e}"
        );
    }
}

/// Cholesky against the host's, at order 512: a diagonally dominant
/// symmetric A, its strict upper triangle NaN, which the factor must
/// neither read nor write; L to the bar, and a zero flag.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn cholesky_matches_a_host_factor() {
    let n = ORDER;
    let gpu = gpu0();
    let mut held = gpu.lock().unwrap();
    let mut state = 410;
    let mut a = vec![f64::NAN; n * n];
    for j in 0..n {
        a[j + j * n] = 2.0 + 0.5 * uniform(&mut state);
        for i in j + 1..n {
            a[i + j * n] = uniform(&mut state) / n as f64;
        }
    }
    // The host's factor, column by column, over the lower triangle.
    let mut want = a.clone();
    for j in 0..n {
        let d = (want[j + j * n] - (0..j).map(|p| want[j + p * n].powi(2)).sum::<f64>()).sqrt();
        want[j + j * n] = d;
        for i in j + 1..n {
            let s: f64 = (0..j).map(|p| want[i + p * n] * want[j + p * n]).sum();
            want[i + j * n] = (want[i + j * n] - s) / d;
        }
    }
    let mut factor = device(held.stream(), &a);
    let mut tile = DeviceBuf::<f64>::new(CHOLESKY_TILE * CHOLESKY_TILE);
    let mut flag = DeviceBuf::<i32>::new(1);
    held.cholesky(n, &mut factor, &mut tile, &mut flag).unwrap();
    assert_eq!(host(held.stream(), &flag, 1), [0]);
    let got = host(held.stream(), &factor, n * n);
    let lower = |m: &[f64]| -> Vec<f64> {
        (0..n)
            .flat_map(|j| (j..n).map(move |i| i + j * n))
            .map(|ix| m[ix])
            .collect()
    };
    let error = relative(&lower(&got), &lower(&want));
    assert!(error <= BAR, "relative error {error:e} past {BAR:e}");
    let upper_nan = (0..n).all(|j| (0..j).all(|i| got[i + j * n].is_nan()));
    assert!(upper_nan, "the strict upper triangle was written");
}

/// `inverse_sqrt` against the host's `1.0 / x.sqrt()`, bit for bit: the
/// kernel's is a correctly rounded root and division too, contraction off
/// on HIP (`fb_recipes_launch`'s `maps_each_element_as_the_host_does` has
/// the argument), and the floor's elements, at and below it, 0.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn inverse_sqrt_matches_the_host() {
    const FLOOR: f64 = 1e-10;
    let gpu = gpu0();
    let mut held = gpu.lock().unwrap();
    let mut state = 7;
    let mut values: Vec<f64> = (0..1000)
        .map(|_| 10f64.powf(6.0 * uniform(&mut state)))
        .collect();
    values.extend([FLOOR, 0.5 * FLOOR, 0.0, -1.0]);
    let mut data = device(held.stream(), &values);
    held.inverse_sqrt(&mut data, FLOOR).unwrap();
    let got = host(held.stream(), &data, values.len());
    for (&x, &y) in values.iter().zip(&got) {
        let want = if x <= FLOOR { 0.0 } else { 1.0 / x.sqrt() };
        assert_eq!(
            y.to_bits(),
            want.to_bits(),
            "1/sqrt({x}): {y} against {want}"
        );
    }
}

/// `diis_solve` on two residuals of squared norms 1 and 4, orthogonal: the
/// coefficients minimising `|c_0 e_0 + c_1 e_1|` on `c_0 + c_1 = 1` are
/// proportional to the inverse norms squared, 0.8 and 0.2, within a few
/// ulps of an order-3 elimination (1e-14); a zero Gram is singular and gives
/// the unit selector at `newest`.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn diis_solve_weighs_residuals_by_their_inverse_norms() {
    let gpu = gpu0();
    let mut held = gpu.lock().unwrap();
    // The Gram at leading dimension 3, of which two columns are read.
    let gram = device(held.stream(), &[1.0, 0.0, 0.0, 0.0, 4.0, 0.0]);
    let mut coeff = device(held.stream(), &[f64::NAN; 2]);
    let mut flag = DeviceBuf::<i32>::new(1);
    held.diis_solve(2, 3, 1, &gram, &mut coeff, &mut flag)
        .unwrap();
    let (c, singular) = (
        host(held.stream(), &coeff, 2),
        host(held.stream(), &flag, 1),
    );
    assert_eq!(singular, [0]);
    assert!(
        (c[0] - 0.8).abs() < 1e-14 && (c[1] - 0.2).abs() < 1e-14,
        "{c:?}"
    );

    let zero = device(held.stream(), &[0.0; 6]);
    held.diis_solve(2, 3, 1, &zero, &mut coeff, &mut flag)
        .unwrap();
    let (c, singular) = (
        host(held.stream(), &coeff, 2),
        host(held.stream(), &flag, 1),
    );
    assert_eq!((c, singular), (vec![0.0, 1.0], vec![1]));
}
