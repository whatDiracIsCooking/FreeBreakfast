//! `trsm` on a card (tier 4, device 0): every side, fill, op and unit flag,
//! in `f32` and `f64`, against a host substitution at orders around the
//! 64-row block's edges and right-hand sides around the kernel's 48-column
//! tile's (one right-hand side is the library's `trsv` path), with padded
//! leading dimensions; `_C` the same as `_T`; `alpha` zero over NaN; the
//! empty calls; and the refusals. The port of `test/recipes/trsm_gpu.cpp`,
//! less its comparison with the library's trsm, which this tree does not
//! bind.
//!
//! Every element a correct trsm must not read (A's other triangle, its
//! diagonal under `WWR_BLAS_DIAG_UNIT`, the padding rows of A and B) holds
//! NaN, so a read of one shows up in the result, and the padding of B must
//! come back bitwise as it went in, as must all of A.
//!
//! The bar, by analysis: substitution's forward error is about
//! `k eps |M⁻¹| |M| |X|`, and A is built so that `|M⁻¹| |M|` is O(1) (its
//! diagonal in [1, 2], the rest of its triangle in [-2/k, 2/k]); a factor 8
//! covers that and the gemm's different summation order. So `8 k eps` on
//! `|X|`, which is O(1) here, with eps the element type's. The oracle is
//! the same substitution in `f64` from the element type's inputs.

use std::sync::Arc;

use fb_core_wwr::blas::BlasHandle;
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::runtime::tests::DeviceBytes;
use fb_recipes_launch::{TrsmScalar, trsm};
use wwr_sys::{
    wwr_blas_diag_type::{self, WWR_BLAS_DIAG_NON_UNIT as NON_UNIT, WWR_BLAS_DIAG_UNIT as UNIT},
    wwr_blas_fill_mode::{
        self, WWR_BLAS_FILL_MODE_LOWER as LOWER, WWR_BLAS_FILL_MODE_UPPER as UPPER,
    },
    wwr_blas_operation::{
        self, WWR_BLAS_OP_C as OP_C, WWR_BLAS_OP_N as OP_N, WWR_BLAS_OP_T as OP_T,
    },
    wwr_blas_side_mode::{self, WWR_BLAS_SIDE_LEFT as LEFT, WWR_BLAS_SIDE_RIGHT as RIGHT},
};

/// An element type under test, with what the oracle and the bit checks
/// need of it.
trait Element: TrsmScalar + std::fmt::Debug {
    /// The type's machine epsilon.
    const EPS: f64;
    fn from_f64(x: f64) -> Self;
    fn to_f64(self) -> f64;
    /// The value's bits, so two NaNs compare.
    fn bits(self) -> u64;
    fn from_bytes(bytes: &[u8]) -> Self;
}

impl Element for f64 {
    const EPS: f64 = f64::EPSILON;
    fn from_f64(x: f64) -> Self {
        x
    }
    fn to_f64(self) -> f64 {
        self
    }
    fn bits(self) -> u64 {
        self.to_bits()
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        f64::from_ne_bytes(bytes.try_into().unwrap())
    }
}

impl Element for f32 {
    const EPS: f64 = f32::EPSILON as f64;
    fn from_f64(x: f64) -> Self {
        x as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
    fn bits(self) -> u64 {
        u64::from(self.to_bits())
    }
    fn from_bytes(bytes: &[u8]) -> Self {
        f32::from_ne_bytes(bytes.try_into().unwrap())
    }
}

/// One solve's modes.
#[derive(Clone, Copy, Debug)]
struct Shape {
    side: wwr_blas_side_mode,
    fill: wwr_blas_fill_mode,
    op: wwr_blas_operation,
    diag: wwr_blas_diag_type,
}

/// Every op and unit flag at one side and fill.
fn shapes(side: wwr_blas_side_mode, fill: wwr_blas_fill_mode) -> Vec<Shape> {
    let mut shapes = Vec::new();
    for op in [OP_N, OP_T] {
        for diag in [NON_UNIT, UNIT] {
            shapes.push(Shape {
                side,
                fill,
                op,
                diag,
            });
        }
    }
    shapes
}

/// Uniform doubles in [-1, 1) from a 64-bit LCG (Knuth's MMIX constants):
/// the tests' own generator, so the problems are the same on every run.
struct Uniform(u64);

impl Uniform {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // The top 53 bits, as a double in [0, 1), then onto [-1, 1).
        (self.0 >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
    }
}

/// One problem as the host holds it: A (order k, leading dimension lda)
/// and B (m × n, leading dimension ldb).
struct Problem<T> {
    shape: Shape,
    m: usize,
    n: usize,
    k: usize,
    lda: usize,
    ldb: usize,
    a: Vec<T>,
    b: Vec<T>,
}

/// A well-conditioned triangular A: its diagonal in [1, 2], the rest of its
/// stored triangle in [-2/k, 2/k], so op(A) is the identity plus a small
/// perturbation and its inverse is O(1). Every element trsm must not read
/// is NaN, and B's padding rows are NaN too.
fn problem<T: Element>(shape: Shape, m: usize, n: usize, seed: u64) -> Problem<T> {
    let k = if shape.side == LEFT { m } else { n };
    let (lda, ldb) = (k + 3, m + 5);
    let mut uniform = Uniform(seed);
    let off_scale = 2.0 / k as f64;
    let mut a = vec![T::from_f64(f64::NAN); lda * k];
    for j in 0..k {
        for i in 0..k {
            if i == j {
                if shape.diag == NON_UNIT {
                    a[i + lda * j] = T::from_f64(1.5 + 0.5 * uniform.next());
                }
            } else if (i > j) == (shape.fill == LOWER) {
                a[i + lda * j] = T::from_f64(off_scale * uniform.next());
            }
        }
    }
    let mut b = vec![T::from_f64(f64::NAN); ldb * n];
    for j in 0..n {
        for i in 0..m {
            b[i + ldb * j] = T::from_f64(uniform.next());
        }
    }
    Problem {
        shape,
        m,
        n,
        k,
        lda,
        ldb,
        a,
        b,
    }
}

/// The oracle: the solve by plain substitution, in `f64` from T's inputs.
/// M is op(A) on the left and op(A)ᵀ on the right, read from the stored
/// triangle alone, and `M Y = alpha C` with C = B or Bᵀ; the m × n packed
/// X.
fn host_trsm<T: Element>(p: &Problem<T>, alpha: f64) -> Vec<f64> {
    let Problem {
        shape,
        m,
        n,
        k,
        lda,
        ldb,
        ..
    } = *p;
    let left = shape.side == LEFT;
    let trans = shape.op != OP_N;
    let lower_fill = shape.fill == LOWER;
    let m_lower = if left {
        lower_fill != trans
    } else {
        lower_fill == trans
    };
    // M, dense and packed: A(i, j) from its stored triangle, zero in the
    // other and 1 on a unit diagonal, so a NaN there never reaches the
    // oracle; M(i, j) is op(A)(i, j) on the left and op(A)(j, i) on the
    // right.
    let mut dense = vec![0.0; k * k];
    for j in 0..k {
        for i in 0..k {
            let stored = if i == j {
                if shape.diag == UNIT {
                    1.0
                } else {
                    p.a[i + lda * j].to_f64()
                }
            } else if (i > j) == lower_fill {
                p.a[i + lda * j].to_f64()
            } else {
                0.0
            };
            // op(A)(r, c) = A(c, r) under T; M = op(A)ᵀ on the right.
            let (r, c) = if trans == left { (j, i) } else { (i, j) };
            dense[r * k + c] = stored;
        }
    }
    let ncols = if left { n } else { m };
    let mut x = vec![0.0; m * n];
    let mut y = vec![0.0; k];
    for c in 0..ncols {
        for (r, y) in y.iter_mut().enumerate() {
            let at = if left { r + ldb * c } else { c + ldb * r };
            *y = alpha * p.b[at].to_f64();
        }
        for step in 0..k {
            let r = if m_lower { step } else { k - 1 - step };
            let row = &dense[r * k..(r + 1) * k];
            let before = if m_lower { 0..r } else { r + 1..k };
            let v = before.fold(y[r], |v, q| v - row[q] * y[q]);
            y[r] = v / row[r];
        }
        for (r, y) in y.iter().enumerate() {
            x[if left { r + m * c } else { c + m * r }] = *y;
        }
    }
    x
}

/// A device and its BLAS handle, after the kernel library's load check.
fn device() -> (Arc<Stream>, BlasHandle) {
    fb_recipes_sys::check_library().unwrap();
    let stream = Arc::new(Stream::new(0));
    let blas = BlasHandle::new(stream.clone());
    (stream, blas)
}

/// A device buffer's elements.
fn download<T: Element>(device: &DeviceBytes, stream: &Stream) -> Vec<T> {
    device
        .read(stream)
        .chunks_exact(size_of::<T>())
        .map(T::from_bytes)
        .collect()
}

/// `trsm` on the problem with `op` in place of its own: B as it came back,
/// A checked unchanged.
fn run<T: Element>(
    stream: &Stream,
    blas: &BlasHandle,
    p: &Problem<T>,
    alpha: T,
    op: wwr_blas_operation,
) -> Vec<T> {
    let a = DeviceBytes::upload(&p.a, stream);
    let mut b = DeviceBytes::upload(&p.b, stream);
    let i = |v: usize| v as i32;
    trsm(
        blas,
        p.shape.side,
        p.shape.fill,
        op,
        p.shape.diag,
        i(p.m),
        i(p.n),
        alpha,
        &a,
        i(p.lda),
        &mut b,
        i(p.ldb),
    )
    .unwrap();
    let after: Vec<T> = download(&a, stream);
    assert!(
        after.iter().zip(&p.a).all(|(x, y)| x.bits() == y.bits()),
        "A was written: {:?}",
        p.shape
    );
    download(&b, stream)
}

/// The worst error of `got` against `want` over B's m × n, scaled by
/// `max(1, |want|)`, Inf if any is NaN, with the padding rows of B checked
/// bitwise against the problem's.
fn worst_error<T: Element>(p: &Problem<T>, got: &[T], want: &[f64]) -> f64 {
    let mut worst = 0.0f64;
    for j in 0..p.n {
        for i in 0..p.ldb {
            let g = got[i + p.ldb * j];
            if i >= p.m {
                assert_eq!(
                    g.bits(),
                    p.b[i + p.ldb * j].bits(),
                    "padding ({i}, {j}): {:?}",
                    p.shape
                );
                continue;
            }
            let w = want[i + p.m * j];
            let e = (g.to_f64() - w).abs() / w.abs().max(1.0);
            worst = if e.is_nan() {
                f64::INFINITY
            } else {
                worst.max(e)
            };
        }
    }
    worst
}

/// The module doc's bar at order k.
fn tolerance<T: Element>(k: usize) -> f64 {
    8.0 * k.max(1) as f64 * T::EPS
}

/// The orders and right-hand sides: one block, its edges, two blocks and
/// their edge, and several; the right-hand sides one (the library's trsv),
/// two, a tile's edge, and past two tiles.
const SIZES: [(usize, usize); 12] = [
    (1, 1),
    (1, 49),
    (2, 2),
    (63, 48),
    (64, 1),
    (64, 49),
    (65, 2),
    (65, 97),
    (128, 48),
    (129, 1),
    (129, 97),
    (200, 49),
];

/// Every op and unit flag at one side and fill, in T, at every size,
/// against the host substitution at the bar.
fn check_against_host<T: Element>(side: wwr_blas_side_mode, fill: wwr_blas_fill_mode) {
    let (stream, blas) = device();
    let alpha = -1.5;
    for shape in shapes(side, fill) {
        for (k, rhs) in SIZES {
            let (m, n) = if side == LEFT { (k, rhs) } else { (rhs, k) };
            let p: Problem<T> = problem(shape, m, n, (k * 1000 + rhs) as u64);
            let got = run(&stream, &blas, &p, T::from_f64(alpha), shape.op);
            let err = worst_error(&p, &got, &host_trsm(&p, alpha));
            assert!(
                err <= tolerance::<T>(k),
                "{shape:?}, k = {k}, rhs = {rhs}: error {err:e} over {:e}",
                tolerance::<T>(k)
            );
        }
    }
}

/// Needs a GPU. On the left over a lower triangle, the density fit's solve
/// among them.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn left_lower_matches_the_host_substitution() {
    check_against_host::<f64>(LEFT, LOWER);
    check_against_host::<f32>(LEFT, LOWER);
}

/// Needs a GPU. On the left over an upper triangle.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn left_upper_matches_the_host_substitution() {
    check_against_host::<f64>(LEFT, UPPER);
    check_against_host::<f32>(LEFT, UPPER);
}

/// Needs a GPU. On the right over a lower triangle, the Cholesky panel's
/// solve among them.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn right_lower_matches_the_host_substitution() {
    check_against_host::<f64>(RIGHT, LOWER);
    check_against_host::<f32>(RIGHT, LOWER);
}

/// Needs a GPU. On the right over an upper triangle.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn right_upper_matches_the_host_substitution() {
    check_against_host::<f64>(RIGHT, UPPER);
    check_against_host::<f32>(RIGHT, UPPER);
}

/// Needs a GPU. On a real T, `_C` is `_T`: the same code path, so the same
/// result to the bit, on both sides.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_conjugate_transpose_is_the_transpose() {
    let (stream, blas) = device();
    for side in [LEFT, RIGHT] {
        let shape = Shape {
            side,
            fill: UPPER,
            op: OP_T,
            diag: NON_UNIT,
        };
        let p: Problem<f64> = problem(shape, 90, 70, 97);
        let t = run(&stream, &blas, &p, 1.0, OP_T);
        let c = run(&stream, &blas, &p, 1.0, OP_C);
        assert!(
            t.iter().zip(&c).all(|(x, y)| x.to_bits() == y.to_bits()),
            "{shape:?}"
        );
    }
}

/// `alpha = 0` on one shape: B comes back zero over its m × n and untouched
/// in its padding, with A and B all NaN, so neither was read. Several
/// blocks, and several right-hand sides, so not the trsv path; and one
/// right-hand side, which must not take it either.
fn check_alpha_zero<T: Element>(stream: &Stream, blas: &BlasHandle, shape: Shape) {
    for (k, rhs) in [(130, 20), (70, 1)] {
        let (m, n) = if shape.side == LEFT {
            (k, rhs)
        } else {
            (rhs, k)
        };
        let mut p: Problem<T> = problem(shape, m, n, 5);
        p.a.fill(T::from_f64(f64::NAN));
        p.b.fill(T::from_f64(f64::NAN));
        let got = run(stream, blas, &p, T::from_f64(0.0), shape.op);
        let err = worst_error(&p, &got, &vec![0.0; m * n]);
        assert!(err == 0.0, "{shape:?}, k = {k}, rhs = {rhs}: {err:e}");
    }
}

/// Needs a GPU. `alpha = 0` over NaN, every shape, in both types.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn alpha_zero_writes_zero_and_reads_nothing() {
    let (stream, blas) = device();
    for side in [LEFT, RIGHT] {
        for fill in [LOWER, UPPER] {
            for shape in shapes(side, fill) {
                check_alpha_zero::<f64>(&stream, &blas, shape);
                check_alpha_zero::<f32>(&stream, &blas, shape);
            }
        }
    }
}

/// `trsm` in doubles over uninitialised device memory of `a_len` and
/// `b_len` elements: what the empty calls and the refusals make.
fn call(
    side: wwr_blas_side_mode,
    (m, n): (i32, i32),
    (a_len, lda): (usize, i32),
    (b_len, ldb): (usize, i32),
) {
    let (_stream, blas) = device();
    let a = DeviceBytes::new(a_len * size_of::<f64>());
    let mut b = DeviceBytes::new(b_len * size_of::<f64>());
    trsm::<f64>(
        &blas, side, LOWER, OP_N, NON_UNIT, m, n, 1.0, &a, lda, &mut b, ldb,
    )
    .unwrap();
}

/// Needs a GPU. With m or n zero nothing is enqueued, whatever the slices
/// hold: one element each here, short of any matrix.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn an_empty_solve_succeeds() {
    call(LEFT, (0, 5), (1, 1), (1, 1));
    call(LEFT, (5, 0), (25, 5), (1, 5));
    call(RIGHT, (5, 0), (1, 1), (1, 5));
}

/// Needs a GPU. A negative m is refused before anything is enqueued.
#[test]
#[ignore = "tier 4: needs a GPU"]
#[should_panic(expected = "m and n are not negative")]
fn a_negative_count_is_refused() {
    call(LEFT, (-1, 5), (1, 1), (1, 1));
}

/// Needs a GPU. On the left A's order is m, so an lda of 4 is short of
/// m = 5.
#[test]
#[ignore = "tier 4: needs a GPU"]
#[should_panic(expected = "A's leading dimension is at least max(1, k)")]
fn a_short_lda_is_refused_on_the_left() {
    call(LEFT, (5, 8), (64, 4), (64, 5));
}

/// Needs a GPU. On the right A's order is n, so an lda of 7 is short of
/// n = 8 though it covers m.
#[test]
#[ignore = "tier 4: needs a GPU"]
#[should_panic(expected = "A's leading dimension is at least max(1, k)")]
fn a_short_lda_is_refused_on_the_right() {
    call(RIGHT, (5, 8), (64, 7), (64, 5));
}

/// Needs a GPU. B's leading dimension is at least m on either side.
#[test]
#[ignore = "tier 4: needs a GPU"]
#[should_panic(expected = "B's leading dimension is at least max(1, m)")]
fn a_short_ldb_is_refused() {
    call(RIGHT, (5, 8), (64, 8), (64, 4));
}

/// Needs a GPU. A 5 × 5 triangle at lda 6 spans 29 elements, so an A of 28
/// is refused.
#[test]
#[ignore = "tier 4: needs a GPU"]
#[should_panic(expected = "a holds the k × k triangle at lda")]
fn a_short_a_is_refused() {
    call(LEFT, (5, 8), (28, 6), (64, 5));
}

/// Needs a GPU. A 5 × 8 matrix at ldb 6 spans 47 elements, so a B of 46 is
/// refused.
#[test]
#[ignore = "tier 4: needs a GPU"]
#[should_panic(expected = "b holds the m × n matrix at ldb")]
fn a_short_b_is_refused() {
    call(LEFT, (5, 8), (29, 6), (46, 6));
}
