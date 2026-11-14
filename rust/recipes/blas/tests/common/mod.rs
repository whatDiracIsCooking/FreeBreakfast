//! What the sandwich tests share: seeded operands, the host reference and
//! the bars, those of the C++ `test/recipes/{gemmwich,gmmwich}_gpu.cpp`.
//!
//! Every scalar is held on the host as a complex `f64` (zero imaginary part
//! on a real type), rounded through the device type first, so the reference
//! is computed from exactly the values uploaded, in `f64` arithmetic: one
//! reference path for all four types, and `ᴴ` the conjugate transpose,
//! which on a real type is the transpose.

use fb_core_wwr::blas::BlasScalar;
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::runtime::tests::DeviceBytes;
use fb_recipes_blas::BlasOp;
use wwr_sys::{wwr_blas_operation, wwr_complex64, wwr_complex128};

/// The C++ cases' shapes: `OpX(X)` is `N` × `M` and C `M` × `M`. N is odd
/// and not a tile multiple, and M differs from it, so a swapped dimension
/// or leading dimension fails rather than reading in bounds by luck.
pub const N: usize = 37;
/// See [`N`].
pub const M: usize = 23;

/// A complex `f64`, `(re, im)`.
pub type Z = (f64, f64);

/// A BLAS scalar as the tests see it.
pub trait Scalar: BlasScalar {
    /// Whether the type is complex, so operands get imaginary parts.
    const COMPLEX: bool;
    /// The C++ bar, per unit of n: two chained products of n-term sums, so
    /// the error grows with n; a complex product is four real ones, so
    /// twice the real bar.
    const TOL_PER_N: f64;
    /// `z` rounded to this type.
    fn from_z(z: Z) -> Self;
    /// The value as a complex `f64`, exactly.
    fn to_z(self) -> Z;
    /// One value from its native-endian bytes.
    fn from_bytes(b: &[u8]) -> Self;
}

impl Scalar for f32 {
    const COMPLEX: bool = false;
    const TOL_PER_N: f64 = 1e-5;
    fn from_z(z: Z) -> Self {
        z.0 as f32
    }
    fn to_z(self) -> Z {
        (self.into(), 0.0)
    }
    fn from_bytes(b: &[u8]) -> Self {
        f32::from_ne_bytes(b.try_into().unwrap())
    }
}

impl Scalar for f64 {
    const COMPLEX: bool = false;
    const TOL_PER_N: f64 = 1e-13;
    fn from_z(z: Z) -> Self {
        z.0
    }
    fn to_z(self) -> Z {
        (self, 0.0)
    }
    fn from_bytes(b: &[u8]) -> Self {
        f64::from_ne_bytes(b.try_into().unwrap())
    }
}

impl Scalar for wwr_complex64 {
    const COMPLEX: bool = true;
    const TOL_PER_N: f64 = 2e-5;
    fn from_z(z: Z) -> Self {
        wwr_complex64 {
            re: z.0 as f32,
            im: z.1 as f32,
        }
    }
    fn to_z(self) -> Z {
        (self.re.into(), self.im.into())
    }
    fn from_bytes(b: &[u8]) -> Self {
        Self {
            re: f32::from_bytes(&b[..4]),
            im: f32::from_bytes(&b[4..]),
        }
    }
}

impl Scalar for wwr_complex128 {
    const COMPLEX: bool = true;
    const TOL_PER_N: f64 = 2e-13;
    fn from_z(z: Z) -> Self {
        wwr_complex128 { re: z.0, im: z.1 }
    }
    fn to_z(self) -> Z {
        (self.re, self.im)
    }
    fn from_bytes(b: &[u8]) -> Self {
        Self {
            re: f64::from_bytes(&b[..8]),
            im: f64::from_bytes(&b[8..]),
        }
    }
}

/// Whether `Op` is `N`.
pub fn is_n<Op: BlasOp>() -> bool {
    Op::RAW == wwr_blas_operation::WWR_BLAS_OP_N
}

/// A `rows` × `cols` column-major matrix on the host, (i, j) at
/// `i + j rows`.
#[derive(Clone, Debug)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub v: Vec<Z>,
}

/// splitmix64, so a seed fixes every value on every machine.
fn splitmix(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9e37_79b9_7f4a_7c15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Uniform on [-1, 1).
fn uniform(state: &mut u64) -> f64 {
    (splitmix(state) >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
}

/// A seeded `rows` × `cols` matrix of `S`, entries uniform on [-1, 1) in
/// each part (imaginary parts only on a complex `S`), rounded to `S`.
pub fn random<S: Scalar>(rows: usize, cols: usize, seed: u64) -> Mat {
    let mut state = seed;
    let v = (0..rows * cols)
        .map(|_| {
            let re = uniform(&mut state);
            let im = if S::COMPLEX { uniform(&mut state) } else { 0.0 };
            S::from_z((re, im)).to_z()
        })
        .collect();
    Mat { rows, cols, v }
}

/// `Aᴴ`: the conjugate transpose, the transpose on a real matrix.
pub fn adjoint(a: &Mat) -> Mat {
    let mut v = vec![(0.0, 0.0); a.v.len()];
    for j in 0..a.cols {
        for i in 0..a.rows {
            let (re, im) = a.v[i + j * a.rows];
            v[j + i * a.cols] = (re, -im);
        }
    }
    Mat {
        rows: a.cols,
        cols: a.rows,
        v,
    }
}

/// `A B`.
pub fn product(a: &Mat, b: &Mat) -> Mat {
    assert_eq!(a.cols, b.rows);
    let mut v = vec![(0.0, 0.0); a.rows * b.cols];
    for j in 0..b.cols {
        for k in 0..a.cols {
            let (br, bi) = b.v[k + j * b.rows];
            for i in 0..a.rows {
                let (ar, ai) = a.v[i + k * a.rows];
                let c = &mut v[i + j * a.rows];
                c.0 += ar * br - ai * bi;
                c.1 += ar * bi + ai * br;
            }
        }
    }
    Mat {
        rows: a.rows,
        cols: b.cols,
        v,
    }
}

/// `a`'s values on the device, as `S`, copied on `owner`.
pub fn upload<S: Scalar>(a: &Mat, owner: &Stream) -> DeviceBytes {
    let host: Vec<S> = a.v.iter().map(|&z| S::from_z(z)).collect();
    DeviceBytes::upload(&host, owner)
}

/// Room on the device for `len` elements of `S`.
pub fn room<S: Scalar>(len: usize) -> DeviceBytes {
    DeviceBytes::new(len * size_of::<S>())
}

/// The device's values, as complex `f64`s.
pub fn download<S: Scalar>(device: &DeviceBytes, owner: &Stream) -> Vec<Z> {
    device
        .read(owner)
        .chunks_exact(size_of::<S>())
        .map(|b| S::from_bytes(b).to_z())
        .collect()
}

/// Every entry of `got` within `S`'s bar of `want`, the modulus of the
/// difference; `what` names the case in the message.
#[track_caller]
pub fn assert_matches<S: Scalar>(got: &[Z], want: &Mat, what: &str) {
    assert_eq!(got.len(), want.v.len(), "{what}: length");
    let tol = S::TOL_PER_N * N as f64;
    for (e, (&g, &w)) in got.iter().zip(&want.v).enumerate() {
        let err = (g.0 - w.0).hypot(g.1 - w.1);
        assert!(
            err <= tol,
            "{what}: at ({}, {}), got {g:?}, want {w:?}, off by {err:e} > {tol:e}",
            e % want.rows,
            e / want.rows,
        );
    }
}
