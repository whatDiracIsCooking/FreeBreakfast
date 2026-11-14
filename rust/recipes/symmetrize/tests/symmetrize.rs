//! Both calls on a card (tier 4, device 0), one run each, on the committed
//! nonsymmetric 37 × 37 (`tests/data/a37.npy`, written by
//! `tests/data/matrix.py`): one full 32-wide tile and a partial one, so the
//! diagonal tiles, the one off-diagonal pair and the edge guards all run.
//!
//! The bar is bitwise, by analysis. The average's `(a + b) · ½` is a
//! correctly rounded sum and an exact halving (no element is near the
//! subnormal range), and IEEE addition commutes, so `(i, j)` and `(j, i)`
//! carry the same bits as each other and as the host's `(a + b) / 2`. The
//! mirror is a copy. The block sits at leading dimension 40 with its padding
//! rows holding a NaN payload, which must come back unchanged.

use std::path::Path;

use fb_core_wwr::runtime::Stream;
use fb_core_wwr::runtime::tests::DeviceBytes;
use fb_recipes_symmetrize::{symmetrize, symmetrize_from};
use wwr_sys::wwr_blas_fill_mode;

/// The matrix's order, and the leading dimension it is laid out at.
const N: usize = 37;
const LDA: usize = 40;

/// What the padding rows hold: a quiet NaN with a payload no arithmetic
/// writes.
const PAD: u64 = 0x7ff8_0000_0000_1508;

/// The committed matrix at leading dimension [`LDA`], padding [`PAD`].
fn padded() -> Vec<f64> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/a37.npy");
    let a = fb_io_npy::read_npy(&path).unwrap();
    assert_eq!(a.shape, [N, N]);
    let mut out = vec![f64::from_bits(PAD); LDA * N];
    for j in 0..N {
        out[j * LDA..j * LDA + N].copy_from_slice(&a.data[j * N..(j + 1) * N]);
    }
    out
}

/// `host` uploaded to device 0, `op` run on it, and the result's bits.
fn on_device(host: &[f64], op: impl FnOnce(&Stream, &mut DeviceBytes)) -> Vec<u64> {
    fb_recipes_sys::check_library().unwrap();
    let stream = Stream::new(0);
    let mut device = DeviceBytes::upload(host, &stream);
    op(&stream, &mut device);
    device
        .read(&stream)
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| u64::from_ne_bytes(*c))
        .collect()
}

/// Every element of `got` against `want(i, j)` for the block and [`PAD`]
/// for the padding, bit for bit.
fn assert_bits(got: &[u64], want: impl Fn(usize, usize) -> f64) {
    let mut misses = Vec::new();
    for j in 0..N {
        for i in 0..LDA {
            let expected = if i < N { want(i, j).to_bits() } else { PAD };
            let bits = got[i + j * LDA];
            if bits != expected {
                misses.push(format!(
                    "({i}, {j}): {:e}, expected {:e}",
                    f64::from_bits(bits),
                    f64::from_bits(expected)
                ));
            }
        }
    }
    assert!(
        misses.is_empty(),
        "{} differ:\n{}",
        misses.len(),
        misses.join("\n")
    );
}

/// Needs a GPU. After the average, `(i, j)` and `(j, i)` are both the
/// host's `(a + b) / 2`, so the same bits; a second average of the now
/// symmetric matrix leaves every bit as it was.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_average_is_the_hosts_bit_for_bit() {
    let a = padded();
    let at = |i: usize, j: usize| a[i + j * LDA];
    let got = on_device(&a, |stream, device| {
        symmetrize(stream, N, device, LDA).unwrap();
    });
    assert_bits(&got, |i, j| (at(i, j) + at(j, i)) / 2.0);

    let symmetric: Vec<f64> = got.iter().map(|&b| f64::from_bits(b)).collect();
    let again = on_device(&symmetric, |stream, device| {
        symmetrize(stream, N, device, LDA).unwrap();
    });
    assert_eq!(again, got, "a symmetric input changed");
}

/// Needs a GPU. Filled from the lower triangle, the strict upper is the
/// lower's bits and the lower, diagonal included, is untouched; filled
/// from the upper, the reverse.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn the_mirror_copies_one_triangle_onto_the_other() {
    let a = padded();
    let at = |i: usize, j: usize| a[i + j * LDA];
    let lower = on_device(&a, |stream, device| {
        symmetrize_from(
            stream,
            wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER,
            N,
            device,
            LDA,
        )
        .unwrap();
    });
    assert_bits(&lower, |i, j| if i < j { at(j, i) } else { at(i, j) });

    let upper = on_device(&a, |stream, device| {
        symmetrize_from(
            stream,
            wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_UPPER,
            N,
            device,
            LDA,
        )
        .unwrap();
    });
    assert_bits(&upper, |i, j| if i > j { at(j, i) } else { at(i, j) });
}

/// Needs a GPU. A slice shorter than the block panics before anything is
/// enqueued.
#[test]
#[ignore = "tier 4: needs a GPU"]
#[should_panic(expected = "symmetrize: the 3 × 3 block at lda 4 needs 11 elements, but a holds 10")]
fn a_slice_short_of_the_block_panics() {
    let stream = Stream::new(0);
    let mut device = DeviceBytes::new(10 * size_of::<f64>());
    let _ = symmetrize(&stream, 3, &mut device, 4);
}
