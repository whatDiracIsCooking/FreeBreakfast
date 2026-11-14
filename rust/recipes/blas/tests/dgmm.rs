//! The packed dgmm on a card, against the scaling on the host: small
//! integers, so every product is exact and the comparison is bitwise.

use std::rc::Rc;

use fb_core_wwr::blas::BlasHandle;
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::runtime::tests::DeviceBytes;
use fb_recipes_blas::{BlasSide, Left, Right, dgmm};
use wwr_sys::wwr_blas_side_mode;

fn values(len: usize, seed: usize) -> Vec<f64> {
    (0..len)
        .map(|i| ((i * 7 + seed) % 11) as f64 - 5.0)
        .collect()
}

fn download(device: &DeviceBytes, owner: &Stream) -> Vec<f64> {
    device
        .read(owner)
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| f64::from_ne_bytes(*c))
        .collect()
}

/// One packed m × n dgmm on a card, checked bitwise against the host's.
fn check<Side: BlasSide>((m, n): (usize, usize)) {
    let left = Side::RAW == wwr_blas_side_mode::WWR_BLAS_SIDE_LEFT;
    let a = values(m * n, 1);
    let x = values(if left { m } else { n }, 2);
    let want: Vec<f64> = (0..m * n)
        .map(|e| {
            let (i, j) = (e % m, e / m);
            x[if left { i } else { j }] * a[e]
        })
        .collect();
    let owner = Rc::new(Stream::new(0));
    let blas = BlasHandle::new(owner.clone());
    let (da, dx) = (
        DeviceBytes::upload(&a, &owner),
        DeviceBytes::upload(&x, &owner),
    );
    let mut dc = DeviceBytes::new(m * n * size_of::<f64>());
    let i = |v: usize| v as i32;
    dgmm::<Side, f64>(&blas, i(m), i(n), &da, &dx, &mut dc).unwrap();
    assert_eq!(download(&dc, &owner), want);
}

/// Needs a GPU. Both sides on a rectangular shape, so x's length is m on
/// the left and n on the right, and every leading dimension is m.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn each_side_scales_its_rows_or_columns() {
    check::<Left>((5, 3));
    check::<Right>((5, 3));
}
