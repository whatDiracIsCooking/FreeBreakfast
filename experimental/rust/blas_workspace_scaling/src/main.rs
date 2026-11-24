//! What device memory a `dtrsm` needs from its BLAS handle, against its shape
//! (docs/rust-redesign.html, "One handle set per card, under one lock": is a
//! fixed 128 MiB workspace enough?). Through wwr-sys, one shape per process;
//! run_all.sh runs the shapes under the card's lock.
//!
//! ```text
//! blas-workspace-scaling <left|right> <m> <n>
//! ```
//!
//! B is m x n; L is lower, of order m (`left`: B <- L^-1 B) or n (`right`:
//! B <- B L^-1). Two calls, each on a fresh handle:
//!
//! - `touched_bytes`: the handle is given a 0xFF-filled workspace larger than
//!   any plausible need; the highest byte the call changed, plus one, is what
//!   the call used of it. `big_call_mib` is the card's free memory lost over
//!   that call, and should be 0: the workspace covered it.
//! - `own_mib`: the handle is given a 256-byte workspace; the free memory lost
//!   over the call is what the library took by itself, outside the caller's.
//!
//! Free memory is read after a device sync. Positive means memory taken.

use std::ffi::c_void;
use std::ptr::null_mut;

use wwr_sys::*;

const TINY_BYTES: usize = 256;
const MIB: usize = 1 << 20;

fn die(msg: &str) -> ! {
    eprintln!("blas-workspace-scaling: {msg}");
    std::process::exit(1)
}

fn rt(s: wwr_status, what: &str) {
    if s != wwr_status::WWR_STATUS_SUCCESS {
        die(&format!("{what}: {s:?}"));
    }
}

fn blas_ok(s: wwr_blas_status, what: &str) {
    if s != wwr_blas_status::WWR_BLAS_STATUS_SUCCESS {
        die(&format!("{what}: {s:?}"));
    }
}

/// The card's free bytes, after a device sync.
fn free_bytes() -> i64 {
    let (mut free, mut total) = (0usize, 0usize);
    // SAFETY: plain runtime calls with valid out-pointers.
    unsafe {
        rt(wwr_device_synchronize(), "device sync");
        rt(wwr_mem_get_info(&mut free, &mut total), "mem_get_info");
    }
    free as i64
}

fn mib(before: i64, after: i64) -> f64 {
    (before - after) as f64 / MIB as f64
}

fn dalloc(bytes: usize) -> *mut c_void {
    let mut p = null_mut();
    // SAFETY: a valid out-pointer.
    rt(unsafe { wwr_malloc(&mut p, bytes) }, "malloc");
    p
}

fn upload(h: &[f64]) -> *mut f64 {
    let p = dalloc(std::mem::size_of_val(h)).cast();
    // SAFETY: p holds h.len() doubles.
    let s = unsafe {
        wwr_memcpy(
            p,
            h.as_ptr().cast(),
            std::mem::size_of_val(h),
            wwr_memcpy_kind::WWR_MEMCPY_HOST_TO_DEVICE,
        )
    };
    rt(s, "upload");
    p.cast()
}

/// The highest byte of a 0xFF-filled workspace the library changed, plus one.
fn touched_bytes(ws: *const c_void, len: usize) -> usize {
    let mut h = vec![0u8; len];
    // SAFETY: ws is a len-byte allocation.
    let s = unsafe {
        wwr_memcpy(
            h.as_mut_ptr().cast(),
            ws,
            len,
            wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_HOST,
        )
    };
    rt(s, "download workspace");
    h.iter().rposition(|&b| b != 0xFF).map_or(0, |i| i + 1)
}

fn random(n: usize, seed: u64) -> Vec<f64> {
    let mut x = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            x = x
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (x >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        })
        .collect()
}

struct Shape {
    side: wwr_blas_side_mode,
    m: i32,
    n: i32,
    k: i32,
    l: *const f64,
}

impl Shape {
    /// A new handle with `ws` set, one solve in place on `b`; the free memory
    /// lost over the call, in MiB.
    fn call(&self, ws: *mut c_void, len: usize, b: *mut f64) -> f64 {
        let mut h = null_mut();
        let one = 1.0f64;
        // SAFETY: valid out-pointer; ws is at least len bytes; l is k x k and
        // b is m x n.
        unsafe {
            blas_ok(wwr_blas_create(&mut h), "blas create");
            blas_ok(wwr_blas_set_workspace(h, ws, len), "set workspace");
            let before = free_bytes();
            blas_ok(
                wwr_blas_dtrsm(
                    h,
                    self.side,
                    wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER,
                    wwr_blas_operation::WWR_BLAS_OP_N,
                    wwr_blas_diag_type::WWR_BLAS_DIAG_NON_UNIT,
                    self.m,
                    self.n,
                    &one,
                    self.l,
                    self.k,
                    b,
                    self.m,
                ),
                "dtrsm",
            );
            mib(before, free_bytes())
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: blas-workspace-scaling <left|right> <m> <n>";
    let dim = |i: usize| -> i32 {
        args.get(i)
            .and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| die(usage))
    };
    let (m, n) = (dim(1), dim(2));
    let (side, k) = match args.first().map(String::as_str) {
        Some("left") => (wwr_blas_side_mode::WWR_BLAS_SIDE_LEFT, m),
        Some("right") => (wwr_blas_side_mode::WWR_BLAS_SIDE_RIGHT, n),
        _ => die(usage),
    };
    // SAFETY: plain runtime calls; the free creates the context before any
    // reading.
    unsafe {
        rt(wwr_set_device(0), "set device");
        rt(wwr_free(null_mut()), "context");
    }
    let (mu, nu, ku) = (m as usize, n as usize, k as usize);
    let mut l = random(ku * ku, 3);
    for j in 0..ku {
        for i in 0..j {
            l[i + j * ku] = 0.0;
        }
        l[j + j * ku] = f64::from(k);
    }
    let shape = Shape {
        side,
        m,
        n,
        k,
        l: upload(&l),
    };
    drop(l);
    let rhs = random(mu * nu, 4);
    let (b_big, b_tiny) = (upload(&rhs), upload(&rhs));

    // Larger than three copies of B, so a temporary the size of B still fits.
    let big = 3 * mu * nu * 8 + 64 * MIB;
    let ws = dalloc(big);
    // SAFETY: ws is a big-byte allocation.
    rt(unsafe { wwr_memset(ws, 0xFF, big) }, "memset");
    let big_call_mib = shape.call(ws, big, b_big);
    let touched = touched_bytes(ws, big);
    // SAFETY: nothing uses ws after the sync in `call`.
    rt(unsafe { wwr_free(ws) }, "free workspace");

    let own_mib = shape.call(dalloc(TINY_BYTES), TINY_BYTES, b_tiny);

    println!(
        "side={} m={m} n={n} k={k} touched_bytes={touched} touched_mib={:.3} \
         big_call_mib={big_call_mib:.2} own_mib={own_mib:.2} big_ws_mib={}",
        args[0],
        touched as f64 / MIB as f64,
        big / MIB
    );
}
