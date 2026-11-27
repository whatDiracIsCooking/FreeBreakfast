//! The handle ring's two measurements (the v0.0.4 prototype's page at b0935805, "A ring of
//! library handles, and no library stream"), through wwr-sys, the path the
//! Rust tree takes. One case per process; run_all.sh runs them all under the
//! card's lock.
//!
//! ```text
//! handle-ring handles-blas <count>
//! handle-ring handles-solver <count>
//! handle-ring ring <dgemm_square|dgemm_splitk|dtrsm> <stream_then_ws|ws_then_stream|stream_only>[_tiny]
//! ```
//!
//! `handles-*`: that many handles in one process, each on its own stream; the
//! card's free memory read around each creation, each `SetWorkspace` and each
//! handle's first call. Picks between a ring and a handle per step.
//! `handles-solver` then moves the first handle onto the second's stream, idle,
//! and calls it again, and times `SetStream` alone.
//!
//! `ring`: one BLAS handle moved between two streams, each with its own
//! sentinel-filled 32 MiB workspace, four calls alternating. `stream_then_ws`
//! sets the stream and then the workspace before each call (the proposal's
//! order), `ws_then_stream` the reverse, `stream_only` sets the first
//! workspace once and afterwards only the stream. `touched_mib` is how much of
//! the workspace given (in `stream_only`, the first) the call wrote: whether
//! the call used it. cuBLAS writes none of a 32 MiB workspace on these shapes,
//! so each mode also has a `_tiny` form, every workspace given a length of 256
//! bytes: cuBLAS then picks another split-K algorithm, and `max_diff_vs_tiny`
//! of 0 against a nonzero `max_diff_vs_default` says the call saw the tiny
//! length. `dtrsm` is the case HIP reads: rocBLAS writes its workspace there,
//! so `touched_mib` answers, and `call_mib` shows a call growing past a
//! workspace too small. Then the idle switch is timed.
//!
//! Deltas are the drop in the card's free memory over the step, in MiB, after
//! a device sync: positive means the step took memory.

use std::ffi::c_void;
use std::ptr::null_mut;
use std::time::Instant;

use wwr_sys::*;

const WS_BYTES: usize = 32 << 20;
const SOLVER_N: i32 = 2048;
const SWITCHES: u32 = 1000;
const TINY_BYTES: usize = 256;

fn die(msg: &str) -> ! {
    eprintln!("handle-ring: {msg}");
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

fn solver_ok(s: wwr_solver_status, what: &str) {
    if s != wwr_solver_status::WWR_SOLVER_STATUS_SUCCESS {
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
    (before - after) as f64 / f64::from(1u32 << 20)
}

fn us_since(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1e6
}

fn dalloc(bytes: usize) -> *mut c_void {
    let mut p = null_mut();
    // SAFETY: a valid out-pointer.
    rt(unsafe { wwr_malloc(&mut p, bytes) }, "malloc");
    p
}

fn upload_into(dst: *mut f64, h: &[f64]) {
    // SAFETY: dst holds at least h.len() doubles (every caller allocates so).
    let s = unsafe {
        wwr_memcpy(
            dst.cast(),
            h.as_ptr().cast(),
            std::mem::size_of_val(h),
            wwr_memcpy_kind::WWR_MEMCPY_HOST_TO_DEVICE,
        )
    };
    rt(s, "upload");
}

fn upload(h: &[f64]) -> *mut f64 {
    let p = dalloc(std::mem::size_of_val(h)).cast();
    upload_into(p, h);
    p
}

fn download_f64(src: *const f64, n: usize) -> Vec<f64> {
    let mut h = vec![0.0; n];
    // SAFETY: src holds n doubles.
    let s = unsafe {
        wwr_memcpy(
            h.as_mut_ptr().cast(),
            src.cast(),
            n * 8,
            wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_HOST,
        )
    };
    rt(s, "download");
    h
}

/// The highest byte of a 0xFF-filled workspace the library changed, plus one,
/// in MiB.
fn touched_mib(ws: *const c_void) -> f64 {
    let mut h = vec![0u8; WS_BYTES];
    // SAFETY: ws is a WS_BYTES allocation.
    let s = unsafe {
        wwr_memcpy(
            h.as_mut_ptr().cast(),
            ws,
            WS_BYTES,
            wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_HOST,
        )
    };
    rt(s, "download workspace");
    let end = h.iter().rposition(|&b| b != 0xFF).map_or(0, |i| i + 1);
    end as f64 / f64::from(1u32 << 20)
}

fn fill_sentinel(ws: *mut c_void) {
    // SAFETY: ws is a WS_BYTES allocation.
    rt(unsafe { wwr_memset(ws, 0xFF, WS_BYTES) }, "memset");
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

/// A symmetric matrix of order n, column-major.
fn symmetric(n: usize) -> Vec<f64> {
    let r = random(n * n, 7);
    let mut a = vec![0.0; n * n];
    for j in 0..n {
        for i in 0..n {
            a[i + j * n] = r[i + j * n] + r[j + i * n] + if i == j { n as f64 } else { 0.0 };
        }
    }
    a
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f64::max)
}

fn stream_create() -> wwr_stream {
    let mut s = null_mut();
    // SAFETY: a valid out-pointer.
    rt(unsafe { wwr_stream_create(&mut s) }, "stream create");
    s
}

fn blas_create() -> wwr_blas_handle {
    let mut h = null_mut();
    // SAFETY: a valid out-pointer.
    blas_ok(unsafe { wwr_blas_create(&mut h) }, "blas create");
    h
}

/// The BLAS call a ring case makes, column-major, into `out`.
enum Routine {
    /// out = A B, A m x k, B k x n.
    Gemm {
        m: i32,
        n: i32,
        k: i32,
        a: *const f64,
        b: *const f64,
    },
    /// out = L^-1 B in place, L lower m x m, B m x n: the DF fit's shape.
    /// rocBLAS takes device memory for it, where gemm takes none.
    Trsm {
        m: i32,
        n: i32,
        l: *const f64,
        b: *const f64,
    },
}

impl Routine {
    fn gemm(m: i32, n: i32, k: i32) -> Self {
        let (mu, nu, ku) = (m as usize, n as usize, k as usize);
        Self::Gemm {
            m,
            n,
            k,
            a: upload(&random(mu * ku, 1)),
            b: upload(&random(ku * nu, 2)),
        }
    }

    fn trsm(m: i32, n: i32) -> Self {
        let (mu, nu) = (m as usize, n as usize);
        let mut l = random(mu * mu, 3);
        for j in 0..mu {
            for i in 0..j {
                l[i + j * mu] = 0.0;
            }
            l[j + j * mu] = m as f64;
        }
        Self::Trsm {
            m,
            n,
            l: upload(&l),
            b: upload(&random(mu * nu, 4)),
        }
    }

    fn len_out(&self) -> usize {
        match *self {
            Self::Gemm { m, n, .. } | Self::Trsm { m, n, .. } => m as usize * n as usize,
        }
    }

    fn call(&self, h: wwr_blas_handle, out: *mut f64) -> wwr_blas_status {
        let (one, zero) = (1.0f64, 0.0f64);
        let op = wwr_blas_operation::WWR_BLAS_OP_N;
        match *self {
            // SAFETY: a, b and out are allocations of the shapes given.
            Self::Gemm { m, n, k, a, b } => unsafe {
                wwr_blas_dgemm(h, op, op, m, n, k, &one, a, m, b, k, &zero, out, m)
            },
            Self::Trsm { m, n, l, b } => {
                // SAFETY: b and out hold m x n doubles; a blocking copy, so it
                // is done before the call is enqueued on any stream.
                rt(
                    unsafe {
                        wwr_memcpy(
                            out.cast(),
                            b.cast(),
                            self.len_out() * 8,
                            wwr_memcpy_kind::WWR_MEMCPY_DEVICE_TO_DEVICE,
                        )
                    },
                    "copy B",
                );
                // SAFETY: l is m x m, out m x n.
                unsafe {
                    wwr_blas_dtrsm(
                        h,
                        wwr_blas_side_mode::WWR_BLAS_SIDE_LEFT,
                        wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER,
                        op,
                        wwr_blas_diag_type::WWR_BLAS_DIAG_NON_UNIT,
                        m,
                        n,
                        &one,
                        l,
                        m,
                        out,
                        m,
                    )
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// One BLAS handle moved between two streams, a workspace per stream.
// ---------------------------------------------------------------------------

fn ring(routine: &str, mode: &str) {
    let g = match routine {
        "dgemm_square" => Routine::gemm(4096, 4096, 4096),
        "dgemm_splitk" => Routine::gemm(256, 256, 1 << 16),
        "dtrsm" => Routine::trsm(4096, 4096),
        _ => die(&format!("unknown routine {routine}")),
    };
    // A `_tiny` mode gives each workspace a length of TINY_BYTES: cuBLAS then
    // picks another split-K algorithm, so the result says which workspace the
    // call saw even where it writes none.
    let (order, len) = match mode.strip_suffix("_tiny") {
        Some(order) => (order, TINY_BYTES),
        None => (mode, WS_BYTES),
    };
    if !matches!(order, "stream_then_ws" | "ws_then_stream" | "stream_only") {
        die(&format!("unknown ring mode {mode}"));
    }
    let c = dalloc(g.len_out() * 8).cast::<f64>();
    let c_ref = dalloc(g.len_out() * 8).cast::<f64>();
    let streams = [stream_create(), stream_create()];
    let ws = [dalloc(WS_BYTES), dalloc(WS_BYTES)];

    // The reference: a default handle on the default stream.
    let href = blas_create();
    let r0 = free_bytes();
    blas_ok(g.call(href, c_ref), "reference call");
    // What the default handle takes for the call: rocBLAS's pool growing.
    let ref_call_mib = mib(r0, free_bytes());
    let reference = download_f64(c_ref, g.len_out());
    // And one with a tiny workspace, on the default stream.
    let htiny = blas_create();
    let tiny_ws = dalloc(TINY_BYTES);
    // SAFETY: a live handle and a TINY_BYTES workspace.
    blas_ok(
        unsafe { wwr_blas_set_workspace(htiny, tiny_ws, TINY_BYTES) },
        "set tiny workspace",
    );
    blas_ok(g.call(htiny, c_ref), "tiny reference call");
    free_bytes();
    let reference_tiny = download_f64(c_ref, g.len_out());

    let f0 = free_bytes();
    let h = blas_create();
    let f1 = free_bytes();
    print!(
        "routine={routine} mode={mode} handle_mib={:.2} ref_call_mib={ref_call_mib:.2} \
         tiny_vs_default={:.3e}",
        mib(f0, f1),
        max_abs_diff(&reference_tiny, &reference)
    );
    for call in 0..4 {
        let i = call % 2;
        // Which workspace this call should use, and so which one to read.
        let read = if order == "stream_only" { 0 } else { i };
        fill_sentinel(ws[read]);
        let before = free_bytes();
        let t0 = Instant::now();
        // SAFETY: a live handle, stream and WS_BYTES workspace; the handle is
        // idle (free_bytes synced the device).
        let (s_stream, s_ws) = unsafe {
            match order {
                "stream_then_ws" => {
                    let s = wwr_blas_set_stream(h, streams[i]);
                    (s, wwr_blas_set_workspace(h, ws[i], len))
                }
                "ws_then_stream" => {
                    let w = wwr_blas_set_workspace(h, ws[i], len);
                    (wwr_blas_set_stream(h, streams[i]), w)
                }
                _ => {
                    let w = if call == 0 {
                        wwr_blas_set_workspace(h, ws[0], len)
                    } else {
                        wwr_blas_status::WWR_BLAS_STATUS_SUCCESS
                    };
                    (wwr_blas_set_stream(h, streams[i]), w)
                }
            }
        };
        let set_us = us_since(t0);
        let after_set = free_bytes();
        let s = g.call(h, c);
        let after_call = free_bytes();
        let (diff, diff_tiny) = if s == wwr_blas_status::WWR_BLAS_STATUS_SUCCESS {
            let out = download_f64(c, g.len_out());
            (
                max_abs_diff(&out, &reference),
                max_abs_diff(&out, &reference_tiny),
            )
        } else {
            (-1.0, -1.0)
        };
        print!(
            " | call{call} stream={i} set_stream={s_stream:?} set_workspace={s_ws:?} \
             set_us={set_us:.1} set_mib={:.2} status={s:?} call_mib={:.2} \
             touched_mib={:.3} max_diff_vs_default={diff:.3e} \
             max_diff_vs_tiny={diff_tiny:.3e}",
            mib(before, after_set),
            mib(after_set, after_call),
            touched_mib(ws[read]),
        );
    }
    // The idle switch, timed: SWITCHES moves between the two streams.
    let before = free_bytes();
    let t0 = Instant::now();
    for n in 0..SWITCHES as usize {
        let i = n % 2;
        // SAFETY: as above; nothing is enqueued.
        unsafe {
            blas_ok(wwr_blas_set_stream(h, streams[i]), "set stream");
            blas_ok(wwr_blas_set_workspace(h, ws[i], WS_BYTES), "set workspace");
        }
    }
    let pair_us = us_since(t0) / f64::from(SWITCHES);
    let t0 = Instant::now();
    for n in 0..SWITCHES as usize {
        // SAFETY: as above.
        blas_ok(
            unsafe { wwr_blas_set_stream(h, streams[n % 2]) },
            "set stream",
        );
    }
    let stream_us = us_since(t0) / f64::from(SWITCHES);
    println!(
        " | idle_switch stream_and_ws_us={pair_us:.2} stream_only_us={stream_us:.2} \
         switches_mib={:.2}",
        mib(before, free_bytes())
    );
}

// ---------------------------------------------------------------------------
// Several handles in one process: what does each one after the first cost?
// ---------------------------------------------------------------------------

fn handles_blas(count: usize) {
    let g = Routine::gemm(2048, 2048, 2048);
    let c = dalloc(g.len_out() * 8).cast::<f64>();
    let streams: Vec<_> = (0..count).map(|_| stream_create()).collect();
    let ws: Vec<_> = (0..count).map(|_| dalloc(WS_BYTES)).collect();
    let mut h = Vec::with_capacity(count);
    print!("routine=handles_blas count={count}");
    for i in 0..count {
        let f0 = free_bytes();
        let hi = blas_create();
        // SAFETY: a live handle and stream.
        blas_ok(unsafe { wwr_blas_set_stream(hi, streams[i]) }, "set stream");
        let f1 = free_bytes();
        // SAFETY: a live handle and a WS_BYTES workspace.
        blas_ok(
            unsafe { wwr_blas_set_workspace(hi, ws[i], WS_BYTES) },
            "set workspace",
        );
        let f2 = free_bytes();
        print!(
            " | handle{i} create_mib={:.2} set_workspace_mib={:.2}",
            mib(f0, f1),
            mib(f1, f2)
        );
        h.push(hi);
    }
    let mut reference = Vec::new();
    for (i, &hi) in h.iter().enumerate() {
        let f0 = free_bytes();
        blas_ok(g.call(hi, c), "gemm");
        let f1 = free_bytes();
        let out = download_f64(c, g.len_out());
        if i == 0 {
            reference = out.clone();
        }
        print!(
            " | handle{i} first_call_mib={:.2} max_diff={:.3e}",
            mib(f0, f1),
            max_abs_diff(&out, &reference)
        );
    }
    println!();
}

fn handles_solver(count: usize) {
    let n = SOLVER_N as usize;
    let a_host = symmetric(n);
    let a = dalloc(n * n * 8).cast::<f64>();
    let w = dalloc(n * 8).cast::<f64>();
    let info = dalloc(4).cast::<i32>();
    let streams: Vec<_> = (0..count).map(|_| stream_create()).collect();
    let mut h: Vec<wwr_solver_handle> = Vec::with_capacity(count);
    print!("routine=handles_solver count={count}");
    for (i, &stream) in streams.iter().enumerate() {
        let f0 = free_bytes();
        let mut hi = null_mut();
        // SAFETY: a valid out-pointer, then a live handle and stream.
        unsafe {
            solver_ok(wwr_solver_dn_create(&mut hi), "solver create");
            solver_ok(wwr_solver_dn_set_stream(hi, stream), "solver set stream");
        }
        let f1 = free_bytes();
        print!(" | handle{i} create_mib={:.2}", mib(f0, f1));
        h.push(hi);
    }
    let jobz = wwr_solver_eig_mode::WWR_SOLVER_EIG_MODE_VECTOR;
    let uplo = wwr_solver_fill_mode::WWR_SOLVER_FILL_MODE_LOWER;
    let mut lwork = 0i32;
    // SAFETY: a live handle; a and w are allocations of order n.
    solver_ok(
        unsafe {
            wwr_solver_dn_dsyevd_buffer_size(h[0], jobz, uplo, SOLVER_N, a, SOLVER_N, w, &mut lwork)
        },
        "syevd query",
    );
    let work = dalloc(lwork as usize * 8).cast::<f64>();
    // One syevd on a fresh copy of A; the eigenvalues back.
    let syevd = |hi: wwr_solver_handle| -> (wwr_solver_status, Vec<f64>) {
        upload_into(a, &a_host);
        // SAFETY: a live, idle handle; buffers of the queried sizes.
        let s = unsafe {
            wwr_solver_dn_dsyevd(hi, jobz, uplo, SOLVER_N, a, SOLVER_N, w, work, lwork, info)
        };
        free_bytes();
        (s, download_f64(w, n))
    };
    let mut reference = Vec::new();
    for (i, &hi) in h.iter().enumerate() {
        let f0 = free_bytes();
        let (s, out) = syevd(hi);
        let f1 = free_bytes();
        if i == 0 {
            reference = out.clone();
        }
        print!(
            " | handle{i} first_call_status={s:?} first_call_mib={:.2} max_diff={:.3e}",
            mib(f0, f1),
            max_abs_diff(&out, &reference)
        );
    }
    if count > 1 {
        // The ring's move: the first handle, idle, onto the second's stream.
        let f0 = free_bytes();
        let t0 = Instant::now();
        // SAFETY: a live, idle handle and a live stream.
        let ss = unsafe { wwr_solver_dn_set_stream(h[0], streams[1]) };
        let set_us = us_since(t0);
        let (s, out) = syevd(h[0]);
        let f1 = free_bytes();
        print!(
            " | moved set_stream={ss:?} set_us={set_us:.1} status={s:?} mib={:.2} max_diff={:.3e}",
            mib(f0, f1),
            max_abs_diff(&out, &reference)
        );
        let t0 = Instant::now();
        for k in 0..SWITCHES as usize {
            // SAFETY: as above; nothing is enqueued.
            solver_ok(
                unsafe { wwr_solver_dn_set_stream(h[0], streams[k % 2]) },
                "set stream",
            );
        }
        print!(
            " | idle_switch set_stream_us={:.2}",
            us_since(t0) / f64::from(SWITCHES)
        );
    }
    println!();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: handle-ring handles-blas|handles-solver <count> | \
                 handle-ring ring <dgemm_square|dgemm_splitk|dtrsm> <stream_then_ws|ws_then_stream|stream_only>[_tiny]";
    let count = |s: Option<&String>| -> usize {
        s.and_then(|s| s.parse().ok())
            .filter(|&n| n > 0)
            .unwrap_or_else(|| die(usage))
    };
    // SAFETY: plain runtime calls; the free creates the context before any
    // reading.
    unsafe {
        rt(wwr_set_device(0), "set device");
        rt(wwr_free(null_mut()), "context");
    }
    match args.first().map(String::as_str) {
        Some("handles-blas") => handles_blas(count(args.get(1))),
        Some("handles-solver") => handles_solver(count(args.get(1))),
        Some("ring") if args.len() == 3 => ring(&args[1], &args[2]),
        _ => die(usage),
    }
}
