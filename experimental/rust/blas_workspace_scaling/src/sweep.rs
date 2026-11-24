//! The same question as main.rs, across the other library calls the tree
//! makes or will make: what device memory does each need beyond its operands,
//! and how does that grow? One routine at one order per process.
//!
//! ```text
//! routine-sweep <routine> <n>
//! ```
//!
//! BLAS routines (`dtrsm_rt strsm ztrsm dtrmm dtrsv dgemm_sb dsyrk dgemv
//! dgeam ddgmm ddot dnrm2`): matrices are n x n, vectors n (`ddot`, `dnrm2`:
//! n^2; `dgemm_sb`: 16 products of order n/4). Read as in main.rs:
//! `touched_bytes` of a large 0xFF-filled workspace, and `own_mib`, the free
//! memory lost over a second call on a 256-byte workspace.
//!
//! Solver routines (`dsyevd dpotrf dpotrs dgetrf dgetrs dgesvd`): the call is
//! handed a work buffer of exactly its size query (`work_mib`; `dpotrs` and
//! `dgetrs` take none), and made twice on fresh inputs. `call1_mib` and
//! `call2_mib` are the free memory lost over each: what the call took beyond
//! its work buffer. The first includes the library's once-per-process
//! first-call growth (about 10 MiB on HIP).
//!
//! Free memory is read after a device sync. Positive means memory taken.

use std::ffi::c_void;
use std::ptr::null_mut;

use wwr_sys::*;

const TINY_BYTES: usize = 256;
const MIB: usize = 1 << 20;

fn die(msg: &str) -> ! {
    eprintln!("routine-sweep: {msg}");
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
    (before - after) as f64 / MIB as f64
}

fn dalloc(bytes: usize) -> *mut c_void {
    let mut p = null_mut();
    // SAFETY: a valid out-pointer.
    rt(unsafe { wwr_malloc(&mut p, bytes.max(8)) }, "malloc");
    p
}

fn upload<T: Copy>(h: &[T]) -> *mut T {
    let bytes = std::mem::size_of_val(h);
    let p = dalloc(bytes);
    // SAFETY: p holds `bytes` bytes.
    let s = unsafe {
        wwr_memcpy(
            p,
            h.as_ptr().cast(),
            bytes,
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

/// A lower triangle of order n with a dominant diagonal, column-major.
fn lower(n: usize) -> Vec<f64> {
    let mut l = random(n * n, 3);
    for j in 0..n {
        for i in 0..j {
            l[i + j * n] = 0.0;
        }
        l[j + j * n] = n as f64;
    }
    l
}

/// A symmetric matrix of order n with a dominant diagonal: positive definite.
fn spd(n: usize) -> Vec<f64> {
    let mut a = random(n * n, 5);
    for j in 0..n {
        for i in 0..j {
            a[i + j * n] = a[j + i * n];
        }
        a[j + j * n] = n as f64;
    }
    a
}

/// One BLAS call on a fresh handle given `ws`; its status and the free memory
/// lost over it, in MiB.
fn blas_call(
    f: &dyn Fn(wwr_blas_handle) -> wwr_blas_status,
    ws: *mut c_void,
    len: usize,
) -> (wwr_blas_status, f64) {
    let mut h = null_mut();
    // SAFETY: valid out-pointer; ws is at least len bytes.
    unsafe {
        blas_ok(wwr_blas_create(&mut h), "blas create");
        blas_ok(wwr_blas_set_workspace(h, ws, len), "set workspace");
    }
    let before = free_bytes();
    let status = f(h);
    (status, mib(before, free_bytes()))
}

/// The two BLAS readings: on a workspace of `big` bytes, then on 256 bytes.
fn blas(
    routine: &str,
    n: i32,
    operand_bytes: usize,
    f: &dyn Fn(wwr_blas_handle) -> wwr_blas_status,
) {
    let big = 2 * operand_bytes + 64 * MIB;
    let ws = dalloc(big);
    // SAFETY: ws is a big-byte allocation.
    rt(unsafe { wwr_memset(ws, 0xFF, big) }, "memset");
    let (big_status, big_call_mib) = blas_call(f, ws, big);
    blas_ok(big_status, "call on the large workspace");
    let touched = touched_bytes(ws, big);
    // SAFETY: nothing uses ws after the sync in `blas_call`.
    rt(unsafe { wwr_free(ws) }, "free workspace");
    let (own_status, own_mib) = blas_call(f, dalloc(TINY_BYTES), TINY_BYTES);
    println!(
        "routine={routine} n={n} operand_mib={:.1} touched_bytes={touched} touched_mib={:.3} \
         big_call_mib={big_call_mib:.2} own_mib={own_mib:.2} own_status={own_status:?}",
        operand_bytes as f64 / MIB as f64,
        touched as f64 / MIB as f64,
    );
}

fn blas_routine(routine: &str, n: i32) {
    let nu = n as usize;
    let (one, zero) = (1.0f64, 0.0f64);
    let left = wwr_blas_side_mode::WWR_BLAS_SIDE_LEFT;
    let right = wwr_blas_side_mode::WWR_BLAS_SIDE_RIGHT;
    let lo = wwr_blas_fill_mode::WWR_BLAS_FILL_MODE_LOWER;
    let op_n = wwr_blas_operation::WWR_BLAS_OP_N;
    let op_t = wwr_blas_operation::WWR_BLAS_OP_T;
    let nonunit = wwr_blas_diag_type::WWR_BLAS_DIAG_NON_UNIT;
    let square = nu * nu * 8;
    // SAFETY, for every closure below: each pointer is a device allocation of
    // the shape its call names, made here and never freed.
    match routine {
        // B <- B L^-T: the density-fitting solve's form.
        "dtrsm_rt" => {
            let (l, b) = (upload(&lower(nu)), upload(&random(nu * nu, 4)));
            blas(routine, n, square, &|h| unsafe {
                wwr_blas_dtrsm(h, right, lo, op_t, nonunit, n, n, &one, l, n, b, n)
            });
        }
        "strsm" => {
            let f32s = |v: Vec<f64>| v.into_iter().map(|x| x as f32).collect::<Vec<f32>>();
            let (l, b) = (upload(&f32s(lower(nu))), upload(&f32s(random(nu * nu, 4))));
            let one = 1.0f32;
            blas(routine, n, square / 2, &|h| unsafe {
                wwr_blas_strsm(h, left, lo, op_n, nonunit, n, n, &one, l, n, b, n)
            });
        }
        "ztrsm" => {
            // Interleaved (re, im) doubles, the layout of wwr_complex128.
            let complex = |v: Vec<f64>| v.into_iter().flat_map(|x| [x, 0.0]).collect::<Vec<f64>>();
            let (l, b) = (
                upload(&complex(lower(nu))),
                upload(&complex(random(nu * nu, 4))),
            );
            let one = [1.0f64, 0.0];
            blas(routine, n, square * 2, &|h| unsafe {
                wwr_blas_ztrsm(
                    h,
                    left,
                    lo,
                    op_n,
                    nonunit,
                    n,
                    n,
                    one.as_ptr().cast(),
                    l.cast(),
                    n,
                    b.cast(),
                    n,
                )
            });
        }
        "dtrmm" => {
            let (l, b) = (upload(&lower(nu)), upload(&random(nu * nu, 4)));
            let c = dalloc(square).cast::<f64>();
            blas(routine, n, square, &|h| unsafe {
                wwr_blas_dtrmm(h, left, lo, op_n, nonunit, n, n, &one, l, n, b, n, c, n)
            });
        }
        "dtrsv" => {
            let (l, x) = (upload(&lower(nu)), upload(&random(nu, 4)));
            blas(routine, n, square, &|h| unsafe {
                wwr_blas_dtrsv(h, lo, op_n, nonunit, n, l, n, x, 1)
            });
        }
        "dgemm_sb" => {
            let (q, batch) = (n / 4, 16);
            let each = (q as usize) * (q as usize);
            let (a, b) = (
                upload(&random(each * batch, 1)),
                upload(&random(each * batch, 2)),
            );
            let c = dalloc(each * batch * 8).cast::<f64>();
            let stride = each as i64;
            blas(routine, n, each * batch * 8, &|h| unsafe {
                wwr_blas_dgemm_strided_batched(
                    h,
                    op_n,
                    op_n,
                    q,
                    q,
                    q,
                    &one,
                    a,
                    q,
                    stride,
                    b,
                    q,
                    stride,
                    &zero,
                    c,
                    q,
                    stride,
                    batch as i32,
                )
            });
        }
        "dsyrk" => {
            let a = upload(&random(nu * nu, 1));
            let c = dalloc(square).cast::<f64>();
            blas(routine, n, square, &|h| unsafe {
                wwr_blas_dsyrk(h, lo, op_n, n, n, &one, a, n, &zero, c, n)
            });
        }
        "dgemv" => {
            let (a, x) = (upload(&random(nu * nu, 1)), upload(&random(nu, 2)));
            let y = dalloc(nu * 8).cast::<f64>();
            blas(routine, n, square, &|h| unsafe {
                wwr_blas_dgemv(h, op_n, n, n, &one, a, n, x, 1, &zero, y, 1)
            });
        }
        "dgeam" => {
            let (a, b) = (upload(&random(nu * nu, 1)), upload(&random(nu * nu, 2)));
            let c = dalloc(square).cast::<f64>();
            blas(routine, n, square, &|h| unsafe {
                wwr_blas_dgeam(h, op_n, op_t, n, n, &one, a, n, &one, b, n, c, n)
            });
        }
        "ddgmm" => {
            let (a, x) = (upload(&random(nu * nu, 1)), upload(&random(nu, 2)));
            let c = dalloc(square).cast::<f64>();
            blas(routine, n, square, &|h| unsafe {
                wwr_blas_ddgmm(h, right, n, n, a, n, x, 1, c, n)
            });
        }
        "ddot" => {
            let (x, y) = (upload(&random(nu * nu, 1)), upload(&random(nu * nu, 2)));
            let len = (nu * nu) as i32;
            blas(routine, n, square, &|h| unsafe {
                let mut r = 0.0f64;
                wwr_blas_ddot(h, len, x, 1, y, 1, &mut r)
            });
        }
        "dnrm2" => {
            let x = upload(&random(nu * nu, 1));
            let len = (nu * nu) as i32;
            blas(routine, n, square, &|h| unsafe {
                let mut r = 0.0f64;
                wwr_blas_dnrm2(h, len, x, 1, &mut r)
            });
        }
        _ => die(&format!("unknown routine {routine}")),
    }
}

fn solver_routine(routine: &str, n: i32) {
    let nu = n as usize;
    let lo = wwr_solver_fill_mode::WWR_SOLVER_FILL_MODE_LOWER;
    let mut h = null_mut();
    // SAFETY: a valid out-pointer.
    solver_ok(unsafe { wwr_solver_dn_create(&mut h) }, "solver create");
    let info = dalloc(8).cast::<i32>();
    let ipiv = dalloc(nu * 4).cast::<i32>();
    let mut lwork = 0i32;
    let mut prep_mib = 0.0f64;
    // Two fresh inputs, one per measured call.
    let general = || {
        let mut a = random(nu * nu, 7);
        for j in 0..nu {
            a[j + j * nu] = n as f64;
        }
        a
    };
    let host = match routine {
        "dsyevd" | "dpotrf" | "dpotrs" => spd(nu),
        _ => general(),
    };
    let a = [upload(&host), upload(&host)];
    let b = [upload(&random(nu * nu, 8)), upload(&random(nu * nu, 9))];
    drop(host);
    // SAFETY, for every call below: each pointer is a device allocation of
    // the shape the call names; `work` holds lwork doubles.
    let call: Box<dyn Fn(usize) -> wwr_solver_status> = match routine {
        "dsyevd" => {
            let jobz = wwr_solver_eig_mode::WWR_SOLVER_EIG_MODE_VECTOR;
            let w = dalloc(nu * 8).cast::<f64>();
            solver_ok(
                unsafe { wwr_solver_dn_dsyevd_buffer_size(h, jobz, lo, n, a[0], n, w, &mut lwork) },
                "size query",
            );
            let work = dalloc(lwork as usize * 8).cast::<f64>();
            Box::new(move |i| unsafe {
                wwr_solver_dn_dsyevd(h, jobz, lo, n, a[i], n, w, work, lwork, info)
            })
        }
        "dpotrf" | "dpotrs" => {
            solver_ok(
                unsafe { wwr_solver_dn_dpotrf_buffer_size(h, lo, n, a[0], n, &mut lwork) },
                "size query",
            );
            let work = dalloc(lwork as usize * 8).cast::<f64>();
            let factor = move |i: usize| unsafe {
                wwr_solver_dn_dpotrf(h, lo, n, a[i], n, work, lwork, info)
            };
            if routine == "dpotrf" {
                Box::new(factor)
            } else {
                // Factor both inputs first; the solves are what is measured.
                let before = free_bytes();
                solver_ok(factor(0), "potrf");
                solver_ok(factor(1), "potrf");
                prep_mib = mib(before, free_bytes());
                lwork = 0;
                Box::new(move |i| unsafe {
                    wwr_solver_dn_dpotrs(h, lo, n, n, a[i], n, b[i], n, info)
                })
            }
        }
        "dgetrf" | "dgetrs" => {
            solver_ok(
                unsafe { wwr_solver_dn_dgetrf_buffer_size(h, n, n, a[0], n, &mut lwork) },
                "size query",
            );
            let work = dalloc(lwork as usize * 8).cast::<f64>();
            let factor =
                move |i: usize| unsafe { wwr_solver_dn_dgetrf(h, n, n, a[i], n, work, ipiv, info) };
            if routine == "dgetrf" {
                Box::new(factor)
            } else {
                // One pivot array, so both solves use the first input's
                // factorization.
                let before = free_bytes();
                solver_ok(factor(0), "getrf");
                prep_mib = mib(before, free_bytes());
                lwork = 0;
                let op = wwr_solver_operation::WWR_SOLVER_OP_N;
                Box::new(move |i| unsafe {
                    wwr_solver_dn_dgetrs(h, op, n, n, a[0], n, ipiv, b[i], n, info)
                })
            }
        }
        "dgesvd" => {
            solver_ok(
                unsafe { wwr_solver_dn_dgesvd_buffer_size(h, n, n, &mut lwork) },
                "size query",
            );
            let work = dalloc(lwork as usize * 8).cast::<f64>();
            let s = dalloc(nu * 8).cast::<f64>();
            let rwork = dalloc(nu * 8).cast::<f64>();
            let (u, vt) = (b[0], b[1]);
            let all = b'A' as std::ffi::c_schar;
            Box::new(move |i| unsafe {
                wwr_solver_dn_dgesvd(
                    h, all, all, n, n, a[i], n, s, u, n, vt, n, work, lwork, rwork, info,
                )
            })
        }
        _ => die(&format!("unknown routine {routine}")),
    };
    print!(
        "routine={routine} n={n} operand_mib={:.1} lwork={lwork} work_mib={:.3} prep_mib={prep_mib:.2}",
        (nu * nu * 8) as f64 / MIB as f64,
        lwork as f64 * 8.0 / MIB as f64
    );
    for i in 0..2 {
        let before = free_bytes();
        let status = call(i);
        print!(
            " call{}_mib={:.2} status{}={status:?}",
            i + 1,
            mib(before, free_bytes()),
            i + 1
        );
    }
    println!();
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let usage = "usage: routine-sweep <routine> <n>";
    let routine = args.first().unwrap_or_else(|| die(usage)).as_str();
    let n: i32 = args
        .get(1)
        .and_then(|s| s.parse().ok())
        .filter(|&n| n > 0)
        .unwrap_or_else(|| die(usage));
    // SAFETY: plain runtime calls; the free creates the context before any
    // reading.
    unsafe {
        rt(wwr_set_device(0), "set device");
        rt(wwr_free(null_mut()), "context");
    }
    match routine {
        "dsyevd" | "dpotrf" | "dpotrs" | "dgetrf" | "dgetrs" | "dgesvd" => {
            solver_routine(routine, n)
        }
        _ => blas_routine(routine, n),
    }
}
