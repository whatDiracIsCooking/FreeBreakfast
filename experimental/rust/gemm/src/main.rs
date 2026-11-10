//! gemm-demo: C = A B^T in fp64 on the GPU, from Rust, checked against a
//! host loop, then timed.
//!
//! Usage: gemm-demo [M [N [K [REPS]]]]   (defaults 512 512 512 20)
//!
//! A is M x K and B is N x K, both column-major and filled from a fixed
//! xorshift stream, so a run is reproducible. Exits 1 if the GPU result
//! strays from the host one by more than 1e-12 relative to the largest |C|.

use std::process::ExitCode;
use std::time::Instant;

use fb_gemm::{Gpu, Matrix, Op, gemm};

/// Uniform in [-1, 1), from a xorshift64 state.
fn fill(n: usize, state: &mut u64) -> Vec<f64> {
    (0..n)
        .map(|_| {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            (*state >> 11) as f64 / (1u64 << 53) as f64 * 2.0 - 1.0
        })
        .collect()
}

/// C = A B^T on the host, column-major, A m x k, B n x k.
fn host_gemm_abt(m: usize, n: usize, k: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for j in 0..n {
        for p in 0..k {
            let bjp = b[j + p * n];
            let a_col = &a[p * m..(p + 1) * m];
            let c_col = &mut c[j * m..(j + 1) * m];
            for (ci, ai) in c_col.iter_mut().zip(a_col) {
                *ci += ai * bjp;
            }
        }
    }
    c
}

fn arg(args: &[String], i: usize, default: usize) -> Result<usize, String> {
    args.get(i)
        .map(|s| {
            s.parse()
                .map_err(|_| format!("argument {i} ({s:?}) is not a size"))
        })
        .unwrap_or(Ok(default))
}

fn run() -> Result<bool, Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let m = arg(&args, 1, 512)?;
    let n = arg(&args, 2, m)?;
    let k = arg(&args, 3, m)?;
    let reps = arg(&args, 4, 20)?.max(1);

    let mut state = 0x9e37_79b9_7f4a_7c15;
    let a_host = fill(m * k, &mut state);
    let b_host = fill(n * k, &mut state);

    let gpu = Gpu::new(0)?;
    let a = Matrix::from_host(&gpu, m, k, &a_host)?;
    let b = Matrix::from_host(&gpu, n, k, &b_host)?;
    let mut c = Matrix::zeros(&gpu, m, n)?;

    gemm(&gpu, 1.0, Op::N, &a, Op::T, &b, 0.0, &mut c)?;
    let c_gpu = c.to_vec()?;

    let t = Instant::now();
    let c_ref = host_gemm_abt(m, n, k, &a_host, &b_host);
    let host_s = t.elapsed().as_secs_f64();

    let scale = c_ref
        .iter()
        .fold(0.0f64, |s, x| s.max(x.abs()))
        .max(f64::MIN_POSITIVE);
    let max_err = c_gpu
        .iter()
        .zip(&c_ref)
        .fold(0.0f64, |e, (g, r)| e.max((g - r).abs()));
    let rel = max_err / scale;
    let ok = rel <= 1e-12;

    // Warm, then time `reps` calls between two syncs.
    gemm(&gpu, 1.0, Op::N, &a, Op::T, &b, 0.0, &mut c)?;
    gpu.sync()?;
    let t = Instant::now();
    for _ in 0..reps {
        gemm(&gpu, 1.0, Op::N, &a, Op::T, &b, 0.0, &mut c)?;
    }
    gpu.sync()?;
    let gpu_s = t.elapsed().as_secs_f64() / reps as f64;
    let flops = 2.0 * m as f64 * n as f64 * k as f64;

    println!("C = A B^T, A {m} x {k}, B {n} x {k}, fp64");
    println!(
        "  max |C_gpu - C_host| = {max_err:.3e}  (relative {rel:.3e}, bar 1e-12): {}",
        if ok { "ok" } else { "FAIL" }
    );
    println!(
        "  gpu  {:9.3} ms/call  {:8.1} GFLOP/s  (mean of {reps})",
        gpu_s * 1e3,
        flops / gpu_s / 1e9
    );
    println!(
        "  host {:9.3} ms       {:8.1} GFLOP/s  (one naive loop)",
        host_s * 1e3,
        flops / host_s / 1e9
    );
    Ok(ok)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("gemm-demo: {e}");
            ExitCode::FAILURE
        }
    }
}
