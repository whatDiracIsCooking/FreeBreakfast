//! `eigh-bench --threads T --out DIR MATRIX.npy...`: the full symmetric
//! eigendecomposition of each matrix by this build's backend (`--features
//! faer` or `--features lapack`), timed at T threads, eigenvalues alone and
//! with eigenvectors.
//!
//! Each `MATRIX.npy` is a packed lower triangle as `s-overlap-host` and
//! `df-metric-host` write it, 1-D of `n (n + 1) / 2`, `(p, q)` with `q <= p`
//! at `p (p + 1) / 2 + q`; it is unpacked to a full column-major n × n. Every
//! (matrix, routine, eigenvectors or not) appends one JSON line to
//! `DIR/<backend>.jsonl` and writes its eigenvalues to
//! `DIR/eigvals/<backend>-<routine>-<matrix>-<vals|vecs>-t<T>.npy`, which
//! `compare.py` reads across the two backends.

#[cfg(all(feature = "faer", feature = "lapack"))]
compile_error!("build one backend: --features faer or --features lapack, not both");
#[cfg(not(any(feature = "faer", feature = "lapack")))]
compile_error!("build one backend: --features faer or --features lapack");

#[cfg(feature = "faer")]
#[path = "faer_backend.rs"]
mod backend;
#[cfg(feature = "lapack")]
#[path = "lapack_backend.rs"]
mod backend;
mod npy;

use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, Instant};

use backend::Solver;

/// A routine is timed again until it has run at least [`MIN_RUNS`] times
/// and this long, or [`MAX_RUNS`] times; one untimed warm-up run first.
const TIME_BUDGET: Duration = Duration::from_secs(3);
const MIN_RUNS: usize = 3;
const MAX_RUNS: usize = 15;

struct Args {
    threads: usize,
    out: PathBuf,
    matrices: Vec<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let usage = "usage: eigh-bench --threads T --out DIR MATRIX.npy...";
    let mut threads = None;
    let mut out = None;
    let mut matrices = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--threads" => {
                let t = it.next().ok_or(usage)?;
                threads = Some(t.parse::<usize>().map_err(|_| usage)?);
            }
            "--out" => out = Some(PathBuf::from(it.next().ok_or(usage)?)),
            _ => matrices.push(PathBuf::from(arg)),
        }
    }
    match (threads, out) {
        (Some(threads), Some(out)) if threads > 0 && !matrices.is_empty() => Ok(Args {
            threads,
            out,
            matrices,
        }),
        _ => Err(usage.into()),
    }
}

/// The full column-major n × n of a packed lower triangle.
fn unpack(packed: &[f64]) -> Result<(usize, Vec<f64>), String> {
    // n (n + 1) / 2 = len.
    let n = ((((8 * packed.len() + 1) as f64).sqrt() as usize) - 1) / 2;
    if n * (n + 1) / 2 != packed.len() {
        return Err(format!("{} values is not a triangle", packed.len()));
    }
    let mut a = vec![0.0; n * n];
    for p in 0..n {
        for q in 0..=p {
            let x = packed[p * (p + 1) / 2 + q];
            a[p + q * n] = x;
            a[q + p * n] = x;
        }
    }
    Ok((n, a))
}

struct Timing {
    runs: usize,
    median: Duration,
    min: Duration,
}

fn time(solver: &mut Solver, a: &[f64]) -> Result<Timing, String> {
    solver.load(a);
    solver.run()?;
    let mut times = Vec::new();
    let mut total = Duration::ZERO;
    while times.len() < MAX_RUNS && (times.len() < MIN_RUNS || total < TIME_BUDGET) {
        solver.load(a);
        let start = Instant::now();
        solver.run()?;
        let elapsed = start.elapsed();
        times.push(elapsed);
        total += elapsed;
    }
    times.sort();
    Ok(Timing {
        runs: times.len(),
        median: times[times.len() / 2],
        min: times[0],
    })
}

/// `max_j ‖A v_j − λ_j v_j‖₂ / max |λ|` and `max |VᵀV − I|`.
fn accuracy(n: usize, a: &[f64], w: &[f64], v: &[f64]) -> (f64, f64) {
    let mut r = vec![0.0; n * n];
    backend::gemm(false, n, n, n, a, v, &mut r);
    let scale = w.iter().fold(0.0_f64, |m, x| m.max(x.abs()));
    let mut residual = 0.0_f64;
    for j in 0..n {
        let col = (0..n)
            .map(|i| (r[i + j * n] - w[j] * v[i + j * n]).powi(2))
            .sum::<f64>()
            .sqrt();
        residual = residual.max(col);
    }
    backend::gemm(true, n, n, n, v, v, &mut r);
    let mut orthogonality = 0.0_f64;
    for j in 0..n {
        for i in 0..n {
            let e = r[i + j * n] - if i == j { 1.0 } else { 0.0 };
            orthogonality = orthogonality.max(e.abs());
        }
    }
    (residual / scale, orthogonality)
}

fn run() -> Result<(), String> {
    let args = parse_args()?;
    rayon::ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .map_err(|e| e.to_string())?;
    let eigvals = args.out.join("eigvals");
    fs::create_dir_all(&eigvals).map_err(|e| e.to_string())?;
    let log_path = args.out.join(format!("{}.jsonl", backend::NAME));
    let mut log = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("{}: {e}", log_path.display()))?;
    eprintln!(
        "{} at {} threads: {}",
        backend::NAME,
        args.threads,
        backend::describe(args.threads)
    );

    for path in &args.matrices {
        let name = matrix_name(path);
        let (n, a) = unpack(&npy::read_1d(path)?)?;
        for vectors in [false, true] {
            for &routine in backend::ROUTINES {
                let mut solver = Solver::new(routine, n, vectors, args.threads);
                let t = time(&mut solver, &a)?;
                let w = solver.values();
                let (residual, orthogonality) = match solver.vectors() {
                    Some(v) => {
                        let (r, o) = accuracy(n, &a, w, v);
                        (format!("{r:e}"), format!("{o:e}"))
                    }
                    None => ("null".into(), "null".into()),
                };
                let mode = if vectors { "vecs" } else { "vals" };
                let file = format!(
                    "{}-{routine}-{name}-{mode}-t{}.npy",
                    backend::NAME,
                    args.threads
                );
                npy::write_1d(&eigvals.join(&file), w)?;
                let line = format!(
                    "{{\"backend\": \"{}\", \"routine\": \"{routine}\", \"matrix\": \"{name}\", \
                     \"n\": {n}, \"threads\": {}, \"vectors\": {vectors}, \"runs\": {}, \
                     \"median_s\": {:e}, \"min_s\": {:e}, \"lambda_min\": {:e}, \
                     \"lambda_max\": {:e}, \"residual\": {residual}, \
                     \"orthogonality\": {orthogonality}, \"eigvals\": \"{file}\"}}",
                    backend::NAME,
                    args.threads,
                    t.runs,
                    t.median.as_secs_f64(),
                    t.min.as_secs_f64(),
                    w[0],
                    w[n - 1],
                );
                writeln!(log, "{line}").map_err(|e| e.to_string())?;
                eprintln!(
                    "{name:>14} n={n:<5} {routine:<16} {mode} t={} median {:>10.3?} (of {}) \
                     residual {residual} orthogonality {orthogonality}",
                    args.threads, t.median, t.runs
                );
            }
        }
    }
    Ok(())
}

/// `S_gly5` from `.../S_gly5.npy`.
fn matrix_name(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("eigh-bench: {message}");
            ExitCode::FAILURE
        }
    }
}
