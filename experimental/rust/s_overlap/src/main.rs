//! overlap-demo: the spherical AO overlap S of a molecule on the GPU, from
//! Rust, optionally checked against a PySCF reference.
//!
//! Usage: overlap-demo XYZ BASIS [REFERENCE.npy]
//!
//! XYZ is in Angstrom; BASIS is a carried name ("sto-3g", "cc-pvdz", ...) or a
//! BSE JSON path. Prints S's size, its symmetry and how far its diagonal is
//! from 1 (every AO here is normalized). With a reference (data/reference.py
//! writes one), also the largest difference from it, and exits 1 past
//! 1e-12 * max|S_ref|.

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use fb_gemm::Gpu;
use fb_s_overlap::{npy, overlap};

fn run() -> Result<bool, Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let [_, xyz, basis, rest @ ..] = &args[..] else {
        return Err("usage: overlap-demo XYZ BASIS [REFERENCE.npy]".into());
    };

    let gpu = Gpu::new(0)?;
    let t = Instant::now();
    let s = overlap(&gpu, Path::new(xyz), basis)?;
    let host = s.to_vec()?;
    let ms = t.elapsed().as_secs_f64() * 1e3;
    let n = s.rows();

    let mut asym = 0.0f64;
    let mut diag = 0.0f64;
    for j in 0..n {
        diag = diag.max((host[j + j * n] - 1.0).abs());
        for i in 0..j {
            asym = asym.max((host[i + j * n] - host[j + i * n]).abs());
        }
    }
    println!("S of {xyz} in {basis}: {n} x {n} spherical AOs, {ms:.1} ms (steps + download)");
    println!("  max |S - S^T|    = {asym:.3e}");
    println!("  max |diag(S) - 1| = {diag:.3e}");

    let Some(reference) = rest.first() else {
        return Ok(true);
    };
    let (shape, want) = npy::read_f64_matrix(Path::new(reference))?;
    if shape != [n, n] {
        return Err(format!("{reference} is {shape:?}, S is {n} x {n}").into());
    }
    let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let worst = host
        .iter()
        .zip(&want)
        .fold(0.0f64, |m, (g, w)| m.max((g - w).abs()));
    let ok = worst <= 1e-12 * scale;
    println!(
        "  max |S - S_ref|  = {worst:.3e}  (bar 1e-12 * max|S_ref| = {:.1e}): {}",
        1e-12 * scale,
        if ok { "ok" } else { "FAIL" }
    );
    Ok(ok)
}

fn main() -> ExitCode {
    match run() {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(e) => {
            eprintln!("overlap-demo: {e}");
            ExitCode::FAILURE
        }
    }
}
