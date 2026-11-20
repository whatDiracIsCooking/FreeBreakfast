//! `df-metric-host XYZ AUX [OUT.npy [REFERENCE.npy]]`: the lower triangle
//! of the Coulomb metric (P|Q) of the molecule in `XYZ` (Angstrom) over the
//! fitting set `AUX` (a compiled-in name such as `cc-pvdz-jkfit`, or a BSE
//! JSON path), computed on the host. It prints the size, the best time of
//! each route (directly spherical, and via the Cartesian integrals), their
//! largest difference and a few numbers of M; writes the packed triangle to
//! `OUT.npy` if given, one dimension of `n (n + 1) / 2`; and compares it
//! with the lower triangle of the full `n × n` `REFERENCE.npy` if given,
//! exiting 1 past `1e-12 max |M_ref|`.

use std::path::Path;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use fb_df_metric_host::{far_pairs, lower_index, metric, metric_via_cartesian};

/// The comparison's bar, relative to `max |M_ref|`: the device metric's.
const BAR: f64 = 1e-12;

/// A route is timed again until it has run this long or [`MAX_REPEATS`]
/// times; the best is printed.
const TIMING_BUDGET: Duration = Duration::from_secs(1);

/// The most times one route is timed.
const MAX_REPEATS: usize = 20;

/// `build`'s matrix, its best time, and how many runs that is the best of.
fn best_of(build: impl Fn() -> Vec<f64>) -> (Vec<f64>, Duration, usize) {
    let mut best = Duration::MAX;
    let mut total = Duration::ZERO;
    let mut m = Vec::new();
    let mut runs = 0;
    while runs < MAX_REPEATS && (runs == 0 || total < TIMING_BUDGET) {
        let start = Instant::now();
        m = build();
        let elapsed = start.elapsed();
        best = best.min(elapsed);
        total += elapsed;
        runs += 1;
    }
    (m, best, runs)
}

fn run(args: &[String]) -> Result<bool, String> {
    let [xyz, aux_name, rest @ ..] = args else {
        return Err("usage: df-metric-host XYZ AUX [OUT.npy [REFERENCE.npy]]".into());
    };
    // Charge and multiplicity do not enter M.
    let molecule = fb_io_xyz::read_xyz(Path::new(xyz), 0, 1).map_err(|e| e.to_string())?;
    let aux = fb_io_basis::resolve_aux_basis(aux_name)
        .map_err(|e| e.to_string())?
        .primitive_normalized()
        .place(&molecule)
        .map_err(|e| e.to_string())?;
    let n = aux.nao_sphe() as usize;
    let l_max = aux
        .shells()
        .iter()
        .map(fb_basis::Shell::angular_momentum)
        .max()
        .unwrap_or(0);

    println!(
        "{} atoms, {} shells, {n} fitting functions, l <= {l_max}; \
         the lower triangle is {} values, {:.3} GiB",
        molecule.natoms(),
        aux.nshells(),
        lower_index(n, 0),
        (lower_index(n, 0) * 8) as f64 / (1u64 << 30) as f64
    );
    let (far, pairs) = far_pairs(&aux);
    println!(
        "{} rayon threads; {far} of {pairs} shell pairs are far field, {:.1}%",
        rayon::current_num_threads(),
        100.0 * far as f64 / pairs as f64
    );
    let (m, direct, direct_runs) = best_of(|| metric(&aux));
    println!("directly spherical: {direct:.2?} (best of {direct_runs})");
    let (via, cartesian, cartesian_runs) = best_of(|| metric_via_cartesian(&aux));
    println!(
        "via Cartesian:      {cartesian:.2?} (best of {cartesian_runs}), {:.2}x",
        cartesian.as_secs_f64() / direct.as_secs_f64()
    );

    let largest = m.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let apart = m
        .iter()
        .zip(&via)
        .fold(0.0_f64, |acc, (g, w)| acc.max((g - w).abs()));
    drop(via);
    let trace: f64 = (0..n).map(|p| m[lower_index(p, p)]).sum();
    println!(
        "max |M| = {largest:.12}, trace = {trace:.12}, \
         the routes differ by {:e} max |M|",
        apart / largest
    );

    if let Some(out) = rest.first() {
        fb_io_npy::write_npy(Path::new(out), &[m.len()], &m).map_err(|e| e.to_string())?;
        println!("wrote {out}");
    }
    let Some(reference) = rest.get(1) else {
        return Ok(true);
    };
    let want = fb_io_npy::read_npy(Path::new(reference)).map_err(|e| e.to_string())?;
    if want.shape != [n, n] {
        return Err(format!("{reference} is {:?}, M is {n} x {n}", want.shape));
    }
    let scale = want.data.iter().fold(0.0_f64, |acc, v| acc.max(v.abs()));
    let mut worst = 0.0_f64;
    for p in 0..n {
        for q in 0..=p {
            worst = worst.max((m[lower_index(p, q)] - want.data[p + q * n]).abs());
        }
    }
    println!(
        "max |M - M_ref| / max |M_ref| = {:e}, bar {BAR:e}",
        worst / scale
    );
    Ok(worst <= BAR * scale)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(message) => {
            eprintln!("df-metric-host: {message}");
            ExitCode::FAILURE
        }
    }
}
