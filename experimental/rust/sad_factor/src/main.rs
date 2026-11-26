//! `sad-factor XYZ BASIS [OUT.npy]`: the SAD guess of the molecule in `XYZ`
//! (Angstrom, neutral) in `BASIS` (a compiled-in name such as `cc-pvdz`, or
//! a BSE JSON path) as a factor L, `D_SAD = 2 L Lᵀ`. It prints each
//! element's rank and eigenvalue gap, L's width beside the molecule's
//! occupied count, the time, and how far `2 L Lᵀ` is from
//! `fb_sad::sad_density`; writes L, `n_ao × rank`, to `OUT.npy` if given;
//! and exits 1 past `1e-12`.

use std::path::Path;
use std::process::ExitCode;
use std::time::Instant;

use fb_sad_factor::{density_of, sad_factor};

/// The bar on `max |2 L Lᵀ - D_SAD|`. D's elements are of order 1 and each
/// of the product's is one rounded dot of at most an atom's rank.
const BAR: f64 = 1e-12;

fn run(args: &[String]) -> Result<bool, String> {
    let [xyz, basis_name, rest @ ..] = args else {
        return Err("usage: sad-factor XYZ BASIS [OUT.npy]".into());
    };
    let molecule = fb_io_xyz::read_xyz(Path::new(xyz), 0, 1).map_err(|e| e.to_string())?;
    let basis = fb_io_basis::resolve_basis(basis_name)
        .map_err(|e| e.to_string())?
        .primitive_normalized()
        .place(&molecule)
        .map_err(|e| e.to_string())?;
    let baked: Option<fb_sad::SadStampLookup> =
        (!basis.name().is_empty()).then_some(fb_io_sad::builtin_sad_density);

    let start = Instant::now();
    let factor = sad_factor(&basis, molecule.atoms(), baked).map_err(|e| e.to_string())?;
    let took = start.elapsed();

    for element in &factor.elements {
        let f = &element.factor;
        println!(
            "Z={:<3} x{:<4} {:>3} AOs, rank {:>2}; smallest kept {:.3e}, largest dropped {:.1e}",
            element.z, element.atoms, f.rows, f.rank, f.smallest_kept, f.largest_dropped
        );
    }
    let n_occ = (molecule.electron_count() / 2) as usize;
    println!(
        "{} AOs, {n_occ} occupied; L is {} x {}, {:.2}x the occupied count, {:.1}% of n_ao; \
         {took:.2?}",
        factor.n_ao,
        factor.n_ao,
        factor.rank,
        factor.rank as f64 / n_occ as f64,
        100.0 * factor.rank as f64 / factor.n_ao as f64
    );

    let want = fb_sad::sad_density(&basis, molecule.atoms(), baked).map_err(|e| e.to_string())?;
    let got = density_of(&factor.columns, factor.n_ao, factor.rank);
    let worst = got
        .iter()
        .zip(&want)
        .fold(0.0_f64, |m, (g, w)| m.max((g - w).abs()));
    println!("max |2 L Lt - D_SAD| = {worst:.2e} (bar {BAR:.0e})");

    if let [out, ..] = rest {
        fb_io_npy::write_npy(Path::new(out), &[factor.n_ao, factor.rank], &factor.columns)
            .map_err(|e| e.to_string())?;
        println!("wrote {out}");
    }
    Ok(worst <= BAR)
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match run(&args) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::FAILURE,
        Err(message) => {
            eprintln!("sad-factor: {message}");
            ExitCode::FAILURE
        }
    }
}
