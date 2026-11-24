//! The DF-ROHF loop run on a card (tier 4, device 0, one run per case),
//! from the SAD guess, against PySCF's DF-ROHF over the same orbital and
//! fitting documents and geometry: `tests/data/df_rohf.json`, the C++
//! tree's own reference (`test/gtest/data/density_fitting/df_rohf.json`)
//! copied, its provenance inside (PySCF 2.14.0 from `uv.lock`, written by
//! `tools/emit/df_reference.py` at `conv_tol = 1e-12`,
//! `conv_tol_grad = 1e-9`, each solution internally stable). Its orbital energies are the eigenvalues of PySCF's default
//! Roothaan matrix, which the published ones are by construction
//! ([`fb_pipeline_scf::add_scf`]). The one-electron cases, which the C++
//! reference does not carry, are `tests/data/df_rohf_orbitals.json`'s own,
//! written by `rohf_reference.py` the same way, with the same keys, by
//! PySCF's ROHF class itself: its `scf.ROHF` hands one electron to `HF1e`,
//! whose energy is the same but whose orbital energies are the core
//! Hamiltonian's, not the Roothaan matrix's.
//!
//! **The bars, by analysis, one run.** The energy: `|E - E_ref| <= 1e-9`
//! hartree, the closed shell's (`tests/scf.rs`). The ROHF energy is
//! stationary in the orbital rotations between occupation classes, which
//! are what the residual measures, so the convergence error is second
//! order in `‖e‖ < 1e-7`, about 1e-14 of the curvature; what is left is
//! the fitted integrals' agreement with PySCF's, about 1e-10 hartree at
//! these sizes. Every orbital energy: `|ε - ε_ref| <= 1e-6` hartree, the
//! C++ bar. An eigenvalue moves at first order in F, whose error is first
//! order in the orbitals': a rotation of about `‖e‖ / 2Δ` between classes,
//! `Δ` the smallest gap across them (above 0.3 hartree in every case
//! here), through a response of F of a few hartree (the largest
//! one-centre integrals), so a few 1e-7 at `‖e‖ = 1e-7`, and the integrals'
//! 1e-10 beside it. Beside them, on the same run: `n_alpha` and `n_beta`
//! the reference's, the residual under the tolerance, and the published
//! `D` the spin densities of the published `C`,
//! `C[:, :n_alpha] C[:, :n_alpha]ᵀ + C[:, :n_beta] C[:, :n_beta]ᵀ`, to
//! `1e-12 max|D|`, the round-off of two gemms of at most 58 terms.
//!
//! **The orbitals**, against PySCF's `mo_coeff` over the same run:
//! `tests/data/df_rohf_orbitals.json`, written by `rohf_reference.py`
//! beside it (its provenance inside), `mo_coeff` and S in the tree's
//! unrenormalized functions. A column is defined up to its sign, and a
//! set of degenerate ones up to a rotation among them, so the reference's
//! columns are grouped into blocks, consecutive energies closer than
//! [`BLOCK_SPACING`] one block, and each block of published columns is
//! held to the subspace of the reference's: with `O = C_refᵀ S C`, the
//! block's leak `‖O[rows outside b, b]‖_F`, `‖sin Θ‖_F` between the two
//! subspaces since both C are S-orthonormal. Both sets of orbitals are
//! eigenvectors of PySCF's Roothaan matrix, the published ones at the
//! last build's densities, so by Davis and Kahan the leak is at most
//! `‖δF‖ √k / gap`, `k` the block's width, `gap` its distance to the
//! nearest energy outside it, and `δF` the matrices' difference, which the
//! orbital energies' analysis above bounds by [`EPSILON_BAR`]. The bar is
//! that bound, `EPSILON_BAR √k / gap`, per block; a block's columns must
//! lie in one occupation class. OH is left out: linear, its ²Π hole is
//! on any combination of the π pair, so its minimum is a circle of states
//! of one energy, and PySCF's C one point of it, not the published one's.

use std::path::Path;
use std::sync::Arc;

use fb_atom::{Atom, Element};
use fb_core_types::Double3;
use fb_core_wwr::runtime::Stream;
use fb_core_wwr::solver::SolverHandle;
use fb_molecule::Molecule;
use fb_pipeline_eri::EriStep;
use fb_pipeline_scf::{Run, ScfStep};
use fb_recipes_diagonalize::syevd_lwork;

/// The energy bar, hartree.
const ENERGY_BAR: f64 = 1e-9;

/// The orbital energy bar, hartree.
const EPSILON_BAR: f64 = 1e-6;

/// The density's bar, relative to its largest element.
const DENSITY_BAR: f64 = 1e-12;

/// Consecutive reference energies closer than this, hartree, are one block
/// of the orbital comparison: a degenerate set, whose columns are defined
/// up to a rotation among them. Every gap a bar is built from is then at
/// least this.
const BLOCK_SPACING: f64 = 1e-3;

/// The reference's case `name`: `df_rohf.json`'s, or else one of
/// `df_rohf_orbitals.json`'s own.
fn case(name: &str) -> serde_json::Value {
    [
        ("df_rohf.json", "df_rohf"),
        ("df_rohf_orbitals.json", "df_rohf_orbitals"),
    ]
    .into_iter()
    .find_map(|(file, key)| {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(file);
        let text = std::fs::read_to_string(&path).unwrap();
        let json: serde_json::Value = serde_json::from_str(&text).unwrap();
        json[key]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name && c.get("mo_energies").is_some())
            .cloned()
    })
    .unwrap_or_else(|| panic!("{name} is in the reference"))
}

/// The case's molecule: its atoms at their coordinates, in bohr, with its
/// charge and multiplicity.
fn molecule(case: &serde_json::Value) -> Molecule {
    let atoms = case["atoms"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            let r: Vec<f64> = a["coord"]
                .as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect();
            Atom {
                position: Double3 {
                    x: r[0],
                    y: r[1],
                    z: r[2],
                },
                element: Element::from_symbol(a["symbol"].as_str().unwrap()),
            }
        })
        .collect();
    let int = |key: &str| case[key].as_i64().unwrap() as i32;
    Molecule::new(atoms, int("charge"), int("multiplicity"))
}

/// The case `name` run at the default options and held to the bars, and
/// its products returned.
fn converges_to_pyscf(name: &str) -> fb_pipeline_scf::ScfOutput {
    let case = case(name);
    let molecule = molecule(&case);
    let step = ScfStep::default();
    let run = Run::new(
        &molecule,
        case["basis"].as_str().unwrap(),
        &EriStep::density_fitting(case["aux_basis"].as_str().unwrap(), 0.0),
        &step,
        1,
        |n| syevd_lwork(&SolverHandle::new(Arc::new(Stream::new(0))), n),
    )
    .expect("the case plans");
    let (record, out) = run.run(&[0]);
    record.result.unwrap();
    let out = out.expect("a converged run brings its products back");
    let count = |key: &str| case[key].as_u64().unwrap() as usize;
    let e_ref = case["energy"].as_f64().unwrap();
    let eps_ref: Vec<f64> = case["mo_energies"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e.as_f64().unwrap())
        .collect();
    let miss = (out.energy - e_ref).abs();
    let eps_miss = out
        .epsilon
        .iter()
        .zip(&eps_ref)
        .fold(0.0_f64, |m, (g, w)| m.max((g - w).abs()));
    println!(
        "{name}: E = {:.12} against {e_ref:.12}, |ΔE| = {miss:e}, max |Δε| = {eps_miss:e}, \
         {} iterations, ‖e‖ = {:e}, wall {:?}",
        out.energy, out.iterations, out.residual, record.wall
    );
    assert_eq!(
        (out.n_alpha, out.n_beta),
        (count("n_alpha"), count("n_beta")),
        "{name}: the occupations"
    );
    assert!(
        miss <= ENERGY_BAR,
        "{name}: |ΔE| = {miss:e} over {ENERGY_BAR:e}"
    );
    assert_eq!(out.epsilon.len(), eps_ref.len(), "{name}: one ε per AO");
    assert!(
        eps_miss <= EPSILON_BAR,
        "{name}: max |Δε| = {eps_miss:e} over {EPSILON_BAR:e}: {:?} against {eps_ref:?}",
        out.epsilon
    );
    assert!(
        out.residual < step.commutator_tolerance,
        "{name}: ‖e‖ = {:e}",
        out.residual
    );

    // D from the published C's occupied columns, on the host.
    let n = out.epsilon.len();
    let c = |mu: usize, i: usize| out.c[i * n + mu];
    let scale = out.d.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut worst = 0.0_f64;
    for nu in 0..n {
        for mu in 0..n {
            let want: f64 = (0..out.n_alpha).map(|i| c(mu, i) * c(nu, i)).sum::<f64>()
                + (0..out.n_beta).map(|i| c(mu, i) * c(nu, i)).sum::<f64>();
            worst = worst.max((out.d[nu * n + mu] - want).abs());
        }
    }
    println!("{name}: max |D - Da - Db| = {worst:e}, max |D| = {scale:e}");
    assert!(
        worst <= DENSITY_BAR * scale,
        "{name}: max |D - Da - Db| = {worst:e}"
    );
    out
}

/// The reference's orbitals of case `name`: `(mo_energy, mo_coeff, S)`,
/// the matrices column-major.
fn reference_orbitals(name: &str) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/df_rohf_orbitals.json");
    let text = std::fs::read_to_string(&path).unwrap();
    let json: serde_json::Value = serde_json::from_str(&text).unwrap();
    let case = json["df_rohf_orbitals"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("{name} is in the orbitals' reference"));
    let list = |key: &str| -> Vec<f64> {
        case[key]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    };
    (list("mo_energy"), list("mo_coeff"), list("overlap"))
}

/// The published C against PySCF's `mo_coeff`, block by block (the
/// module's doc has the bar).
fn orbitals_match_pyscf(name: &str, out: &fb_pipeline_scf::ScfOutput) {
    let (eps_ref, c_ref, s) = reference_orbitals(name);
    let n = eps_ref.len();
    assert_eq!(out.c.len(), n * n, "{name}: C is n × n");
    // O = C_refᵀ (S C), on the host.
    let mut sc = vec![0.0; n * n];
    for j in 0..n {
        for k in 0..n {
            let c_kj = out.c[j * n + k];
            for i in 0..n {
                sc[j * n + i] += s[k * n + i] * c_kj;
            }
        }
    }
    let o = |i: usize, j: usize| -> f64 { (0..n).map(|k| c_ref[i * n + k] * sc[j * n + k]).sum() };
    // The blocks, as [first, last) column ranges of the reference.
    let mut blocks = vec![0];
    for j in 1..n {
        if eps_ref[j] - eps_ref[j - 1] >= BLOCK_SPACING {
            blocks.push(j);
        }
    }
    blocks.push(n);
    let class = |j: usize| (j >= out.n_beta) as u8 + (j >= out.n_alpha) as u8;
    let (mut worst, mut largest) = (0.0_f64, 0.0_f64);
    for b in blocks.windows(2) {
        let (first, last) = (b[0], b[1]);
        assert_eq!(
            class(first),
            class(last - 1),
            "{name}: the block [{first}, {last}) crosses an occupation class"
        );
        let below = (first > 0).then(|| eps_ref[first] - eps_ref[first - 1]);
        let above = (last < n).then(|| eps_ref[last] - eps_ref[last - 1]);
        let gap = below.into_iter().chain(above).fold(f64::INFINITY, f64::min);
        let width = (last - first) as f64;
        let leak = (first..last)
            .flat_map(|j| (0..first).chain(last..n).map(move |i| (i, j)))
            .map(|(i, j)| o(i, j).powi(2))
            .sum::<f64>()
            .sqrt();
        let bar = EPSILON_BAR * width.sqrt() / gap;
        worst = worst.max(leak / bar);
        largest = largest.max(leak);
        assert!(
            leak <= bar,
            "{name}: the orbitals [{first}, {last}) leak {leak:e} out of PySCF's, over \
             {bar:e} (gap {gap:e})"
        );
    }
    println!(
        "{name}: {} orbital blocks, the largest leak {largest:.3e}, at most {worst:.3e} of \
         its bar",
        blocks.len() - 1
    );
}

/// OH, a doublet, in cc-pVDZ over cc-pVDZ-JKFIT: the open shell's nodes
/// alone, neutral, so no guess rescale, and one open orbital, so its
/// rotation is a sign. A miss here and not in the cation is the open
/// path's, not the rescale's. Its orbitals are not compared: its minimum
/// is a circle of states (the module's doc). Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn oh_converges_to_pyscf() {
    converges_to_pyscf("oh_ccpvdz");
}

/// CH₂, a triplet, in cc-pVDZ over cc-pVDZ-JKFIT, off every axis: two open
/// orbitals, so the open block is 2 × 2, its eigensolve a real rotation,
/// and `n_alpha - n_beta = 2` in every factor and density; and its
/// orbitals against PySCF's. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn ch2_triplet_converges_to_pyscf() {
    let out = converges_to_pyscf("ch2_ccpvdz");
    orbitals_match_pyscf("ch2_ccpvdz", &out);
}

/// H₂O⁺, a doublet, in cc-pVDZ over cc-pVDZ-JKFIT: charged, so the SAD
/// guess's ten electrons are rescaled to nine before the first build; and
/// its orbitals against PySCF's. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn water_cation_converges_to_pyscf() {
    let out = converges_to_pyscf("h2o_cation_ccpvdz");
    orbitals_match_pyscf("h2o_cation_ccpvdz", &out);
}

/// CH₂, the triplet, in cc-pVTZ over cc-pVTZ-JKFIT (58 AOs, f orbitals,
/// g fitting functions), the same bars: the open path at a basis whose
/// guess factor's rest is wide (`n - n_beta = 55` columns), and its
/// orbitals against PySCF's. The C++ case
/// is slow, its SCF integral-direct; this one's run is about 0.2 s on an
/// RTX 3080, so it is no slow case. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn ch2_triplet_cc_pvtz_converges_to_pyscf() {
    let out = converges_to_pyscf("ch2_ccpvtz");
    orbitals_match_pyscf("ch2_ccpvtz", &out);
}

/// The H atom, a doublet, in cc-pVDZ over cc-pVDZ-JKFIT: one electron, so
/// no beta electron, no beta factor and `Db = 0` after the first build
/// (`n_alpha = 1`, `n_beta = 0`); and its orbitals against PySCF's. Needs a
/// GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn hydrogen_atom_converges_to_pyscf() {
    let out = converges_to_pyscf("h_ccpvdz");
    orbitals_match_pyscf("h_ccpvdz", &out);
}

/// H₂⁺, a doublet, in cc-pVDZ over cc-pVDZ-JKFIT at 2.0 bohr: no beta
/// electron, and charged, so the SAD guess's two electrons are rescaled to
/// one; and its orbitals against PySCF's. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn hydrogen_molecule_cation_converges_to_pyscf() {
    let out = converges_to_pyscf("h2_cation_ccpvdz");
    orbitals_match_pyscf("h2_cation_ccpvdz", &out);
}
