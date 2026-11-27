//! The localization step on an open shell (tier 4, one run per case):
//! ROHF's doubly and singly occupied blocks each localized among itself
//! by Pipek-Mezey over Mulliken populations, through [`fb_steps::run`]
//! from the SAD guess, against PySCF's localization of the same spaces of
//! its DF-ROHF over the same documents and geometry:
//! `tests/data/localization_rohf.json`, the C++ tree's
//! `test/gtest/data/localization/rohf.json` copied byte for byte, its
//! provenance inside (PySCF 2.14.0, `tools/emit/localization_reference.py`:
//! the SCF at `conv_tol` 1e-12 and `conv_tol_grad` 1e-9, internally
//! stable; each space's Pipek-Mezey at `conv_tol` 1e-12, alternated with
//! PySCF's Jacobi stability check until stable). H₂O⁺, a doublet, whose
//! one singly occupied orbital has nothing to rotate with, and CH₂, a
//! triplet, whose two are localized as a pair, at exponents 2 and 4. The
//! geometries are in bohr and written out as an XYZ file in Angstrom, at
//! CODATA 2018's bohr, the reader's.
//!
//! **The bars, by analysis, one run each**, from
//! `tests/data/localization_rohf_bars.json`, written by
//! `open_localization_reference.py` beside it (its docstring has the
//! analysis and the command; it first reproduces each space's L to 1e-9):
//!
//! - L, summed over the blocks: `|L - L_ref| <= sens_l τ_c + 2 (τ_L +
//!   τ_ref)`. The SCF stops at a residual under `τ_c`, the default
//!   `commutator_tolerance` 1e-7, and the maximum moves with the spaces it
//!   publishes at first order, by at most `sens_l` per unit of residual
//!   (0.09 to 0.44 here); each block's localization stops within its last
//!   sweep's change of its maximum, `τ_L` the default
//!   `functional_tolerance` 1e-10 and `τ_ref` PySCF's `conv_tol` 1e-12.
//! - every population, each orbital's Mulliken population on each atom,
//!   ours matched one to one to PySCF's within each block, since PM fixes
//!   an orbital only up to its sign and the order of a block's orbitals:
//!   `|ΔQ| <= sens_q τ_c + conv_q`, `sens_q` the populations' first-order
//!   reach per unit of residual with the localization following the
//!   spaces, and `conv_q` how far a localization stopped `2 (τ_L + τ_ref)`
//!   short of its maximum can sit from it, through the maximum's smallest
//!   curvature (about 1e-5, the square root of the tolerance, as L is
//!   quadratic there). A population is invariant under any transformation
//!   within an atom's AOs, so PySCF's normalization and order need no
//!   conversion.
//!
//! Beside them, on the same run: `c_local.npy` the host copy's bits and
//! `localization.json` the `Output`'s sums; the blocks unmixed,
//! `2 C_d C_dᵀ + C_s C_sᵀ` the SCF's own D to `1e-12 max|D|` (round-off
//! of products of at most 24 terms); each block ascending in its key,
//! `Σ_k (C_bᵀ S C_local)_ki² ε_k` over the block's own columns k of the
//! written C, each at least the previous less 1e-10 Eh (the closed
//! shell's margin, `localization_device.rs`); L before at most L after,
//! and a sweep run. Every run is NaN-filled under the allocation record.

mod common;

use std::path::Path;

use fb_core_budget::Backing;
use fb_core_flow::Bus;
use fb_steps::{Config, LocalizationStep, LocalizationTable, ScfStep, ScfTable, run};
use serde_json::Value;

use common::ANGSTROM_PER_BOHR;

/// The SCF's residual tolerance, `ScfTable`'s default.
const COMMUTATOR_TOLERANCE: f64 = 1e-7;

/// Each block's localization stopping short of its maximum: the step's
/// default `functional_tolerance` and PySCF's `conv_tol`, for each of the
/// two blocks.
const LOCALIZATION_TOLERANCES: f64 = 2.0 * (1e-10 + 1e-12);

/// The density's bar, relative to its largest element.
const DENSITY_BAR: f64 = 1e-12;

/// The order's margin, hartree.
const ORDER_MARGIN: f64 = 1e-10;

/// A JSON file of `tests/data`.
fn data(file: &str) -> Value {
    serde_json::from_str(&std::fs::read_to_string(common::data(file)).unwrap()).unwrap()
}

/// The entry `name` of `list`.
fn named<'a>(list: &'a Value, name: &str) -> &'a Value {
    list.as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("{name} is in the reference"))
}

/// The record of `exponent` in `list`.
fn at_exponent(list: &Value, exponent: i32) -> &Value {
    list.as_array()
        .unwrap()
        .iter()
        .find(|r| r["exponent"] == exponent)
        .unwrap_or_else(|| panic!("exponent {exponent} is in the reference"))
}

/// The chain's config of `case` at `exponent`: its geometry written to an
/// XYZ file in the target's scratch directory, its bases, `[scf]` and
/// `[localization]` at their defaults but the exponent, NaN-filled.
fn config(case: &Value, exponent: i32) -> Config {
    let name = case["name"].as_str().unwrap();
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR"));
    let atoms = case["atoms"].as_array().unwrap();
    let mut xyz = format!("{}\n{name}, Angstrom\n", atoms.len());
    for a in atoms {
        let r: Vec<f64> = a["coord"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_f64().unwrap() * ANGSTROM_PER_BOHR)
            .collect();
        xyz += &format!(
            "{} {:.17e} {:.17e} {:.17e}\n",
            a["symbol"].as_str().unwrap(),
            r[0],
            r[1],
            r[2]
        );
    }
    let file = format!("open_localization_{name}.xyz");
    std::fs::write(dir.join(&file), xyz).unwrap();
    let text = format!(
        "[molecule]\nxyz = \"{file}\"\ncharge = {}\nmultiplicity = {}\n\
         [basis]\nname = \"{}\"\n[eri]\naux_basis = \"{}\"\n",
        case["charge"],
        case["multiplicity"],
        case["basis"].as_str().unwrap(),
        case["aux_basis"].as_str().unwrap()
    );
    let mut config = Config::parse(&text, dir).unwrap();
    config.scf = Some(ScfTable::default());
    config.localization = Some(LocalizationTable {
        exponent,
        ..LocalizationTable::default()
    });
    config.resources.nan_fill = true;
    config
}

/// The case `name` at `exponent` on the card, held to the bars.
fn localizes_like_pyscf(name: &str, exponent: i32) {
    let reference = data("localization_rohf.json");
    let case = named(&reference["cases"], name);
    let record = at_exponent(&case["pipek_mezey"], exponent);
    let bars = data("localization_rohf_bars.json");
    let bars = at_exponent(&named(&bars["cases"], name)["exponents"], exponent);
    let bar = |key: &str| bars[key].as_f64().unwrap();

    let run = run(config(case, exponent), Backing::Real).expect("the chain runs");
    let dir = std::env::temp_dir().join(format!(
        "fb-steps-open-localization-{}-{name}-{exponent}",
        std::process::id()
    ));
    run.write(&dir).unwrap();
    let shared = run.plan.bus().shared();
    let dims = &shared.dims;
    let (n, n_alpha, n_beta) = (dims.n_ao, dims.n_alpha, dims.n_beta);
    assert_eq!(
        (n_alpha as i64, n_beta as i64),
        (
            case["n_alpha"].as_i64().unwrap(),
            case["n_beta"].as_i64().unwrap()
        ),
        "{name}: the occupations"
    );
    let scf = run.plan.output(ScfStep);
    assert!(scf.converged(), "{name}: the SCF");
    let local = run.plan.output(LocalizationStep);

    // The written products: the host copy's bits, the Output's sums.
    let array = fb_io_npy::read_npy(&dir.join("c_local.npy")).unwrap();
    assert_eq!(array.shape, [n, n_alpha], "{name}: C_local is n × n_alpha");
    assert!(
        array
            .data
            .iter()
            .zip(local.c())
            .all(|(a, b)| a.to_bits() == b.to_bits()),
        "{name}: c_local.npy is not the host copy"
    );
    let c_local = array.data;
    let info: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("localization.json")).unwrap())
            .unwrap();
    let (before, after) = (local.functional_before(), local.functional_after());
    assert_eq!(
        info["functional_before"].as_f64().unwrap().to_bits(),
        before.to_bits()
    );
    assert_eq!(
        info["functional_after"].as_f64().unwrap().to_bits(),
        after.to_bits()
    );
    assert_eq!(info["sweeps"], local.sweeps());
    std::fs::remove_dir_all(&dir).unwrap();

    // L, summed over the blocks.
    let l_ref = bar("functional");
    let l_bar = bar("sens_l") * COMMUTATOR_TOLERANCE + LOCALIZATION_TOLERANCES;
    let l_miss = (after - l_ref).abs();
    println!(
        "{name} p{exponent}: L {before:.12} -> {after:.12} in {} sweeps, against {l_ref:.12}: \
         |dL| = {l_miss:.2e}, bar {l_bar:.2e}; residual {:.2e}",
        local.sweeps(),
        scf.residual()
    );
    assert!(
        l_miss <= l_bar,
        "{name} p{exponent}: |dL| = {l_miss:e} over {l_bar:e}"
    );
    assert!(
        before <= after && local.sweeps() >= 1,
        "{name} p{exponent}: L {before} -> {after} in {} sweeps",
        local.sweeps()
    );

    // Each orbital's Mulliken population on each atom, Q^A_ii =
    // Σ_{μ ∈ A} C_μi (S C)_μi, over the basis's atom AO ranges.
    let basis = &shared.basis;
    let s = common::unpack(&fb_integrals_s_overlap_host::overlap(basis), n);
    let (n_atoms, n_shells) = (basis.natoms(), basis.nshells());
    let offsets: Vec<usize> = (0..=n_atoms)
        .map(|a| {
            let shell = if a < n_atoms {
                basis.atom_shell_offset(a)
            } else {
                n_shells
            };
            if shell < n_shells {
                basis.ao_offset_sphe(shell) as usize
            } else {
                n
            }
        })
        .collect();
    // S C_local, column-major n × n_alpha.
    let sc_local: Vec<f64> = (0..n_alpha * n)
        .map(|k| {
            let (mu, i) = (k % n, k / n);
            (0..n).map(|nu| s[nu * n + mu] * c_local[i * n + nu]).sum()
        })
        .collect();
    let populations: Vec<Vec<f64>> = (0..n_alpha)
        .map(|i| {
            offsets
                .windows(2)
                .map(|r| {
                    (r[0]..r[1])
                        .map(|mu| c_local[i * n + mu] * sc_local[i * n + mu])
                        .sum()
                })
                .collect()
        })
        .collect();
    let q_bar = bar("sens_q") * COMMUTATOR_TOLERANCE + bar("conv_q");
    let mut worst = 0.0_f64;
    for (space, range) in [("doubly", 0..n_beta), ("singly", n_beta..n_alpha)] {
        let theirs: Vec<Vec<f64>> = record[space]["populations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                row.as_array()
                    .unwrap()
                    .iter()
                    .map(|q| q.as_f64().unwrap())
                    .collect()
            })
            .collect();
        assert_eq!(
            theirs.len(),
            range.len(),
            "{name}: the {space} block's width"
        );
        let mut free: Vec<usize> = range.collect();
        for (k, want) in theirs.iter().enumerate() {
            let gap = |i: usize| {
                populations[i]
                    .iter()
                    .zip(want)
                    .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()))
            };
            let (at, best) = free
                .iter()
                .enumerate()
                .map(|(at, &i)| (at, gap(i)))
                .min_by(|a, b| a.1.total_cmp(&b.1))
                .expect("an orbital left to match");
            worst = worst.max(best);
            assert!(
                best <= q_bar,
                "{name} p{exponent}: PySCF's {space} orbital {k} is {best:e} from ours, over \
                 {q_bar:e}"
            );
            free.remove(at);
        }
    }
    println!("{name} p{exponent}: max |dQ| = {worst:.2e}, bar {q_bar:.2e}");

    // The blocks unmixed: 2 C_d C_dᵀ + C_s C_sᵀ is the SCF's D.
    let d = scf.d();
    let scale = d.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
    let mut d_miss = 0.0_f64;
    for nu in 0..n {
        for mu in 0..n {
            let want: f64 = (0..n_alpha)
                .map(|i| {
                    let weight = if i < n_beta { 2.0 } else { 1.0 };
                    weight * c_local[i * n + mu] * c_local[i * n + nu]
                })
                .sum();
            d_miss = d_miss.max((d[nu * n + mu] - want).abs());
        }
    }
    assert!(
        d_miss <= DENSITY_BAR * scale,
        "{name} p{exponent}: max |D - 2 C_d C_dᵀ - C_s C_sᵀ| = {d_miss:e}"
    );

    // Each block ascends in Σ_k (C_bᵀ S C_local)_ki² ε_k.
    let (c, epsilon) = (scf.c(), scf.epsilon());
    for range in [0..n_beta, n_beta..n_alpha] {
        let key: Vec<f64> = range
            .clone()
            .map(|i| {
                range
                    .clone()
                    .map(|k| {
                        let overlap: f64 =
                            (0..n).map(|mu| c[k * n + mu] * sc_local[i * n + mu]).sum();
                        overlap * overlap * epsilon[k]
                    })
                    .sum()
            })
            .collect();
        println!("{name} p{exponent}: <i|F|i> = {key:.6?}");
        for i in 1..key.len() {
            assert!(
                key[i] >= key[i - 1] - ORDER_MARGIN,
                "{name} p{exponent}: <i|F|i> {key:?} does not ascend in the block {range:?}"
            );
        }
    }
}

/// H₂O⁺ at exponent 2: the family's representative. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn water_cation_p2_localizes_like_pyscf() {
    localizes_like_pyscf("h2o_cation_ccpvdz", 2);
}

/// H₂O⁺ at exponent 4. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn water_cation_p4_localizes_like_pyscf() {
    localizes_like_pyscf("h2o_cation_ccpvdz", 4);
}

/// Triplet CH₂ at exponent 2: two singly occupied orbitals localized as a
/// pair. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn ch2_triplet_p2_localizes_like_pyscf() {
    localizes_like_pyscf("ch2_ccpvdz", 2);
}

/// Triplet CH₂ at exponent 4. Needs a GPU.
#[test]
#[ignore = "tier 4: needs a GPU"]
fn ch2_triplet_p4_localizes_like_pyscf() {
    localizes_like_pyscf("ch2_ccpvdz", 4);
}
