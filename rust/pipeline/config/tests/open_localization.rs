//! The localization of an open shell through the binary (tier 4, device
//! 0, one run per case): `calculation = "localization"` at a multiplicity
//! above 1, ROHF's doubly and singly occupied spaces each localized among
//! itself by Pipek-Mezey over Mulliken populations, against PySCF's
//! localization of the same spaces of its DF-ROHF over the same documents
//! and geometry: `steps/tests/data/localization_rohf.json`, the C++ tree's
//! `test/gtest/data/localization/rohf.json` copied byte for byte, its
//! provenance inside (PySCF 2.14.0, `tools/emit/localization_reference.py`:
//! the SCF at `conv_tol` 1e-12 and `conv_tol_grad` 1e-9, internally
//! stable; each space's Pipek-Mezey at `conv_tol` 1e-12, alternated with
//! PySCF's Jacobi stability check until stable). H₂O⁺, a doublet, whose
//! one singly occupied orbital has nothing to rotate with, and CH₂, a
//! triplet, whose two are localized as a pair, at exponents 2 and 4.
//!
//! **The bars, by analysis, one run each**, from
//! `steps/tests/data/localization_rohf_bars.json`, written by
//! `open_localization_reference.py` beside it (its docstring has the
//! analysis and the command; it first reproduces each space's L to 1e-9):
//!
//! - L, summed over the spaces: `|L - L_ref| <= sens_l τ_c + 2 (τ_L +
//!   τ_ref)`. The tree's SCF stops at a residual under `τ_c`, the default
//!   `commutator_tolerance` 1e-7, and the maximum moves with the spaces it
//!   publishes at first order, by at most `sens_l` per unit of residual
//!   (0.09 to 0.44 here); each space's localization stops within its
//!   last sweep's change of its maximum, `τ_L` the default
//!   `functional_tolerance` 1e-10 and `τ_ref` PySCF's `conv_tol` 1e-12.
//! - every population, each orbital's Mulliken population on each atom,
//!   our orbitals matched one to one to PySCF's within each space, since PM
//!   fixes an orbital only up to its sign and the order of a space's
//!   orbitals: `|ΔQ| <= sens_q τ_c + conv_q`, `sens_q` the populations'
//!   first-order reach per unit of residual with the localization following
//!   the spaces, and `conv_q` how far a localization stopped `2 (τ_L +
//!   τ_ref)` short of its maximum can sit from it, through the maximum's
//!   smallest curvature (about 1e-5, the square root of the tolerance, as
//!   L is quadratic there). A population is invariant under any
//!   transformation within an atom's AOs, so PySCF's normalization and
//!   order need no conversion.
//!
//! Beside them, on the same run: the spaces unmixed, `2 C_d C_dᵀ +
//! C_s C_sᵀ` the SCF's own D to `1e-12 max|D|` (round-off of products of
//! at most 24 terms); each space ascending in `<i|F|i> = Σ_k (C_occᵀ S
//! C_local)_ki² ε_k`, as the C++ step sorts it; L before under L after.

mod common;

use std::path::{Path, PathBuf};
use std::process::Command;

use fb_atom::{Atom, Element};
use fb_core_constants::BOHR_PER_ANGSTROM;
use fb_core_types::Double3;
use fb_molecule::Molecule;

/// The SCF's residual tolerance, `ScfStep`'s default.
const COMMUTATOR_TOLERANCE: f64 = 1e-7;

/// Each space's localization stopping short of its maximum: the tree's
/// default `functional_tolerance` and PySCF's `conv_tol`, for each of the
/// two spaces.
const LOCALIZATION_TOLERANCES: f64 = 2.0 * (1e-10 + 1e-12);

/// The density's bar, relative to its largest element.
const DENSITY_BAR: f64 = 1e-12;

/// A JSON file of the steps crate's `tests/data`, where the reference
/// moved with the open-shell localization step.
fn data(file: &str) -> serde_json::Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../steps/tests/data")
        .join(file);
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

/// The entry `name` of `list`.
fn named<'a>(list: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    list.as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == name)
        .unwrap_or_else(|| panic!("{name} is in the reference"))
}

/// The record of `exponent` in `list`.
fn at_exponent(list: &serde_json::Value, exponent: i64) -> &serde_json::Value {
    list.as_array()
        .unwrap()
        .iter()
        .find(|r| r["exponent"] == exponent)
        .unwrap_or_else(|| panic!("exponent {exponent} is in the reference"))
}

/// `free-breakfast <config>` run in `cwd`, exiting 0.
fn free_breakfast(cwd: &Path, config: &str) {
    let path = cwd.join("config.toml");
    std::fs::write(&path, config).unwrap();
    let out = Command::new(env!("CARGO_BIN_EXE_free-breakfast"))
        .arg(&path)
        .current_dir(cwd)
        .output()
        .expect("free-breakfast starts");
    println!("{}", String::from_utf8_lossy(&out.stdout));
    eprintln!("{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(out.status.code(), Some(0), "the run exits 0");
}

/// A float64 `.npy` of `dir`: its shape and data, column-major.
fn npy(dir: &Path, file: &str) -> (Vec<usize>, Vec<f64>) {
    let a = fb_io_npy::read_npy(&dir.join(file)).unwrap();
    (a.shape.to_vec(), a.data)
}

/// The case's molecule, its atoms at the reference's bohr coordinates, and
/// its XYZ text in Angstrom, which the binary reads back to the same bohr
/// to within a rounding.
fn molecule(case: &serde_json::Value) -> (Molecule, String) {
    let atoms: Vec<(String, [f64; 3])> = case["atoms"]
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
            (a["symbol"].as_str().unwrap().to_owned(), [r[0], r[1], r[2]])
        })
        .collect();
    let mut xyz = format!("{}\n{}\n", atoms.len(), case["name"]);
    for (symbol, r) in &atoms {
        let [x, y, z] = r.map(|v| v / BOHR_PER_ANGSTROM);
        xyz += &format!("{symbol} {x:.17e} {y:.17e} {z:.17e}\n");
    }
    let int = |key: &str| case[key].as_i64().unwrap() as i32;
    let molecule = Molecule::new(
        atoms
            .iter()
            .map(|(symbol, r)| Atom {
                position: Double3 {
                    x: r[0],
                    y: r[1],
                    z: r[2],
                },
                element: Element::from_symbol(symbol),
            })
            .collect(),
        int("charge"),
        int("multiplicity"),
    );
    (molecule, xyz)
}

/// The case `name` at `exponent` through the binary, held to the bars.
fn localizes_like_pyscf(name: &str, exponent: i64) {
    let reference = data("localization_rohf.json");
    let case = named(&reference["cases"], name);
    let record = at_exponent(&case["pipek_mezey"], exponent);
    let bars = at_exponent(
        &named(&data("localization_rohf_bars.json")["cases"], name)["exponents"],
        exponent,
    )
    .clone();
    let bar = |key: &str| bars[key].as_f64().unwrap();

    let cwd: PathBuf = Path::new(env!("CARGO_TARGET_TMPDIR"))
        .join(format!("open_localization_{name}_p{exponent}"));
    let _ = std::fs::remove_dir_all(&cwd);
    std::fs::create_dir_all(&cwd).unwrap();
    let (molecule, xyz) = molecule(case);
    std::fs::write(cwd.join("m.xyz"), xyz).unwrap();
    let head = format!(
        "[molecule]\nxyz = \"m.xyz\"\ncharge = {}\nmultiplicity = {}\n[basis_set]\nbasis = \"{}\"\n",
        case["charge"],
        case["multiplicity"],
        case["basis"].as_str().unwrap()
    );
    free_breakfast(
        &cwd,
        &format!(
            "calculation = \"localization\"\noutput = \"run\"\n{head}[eri]\n\
             mode = \"density_fitting\"\naux_basis = \"{}\"\nschwarz_threshold = 0.0\n[localization]\nexponent = {exponent}\n",
            case["aux_basis"].as_str().unwrap()
        ),
    );
    free_breakfast(
        &cwd,
        &format!("calculation = \"overlap\"\noutput = \"overlap\"\n{head}"),
    );
    let run = cwd.join("run");
    let scf: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(run.join("scf.json")).unwrap()).unwrap();
    let (n_alpha, n_beta) = (
        scf["n_alpha"].as_u64().unwrap() as usize,
        scf["n_beta"].as_u64().unwrap() as usize,
    );
    assert_eq!(
        (n_alpha as i64, n_beta as i64),
        (
            case["n_alpha"].as_i64().unwrap(),
            case["n_beta"].as_i64().unwrap()
        ),
        "{name}: the occupations"
    );
    let eps: Vec<f64> = scf["epsilon"]
        .as_array()
        .unwrap()
        .iter()
        .map(|e| e.as_f64().unwrap())
        .collect();
    let (shape, c_local) = npy(&run, "c_local.npy");
    let n = eps.len();
    assert_eq!(shape, [n, n_alpha], "{name}: C local is n × n_alpha");
    let (_, c) = npy(&run, "c.npy");
    let (_, d) = npy(&run, "d.npy");
    let (_, s) = npy(&cwd.join("overlap"), "overlap.npy");

    // L, summed over the spaces.
    let info: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(run.join("localization.json")).unwrap())
            .unwrap();
    let (before, after) = (
        info["functional_before"].as_f64().unwrap(),
        info["functional_after"].as_f64().unwrap(),
    );
    let l_ref = bar("functional");
    let l_bar = bar("sens_l") * COMMUTATOR_TOLERANCE + LOCALIZATION_TOLERANCES;
    let l_miss = (after - l_ref).abs();
    println!(
        "{name} p{exponent}: L {before:.12} -> {after:.12} in {} sweeps, against {l_ref:.12}: \
         |dL| = {l_miss:e}, bar {l_bar:e}",
        info["sweeps"]
    );
    assert!(
        l_miss <= l_bar,
        "{name} p{exponent}: |dL| = {l_miss:e} over {l_bar:e}"
    );
    assert!(before < after, "{name} p{exponent}: {info}");

    // Each orbital's Mulliken population on each atom, Q^A_ii =
    // Σ_{μ ∈ A} C_μi (S C)_μi, over the placed basis's atom AO ranges.
    let basis = fb_io_basis::resolve_basis(case["basis"].as_str().unwrap())
        .unwrap()
        .primitive_normalized()
        .place(&molecule)
        .unwrap();
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
    let sc = |i: usize, mu: usize| -> f64 {
        (0..n).map(|nu| s[nu * n + mu] * c_local[i * n + nu]).sum()
    };
    let populations: Vec<Vec<f64>> = (0..n_alpha)
        .map(|i| {
            offsets
                .windows(2)
                .map(|r| (r[0]..r[1]).map(|mu| c_local[i * n + mu] * sc(i, mu)).sum())
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
            "{name}: the {space} space's width"
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
    println!("{name} p{exponent}: max |dQ| = {worst:e}, bar {q_bar:e}");

    // The spaces unmixed: 2 C_d C_dᵀ + C_s C_sᵀ is the SCF's D.
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

    // Each space ascends in <i|F|i> = Σ_k (C_occᵀ S C_local)_ki² ε_k.
    let energies = common::fock_expectations(&c, &s, &c_local, &eps);
    for range in [0..n_beta, n_beta..n_alpha] {
        assert!(
            common::ascending(&energies[range]),
            "{name} p{exponent}: <i|F|i> {energies:?} does not ascend in each space"
        );
    }
}

/// H₂O⁺ at exponent 2. Needs a GPU.
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
