//! `test/sad/sad_host.cpp`, tier 3, one test per case of its serial form,
//! each named after it (the task form's cases are `fb-pipeline-guess`'s;
//! their serial halves are here), plus the refusals the C++ leaves
//! untested and a lookup standing in for an SCF that would fail.
//!
//! The PySCF reference is read from where the C++ test reads it,
//! `test/sad/data/atomic_scf.json` (its README there has the format and the
//! command that writes it), not copied: one file, so the two trees cannot
//! hold different references.
//!
//! The bar against PySCF is the C++ test's, fixed by analysis: 1e-7 per
//! density element, the density's error at an SCF converged to 1e-9 in the
//! commutator. One run each.

use std::path::PathBuf;
use std::sync::LazyLock;

use fb_atom::{Atom, Element};
use fb_basis::{BasisSet, Primitive, Shell};
use fb_core_error_handling::{ErrorKind, Result};
use fb_core_types::Double3;
use fb_io_basis::parse_basis;
use fb_sad::{SadStampLookup, sad_density};
use serde_json::Value;

const ORIGIN: Double3 = Double3 {
    x: 0.0,
    y: 0.0,
    z: 0.0,
};

const fn at(x: f64, y: f64, z: f64) -> Double3 {
    Double3 { x, y, z }
}

/// The reference file, read once per binary: every case that wants a basis
/// shares it.
static REFERENCE: LazyLock<Value> = LazyLock::new(|| {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../test/sad/data/atomic_scf.json");
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    serde_json::from_str(&text).unwrap()
});

/// One element's shells from the reference file's BSE basis, at the origin,
/// the way a production run gets them: `parse_basis`,
/// `primitive_normalized`, `place`.
fn reference_shells(basis: &str, z: i32) -> Vec<Shell> {
    let stamp = parse_basis(&REFERENCE["bases"][basis].to_string())
        .unwrap()
        .primitive_normalized();
    let element = stamp
        .shells_for(z)
        .unwrap_or_else(|| panic!("{basis} Z={z}"));
    element.iter().map(|s| s.place(ORIGIN)).collect()
}

/// The guess for one atom alone, its shells grouped as `place` would.
fn atom_density(element: Element, shells: Vec<Shell>) -> Result<Vec<f64>> {
    let count = shells.len() as u32;
    let basis = BasisSet::with_atoms(shells, vec![0, count], "");
    sad_density(
        &basis,
        &[Atom {
            position: ORIGIN,
            element,
        }],
        None,
    )
}

/// A basis over `atoms` laid out as `place` would: each atom's shells, from
/// `shells_of` its element, moved to its position and grouped per atom, and
/// named `name` as `place` names a compiled-in set.
fn grouped_basis(
    atoms: &[Atom],
    shells_of: impl Fn(Element) -> Vec<Shell>,
    name: &str,
) -> BasisSet {
    let mut shells = Vec::new();
    let mut offsets = vec![0u32];
    for atom in atoms {
        for s in shells_of(atom.element) {
            shells.push(Shell::new(
                s.angular_momentum(),
                s.primitives().to_vec(),
                atom.position,
            ));
        }
        offsets.push(shells.len() as u32);
    }
    BasisSet::with_atoms(shells, offsets, name)
}

fn atom(element: Element, position: Double3) -> Atom {
    Atom { position, element }
}

fn element(z: i32) -> Element {
    Element::from_z(z as u8).unwrap()
}

fn ccpvdz(element: Element) -> Vec<Shell> {
    reference_shells("cc-pvdz", i32::from(element.z()))
}

fn sto3g(element: Element) -> Vec<Shell> {
    reference_shells("sto-3g", i32::from(element.z()))
}

/// An s shell of one unit-coefficient primitive.
fn s_shell(exponent: f64) -> Shell {
    Shell::new(
        0,
        vec![Primitive {
            exponent,
            coefficient: 1.0,
        }],
        ORIGIN,
    )
}

/// `⟨a|b⟩` for one m of two same-l shells on one center, in this tree's
/// normalization: coefficients multiply unnormalized primitives, and a
/// spherical function has the norm of the shell's `x^l` primitive,
/// `∫ x^{2l} exp(−g r²) = (2l−1)!! π^{3/2} / ((2g)^l g^{3/2})`.
fn same_l_overlap(a: &Shell, b: &Shell) -> f64 {
    let l = a.angular_momentum();
    let mut dfact = 1.0;
    let mut k = 2 * l - 1;
    while k > 1 {
        dfact *= f64::from(k);
        k -= 2;
    }
    let mut sum = 0.0;
    for p in a.primitives() {
        for q in b.primitives() {
            let g = p.exponent + q.exponent;
            sum += p.coefficient * q.coefficient * dfact * (std::f64::consts::PI / g).powf(1.5)
                / (2.0 * g).powi(l);
        }
    }
    sum
}

// --- Against PySCF ------------------------------------------------------------

/// `Sad.AtomBlockMatchesPyscfAtomSphAverageRhf`: H–Ar and Zn in STO-3G, H,
/// He, C, O, Ne in cc-pVDZ, and C and O in cc-pVTZ (f shells).
///
/// PySCF's per-l density blocks are over unit-normalized AOs; a function
/// here is `N_i = √⟨i|i⟩` times PySCF's, so its density element is the
/// reference one over `N_i N_j`. Every element is compared, so the zeros
/// between different (l, m) are checked too.
#[test]
fn atom_block_matches_pyscf_atom_sph_average_rhf() {
    let atoms = REFERENCE["atoms"].as_array().unwrap();
    assert!(!atoms.is_empty());
    for reference in atoms {
        let basis_name = reference["basis"].as_str().unwrap();
        let z = reference["z"].as_i64().unwrap() as i32;
        let label = format!("{}/{basis_name}", reference["symbol"].as_str().unwrap());
        let blocks: Vec<Vec<Vec<f64>>> =
            serde_json::from_value(reference["density_blocks"].clone()).unwrap();
        let shells = reference_shells(basis_name, z);

        // Per l, each shell of that momentum's first AO and norm, in shell
        // order.
        let mut first = vec![Vec::new(); blocks.len()];
        let mut norm = vec![Vec::new(); blocks.len()];
        let mut n = 0;
        for shell in &shells {
            let l = shell.angular_momentum() as usize;
            assert!(l < blocks.len(), "{label}");
            first[l].push(n);
            norm[l].push(same_l_overlap(shell, shell).sqrt());
            n += shell.function_count_sphe() as usize;
        }
        let mut expected = vec![0.0; n * n];
        for l in 0..blocks.len() {
            assert_eq!(blocks[l].len(), first[l].len(), "{label} l={l}");
            for j in 0..first[l].len() {
                for i in 0..first[l].len() {
                    for m in 0..2 * l + 1 {
                        expected[(first[l][i] + m) + (n * (first[l][j] + m))] =
                            blocks[l][i][j] / (norm[l][i] * norm[l][j]);
                    }
                }
            }
        }

        let density = atom_density(element(z), shells).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(density.len(), n * n, "{label}");
        for i in 0..n * n {
            assert!(
                (density[i] - expected[i]).abs() <= 1e-7,
                "{label} ({}, {}): {} against {}",
                i % n,
                i / n,
                density[i],
                expected[i]
            );
        }
    }
}

// --- The assembly ---------------------------------------------------------------

/// H2O in STO-3G: O's 5 AOs, then each H's one.
fn water() -> [Atom; 3] {
    [
        atom(Element::O, ORIGIN),
        atom(Element::H, at(1.43, 0.21, -1.07)),
        atom(Element::H, at(-1.38, 0.35, -1.12)),
    ]
}

/// `Sad.IsBlockDiagonalOverTheAtoms`.
#[test]
fn is_block_diagonal_over_the_atoms() {
    let atoms = water();
    let basis = grouped_basis(&atoms, sto3g, "");
    let density = sad_density(&basis, &atoms, None).unwrap();
    let n = basis.nao_sphe() as usize;
    assert_eq!(n, 7);
    let o_alone = atom_density(Element::O, sto3g(Element::O)).unwrap();
    let h_alone = atom_density(Element::H, sto3g(Element::H)).unwrap();
    // O is the leading 5 × 5 block, each H one diagonal entry; nothing
    // between.
    for j in 0..n {
        for i in 0..n {
            let expected = if i < 5 && j < 5 {
                o_alone[i + 5 * j]
            } else if i == j {
                h_alone[0]
            } else {
                0.0
            };
            assert_eq!(density[i + n * j], expected, "({i}, {j})");
        }
    }
}

/// `Sad.SeatsSubshellsByMomentumNotShellPosition`: O's STO-3G shells are s,
/// s, p; put the p first. The k-th subshell of momentum l still goes on the
/// l block's k-th orbital, so the density is the same one with its rows and
/// columns moved: p (3 AOs) then the two s.
#[test]
fn seats_subshells_by_momentum_not_shell_position() {
    let o = sto3g(Element::O);
    assert_eq!(o.len(), 3);
    let in_order = atom_density(Element::O, o.clone()).unwrap();
    let p_first = atom_density(Element::O, vec![o[2].clone(), o[0].clone(), o[1].clone()]).unwrap();
    let moved = [3, 4, 0, 1, 2]; // in_order AO -> p_first AO
    for j in 0..5 {
        for i in 0..5 {
            assert!(
                (p_first[moved[i] + 5 * moved[j]] - in_order[i + 5 * j]).abs() <= 1e-10,
                "({i}, {j})"
            );
        }
    }
}

/// `Sad.TaskMatchesSerialBitwiseOverFormaldehyde`'s assembly half: H2CO,
/// C2v in the yz plane, bohr, three elements and a repeated one, each atom's
/// block bitwise its guess alone (which the PySCF case holds to PySCF) and
/// nothing between atoms.
#[test]
fn is_block_diagonal_over_formaldehyde() {
    let atoms = [
        atom(Element::C, ORIGIN),
        atom(Element::O, at(0.0, 0.0, 2.2771)),
        atom(Element::H, at(0.0, 1.7817, -1.1111)),
        atom(Element::H, at(0.0, -1.7817, -1.1111)),
    ];
    let basis = grouped_basis(&atoms, ccpvdz, "");
    let n = basis.nao_sphe() as usize;
    assert_eq!(n, 38); // C and O: 3s 2p 1d = 14 each; each H: 2s 1p = 5
    let density = sad_density(&basis, &atoms, None).unwrap();

    // C [0, 14), O [14, 28), H [28, 33) and [33, 38).
    let bounds = [0, 14, 28, 33, 38];
    let alone: Vec<Vec<f64>> = atoms
        .iter()
        .map(|a| atom_density(a.element, ccpvdz(a.element)).unwrap())
        .collect();
    let atom_of = |ao: usize| bounds.partition_point(|&b| b <= ao) - 1;
    for j in 0..n {
        for i in 0..n {
            let a = atom_of(i);
            let expected = if a == atom_of(j) {
                let m = bounds[a + 1] - bounds[a];
                alone[a][(i - bounds[a]) + m * (j - bounds[a])]
            } else {
                0.0
            };
            assert_eq!(density[i + n * j], expected, "({i}, {j})");
        }
    }
}

/// `Sad.TaskReportsTheFirstFailingAtomsError`'s serial half: H seats; O and
/// then N need a p orbital, and two s shells have none. The error is the
/// first failing atom's, O's.
#[test]
fn reports_the_first_failing_atoms_error() {
    let atoms = [
        atom(Element::H, ORIGIN),
        atom(Element::O, at(0.0, 0.0, 2.0)),
        atom(Element::N, at(0.0, 0.0, 4.0)),
    ];
    let basis = grouped_basis(
        &atoms,
        |e| {
            if e == Element::H {
                ccpvdz(e)
            } else {
                vec![s_shell(10.0), s_shell(1.0)]
            }
        },
        "",
    );
    let error = sad_density(&basis, &atoms, None).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    assert!(error.message.contains("Z=8"), "{}", error.message);
}

// --- Baked stamps ---------------------------------------------------------------

/// A lookup that knows only hydrogen in a set called "baked", and gives it a
/// block no SCF would: STO-3G H is one AO, holding one electron.
fn baked_hydrogen(basis: &str, z: i32) -> Option<Vec<f64>> {
    (basis == "baked" && z == 1).then(|| vec![0.25])
}

/// The same, with a block of the wrong size for those shells.
fn misbaked_hydrogen(_basis: &str, z: i32) -> Option<Vec<f64>> {
    (z == 1).then(|| vec![0.25, 0.0, 0.0, 0.25])
}

/// `Sad.ABakedStampStandsInForItsElementsScf`: both H take the baked block;
/// O, which the lookup lacks, is computed.
#[test]
fn a_baked_stamp_stands_in_for_its_elements_scf() {
    let atoms = water();
    let basis = grouped_basis(&atoms, sto3g, "baked");
    let mut expected = sad_density(&basis, &atoms, None).unwrap();
    expected[5 + 7 * 5] = 0.25;
    expected[6 + 7 * 6] = 0.25;
    let lookup: SadStampLookup = baked_hydrogen;
    assert_eq!(sad_density(&basis, &atoms, Some(lookup)).unwrap(), expected);
}

/// A lookup that has oxygen, for a basis whose oxygen cannot seat its SCF.
fn baked_oxygen(_basis: &str, z: i32) -> Option<Vec<f64>> {
    (z == 8).then(|| vec![1.0, 0.0, 0.0, 1.0])
}

/// A baked block skips its element's SCF: oxygen over two s shells is
/// refused by its SCF (`refuses_a_basis_that_cannot_seat_the_atom`), and
/// succeeds, with the baked block, when the lookup has it.
#[test]
fn a_baked_stamp_skips_its_elements_scf() {
    let atoms = [atom(Element::O, ORIGIN)];
    let basis = grouped_basis(&atoms, |_| vec![s_shell(10.0), s_shell(1.0)], "baked");
    assert!(sad_density(&basis, &atoms, None).is_err());
    assert_eq!(
        sad_density(&basis, &atoms, Some(baked_oxygen)).unwrap(),
        [1.0, 0.0, 0.0, 1.0]
    );
}

/// `Sad.ABasisWithoutANameComputesEveryElement`: the name is what says the
/// shells are the set the stamps were computed in.
#[test]
fn a_basis_without_a_name_computes_every_element() {
    let atoms = water();
    let basis = grouped_basis(&atoms, sto3g, "");
    let computed = sad_density(&basis, &atoms, None).unwrap();
    assert_eq!(
        sad_density(&basis, &atoms, Some(misbaked_hydrogen)).unwrap(),
        computed
    );
}

/// `Sad.ABakedBlockOfTheWrongSizeIsInvalidConfig`.
#[test]
fn a_baked_block_of_the_wrong_size_is_invalid_config() {
    let atoms = water();
    let basis = grouped_basis(&atoms, sto3g, "baked");
    let error = sad_density(&basis, &atoms, Some(misbaked_hydrogen)).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
}

/// `Sad.GhostAtomIsEmpty`.
#[test]
fn ghost_atom_is_empty() {
    let density = atom_density(Element::Sentinel, sto3g(Element::H)).unwrap();
    assert_eq!(density, [0.0]);
}

// --- Refusals -------------------------------------------------------------------

/// `Sad.RefusesABasisNotGroupedOverTheAtoms`.
#[test]
fn refuses_a_basis_not_grouped_over_the_atoms() {
    let one = [atom(Element::H, ORIGIN)];
    let ungrouped = BasisSet::new(sto3g(Element::H));
    assert_eq!(
        sad_density(&ungrouped, &one, None).unwrap_err().kind,
        ErrorKind::InvalidConfig
    );
    let two = [atom(Element::H, ORIGIN), atom(Element::H, ORIGIN)];
    let grouped = BasisSet::with_atoms(sto3g(Element::H), vec![0, 1], "");
    assert_eq!(
        sad_density(&grouped, &two, None).unwrap_err().kind,
        ErrorKind::InvalidConfig
    );
}

/// `Sad.RefusesABasisThatCannotSeatTheAtom`: O needs a p orbital; two s
/// shells have none.
#[test]
fn refuses_a_basis_that_cannot_seat_the_atom() {
    let error = atom_density(Element::O, vec![s_shell(10.0), s_shell(1.0)]).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
}

/// An atom past 96 Cartesian AOs: ten f shells are 100, refused before any
/// integral is evaluated.
#[test]
fn refuses_an_atom_past_96_cartesian_aos() {
    let f = Shell::new(
        3,
        vec![Primitive {
            exponent: 1.0,
            coefficient: 1.0,
        }],
        ORIGIN,
    );
    let error = atom_density(Element::C, vec![f; 10]).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Unsupported);
}

/// One atom's shells on two centers are not an atom's: a basis grouped by
/// hand can say so, where `place` never would.
#[test]
fn refuses_an_atom_whose_shells_are_on_two_centers() {
    let shells = vec![
        s_shell(1.0),
        Shell::new(
            0,
            vec![Primitive {
                exponent: 0.5,
                coefficient: 1.0,
            }],
            at(0.0, 0.0, 1.0),
        ),
    ];
    let error = atom_density(Element::He, shells).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
}
