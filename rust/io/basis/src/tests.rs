//! `test/io/basis_host.cpp`'s cases, orbital and fitting (`IoBasisAux`),
//! and `test/gtest/io_basis_gpu.cpp`'s, which need no device here, one test
//! per case, each named after it; and the checks the C++ build makes by
//! construction: each table's names against the documents, and every
//! `data/basis/*.json` and `data/aux_basis/*.json` in its table. Unit tests,
//! so the tables stay private.

use std::path::PathBuf;

use fb_atom::{Atom, Element};
use fb_core_types::Double3;
use fb_molecule::Molecule;
use serde_json::Value;

use super::*;

/// `test/io/io_basis_json.h`'s `kBasis`: STO-3G hydrogen; oxygen with a
/// plain s, an sp block (one column per momentum), a two-column general
/// contraction, and a tagged spherical d.
const BASIS: &str = r#"{
  "elements": {
    "1": {"electron_shells": [
      {"function_type": "gto", "angular_momentum": [0],
       "exponents": ["3.42525091", "0.62391373", "0.16885540"],
       "coefficients": [["0.15432897", "0.53532814", "0.44463454"]]}
    ]},
    "8": {"electron_shells": [
      {"angular_momentum": [0],
       "exponents": ["130.7093200"], "coefficients": [["1.0"]]},
      {"function_type": "gto", "angular_momentum": [0, 1],
       "exponents": ["5.0331513", "1.1695961"],
       "coefficients": [["-0.09996723", "0.39951283"], ["0.15591627", "0.60768372"]]},
      {"function_type": "gto", "angular_momentum": [0],
       "exponents": ["0.38", "0.12"],
       "coefficients": [["1.0", "0.0"], ["0.0", "1.0"]]},
      {"function_type": "gto_spherical", "angular_momentum": [2],
       "exponents": ["1.185"], "coefficients": [["1.0E+00"]]}
    ]}
  }
}"#;

/// A fresh scratch directory, `ScratchDir`: under the system's temporary
/// directory, by process so two runs never share one.
fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("fb-io-basis-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The spherical function count of element `z` in `stamp`.
fn functions(stamp: &BasisStamp, z: i32) -> u32 {
    let shells = stamp.shells_for(z).unwrap();
    shells.iter().map(ShellStamp::function_count_sphe).sum()
}

/// `IoBasis.ParsesElementsIntoShellStamps`.
#[test]
fn parses_elements_into_shell_stamps() {
    let stamp = parse_basis(BASIS).unwrap();
    assert_eq!(stamp.element_shells().len(), 2);
    let h = stamp.shells_for(1).unwrap();
    assert_eq!(h.len(), 1);
    assert_eq!(h[0].angular_momentum(), 0);
    assert_eq!(h[0].nprimitives(), 3);
    assert_eq!(h[0].primitives()[0].exponent, 3.42525091);
    assert_eq!(h[0].primitives()[2].coefficient, 0.44463454);
}

/// `IoBasis.SplitsSpAndGeneralContractionBlocks`.
#[test]
fn splits_sp_and_general_contraction_blocks() {
    let stamp = parse_basis(BASIS).unwrap();
    let o = stamp.shells_for(8).unwrap();
    // s, then sp -> s + p, then the general contraction -> s + s, then d.
    let momenta: Vec<i32> = o.iter().map(ShellStamp::angular_momentum).collect();
    assert_eq!(momenta, [0, 0, 1, 0, 0, 2]);
    // The sp block's p shell takes the second column over the shared exponents.
    assert_eq!(o[2].primitives()[1].exponent, 1.1695961);
    assert_eq!(o[2].primitives()[1].coefficient, 0.60768372);
    // Each general-contraction column is its own shell.
    assert_eq!(o[3].primitives()[0].coefficient, 1.0);
    assert_eq!(o[4].primitives()[0].coefficient, 0.0);
    assert_eq!(o[4].primitives()[1].coefficient, 1.0);
}

/// `IoBasis.ReadBasisFromFile`.
#[test]
fn read_basis_from_file() {
    let dir = scratch("read");
    std::fs::write(dir.join("basis.json"), BASIS).unwrap();
    let stamp = read_basis(&dir.join("basis.json")).unwrap();
    assert_eq!(stamp.element_shells().len(), 2);

    let missing = read_basis(&dir.join("absent.json")).unwrap_err();
    assert_eq!(missing.kind, ErrorKind::Io);
}

/// `IoBasis.MalformedJsonIsIoError`.
#[test]
fn malformed_json_is_io_error() {
    let error = parse_basis(r#"{"elements": "#).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Io);
}

/// `IoBasis.WrongContentIsInvalidConfig`, its ten documents, then the
/// three the crate doc adds: a negative momentum (an abort in the C++), and
/// a leading `+` on an exponent and on an element key.
#[test]
fn wrong_content_is_invalid_config() {
    let shell = |body: &str| format!(r#"{{"elements": {{"1": {{"electron_shells": [{body}]}}}}}}"#);
    let bad = [
        "[]".to_owned(),
        "{}".to_owned(),
        r#"{"elements": []}"#.to_owned(),
        r#"{"elements": {"H": {"electron_shells": []}}}"#.to_owned(),
        r#"{"elements": {"1": {}}}"#.to_owned(),
        shell(r#"{"angular_momentum": [0], "exponents": ["1.0x"], "coefficients": [["1.0"]]}"#),
        shell(r#"{"angular_momentum": [0], "exponents": [1.0], "coefficients": [["1.0"]]}"#),
        shell(
            r#"{"angular_momentum": [0], "exponents": ["1.0", "2.0"], "coefficients": [["1.0"]]}"#,
        ),
        shell(
            r#"{"angular_momentum": [0, 1, 2], "exponents": ["1.0"],
                "coefficients": [["1.0"], ["1.0"]]}"#,
        ),
        shell(r#"{"angular_momentum": [0], "exponents": ["1.0"]}"#),
        shell(r#"{"angular_momentum": [-1], "exponents": ["1.0"], "coefficients": [["1.0"]]}"#),
        shell(r#"{"angular_momentum": [0], "exponents": ["+1.0"], "coefficients": [["1.0"]]}"#),
        r#"{"elements": {"+1": {"electron_shells": []}}}"#.to_owned(),
    ];
    for text in &bad {
        let error = parse_basis(text).unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidConfig, "{text}");
    }
}

/// `IoBasis.OnlySphericalFunctionTypesAreAccepted`.
#[test]
fn only_spherical_function_types_are_accepted() {
    let shell_with = |function_type: &str, l: i32| {
        let tag = if function_type.is_empty() {
            String::new()
        } else {
            format!(r#""function_type": "{function_type}", "#)
        };
        parse_basis(&format!(
            r#"{{"elements": {{"1": {{"electron_shells": [{{{tag}"angular_momentum": [{l}],
                "exponents": ["1.0"], "coefficients": [["1.0"]]}}]}}}}}}"#
        ))
    };
    // An explicit spherical tag is accepted at any l.
    for l in [0, 2, 3] {
        assert!(shell_with("gto_spherical", l).is_ok(), "l={l}");
    }
    // Cartesian functions are not supported, at any l.
    for l in [0, 1, 2, 3] {
        let error = shell_with("gto_cartesian", l).unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidConfig, "l={l}");
        // Refused as Cartesian, not as an unknown type, which is the same kind.
        let says = "Cartesian basis functions are not supported";
        assert!(error.message.contains(says), "{}", error.message);
    }
    // Untagged or bare "gto" is exact as spherical for s and p ...
    assert!(shell_with("", 1).is_ok());
    assert!(shell_with("gto", 0).is_ok());
    // ... and ambiguous from d up.
    for ambiguous in ["", "gto"] {
        let error = shell_with(ambiguous, 2).unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidConfig, "'{ambiguous}'");
    }
    // Slater-type orbitals are not Gaussian shells.
    let error = shell_with("sto", 0).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
}

/// `IoBasisBuiltins.EveryCompiledInSetParsesAndCoversItsElements`. The
/// element counts pin `data/basis/`: a set whose range silently changed, or
/// one re-downloaded with Cartesian shells the reader refuses, fails here.
#[test]
fn every_compiled_in_set_parses_and_covers_its_elements() {
    for (name, elements) in [
        ("sto-3g", 36),
        ("6-31g", 36),
        ("6-31g*", 36),
        ("6-31g**", 36),
        ("def2-SVP", 36),
        ("def2-TZVP", 36),
        ("cc-pVDZ", 35),
        ("cc-pVTZ", 35),
        ("aug-cc-pVDZ", 34),
        ("aug-cc-pVTZ", 34),
        ("minao", 35),
    ] {
        let stamp = resolve_basis(name).unwrap_or_else(|e| panic!("{name}: {}", e.message));
        assert_eq!(stamp.element_shells().len(), elements, "{name}");
    }

    // STO-3G runs to krypton; the correlation-consistent sets have no
    // potassium (BSE defines none).
    assert!(resolve_basis("sto-3g").unwrap().shells_for(36).is_some());
    let dz = resolve_basis("cc-pvdz").unwrap();
    assert!(dz.shells_for(19).is_none());
    assert!(dz.shells_for(20).is_some());

    // 6-31G is carried relabelled spherical from K up: K-Kr are s, 4 sp and
    // 2 d, 27 functions as 5D, where 6D would be 29.
    let pople = resolve_basis("6-31g").unwrap();
    for z in [19, 26, 36] {
        assert_eq!(functions(&pople, z), 27, "Z={z}");
    }

    // 6-31G* and 6-31G** carry their d 5D the same way: C is 6-31G's 9 plus
    // 5 d. ** differs from * only in H and He's p shell.
    for (name, z, count) in [
        ("6-31g*", 1, 2),
        ("6-31g*", 6, 14),
        ("6-31g**", 1, 5),
        ("6-31g**", 6, 14),
    ] {
        assert_eq!(
            functions(&resolve_basis(name).unwrap(), z),
            count,
            "{name} Z={z}"
        );
    }

    // The augmented cc sets have neither potassium nor calcium on BSE.
    let aug = resolve_basis("aug-cc-pvdz").unwrap();
    assert!(aug.shells_for(19).is_none());
    assert!(aug.shells_for(20).is_none());
    assert!(aug.shells_for(21).is_some());

    // MINAO is minimal: one function per occupied free-atom AO.
    let minao = resolve_basis("minao").unwrap();
    assert_eq!(minao.name(), "MINAO");
    assert!(minao.shells_for(19).is_none());
    for (z, count) in [(1, 1), (6, 5), (20, 10), (26, 15), (36, 18)] {
        assert_eq!(functions(&minao, z), count, "Z={z}");
    }
}

/// What the C++ build gets by construction: each name in `table` is its
/// document's own `name`, and every `*.json` of `data/<dir>/` is in `table`.
fn check_table(table: &[(&str, &str)], dir: &str) {
    for (name, json) in table {
        let document: Value = serde_json::from_str(json).unwrap();
        assert_eq!(document["name"], *name);
    }
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../data")
        .join(dir);
    let mut files = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "json") {
            files += 1;
            let text = std::fs::read_to_string(&path).unwrap();
            assert!(
                table.iter().any(|(_, json)| *json == text),
                "{} is not compiled in",
                path.display()
            );
        }
    }
    assert_eq!(files, table.len());
}

/// [`check_table`] over `data/basis/`.
#[test]
fn the_table_is_data_basis() {
    check_table(&BUILTIN, "basis");
}

/// `IoBasisBuiltins.NamesIgnoreCaseHyphensAndUnderscores`.
#[test]
fn names_ignore_case_hyphens_and_underscores() {
    for spelling in ["STO-3G", "sto-3g", "sto3g", "STO_3G", "s t o-3 g"] {
        assert!(is_builtin_basis(spelling), "{spelling}");
    }
    for spelling in ["cc-pVTZ", "ccpvtz", "CC-PVTZ"] {
        assert!(is_builtin_basis(spelling), "{spelling}");
    }
    // Normalization folds separators, never the characters that tell one
    // set from another: 6-31G, 6-31G* and 6-31G** are three bases, and
    // 6-311G is not compiled in.
    for name in ["6-31G", "6-31G*", "6-31G**"] {
        assert_eq!(resolve_basis(name).unwrap().name(), name);
    }
    assert!(!is_builtin_basis("6-311g"));
    assert!(!is_builtin_basis("cc-pvqz"));
    assert!(!is_builtin_basis(""));
}

/// `IoBasisBuiltins.AnUnknownBasisReadsAsAPathAndSaysWhatTheNamesAre`:
/// whatever the user meant, what failed was opening a path.
#[test]
fn an_unknown_basis_reads_as_a_path_and_says_what_the_names_are() {
    let error = resolve_basis("sto-3gg").unwrap_err();
    assert_eq!(error.kind, ErrorKind::Io);
    assert!(error.message.contains("STO-3G"), "{}", error.message);
}

/// `IoBasisBuiltins.APathIsStillRead`: a set this build does not carry
/// needs no rebuild.
#[test]
fn a_path_is_still_read() {
    let path = scratch("path").join("custom.json");
    std::fs::write(&path, BASIS).unwrap();
    let stamp = resolve_basis(path.to_str().unwrap()).unwrap();
    assert_eq!(stamp.element_shells().len(), 2);
}

/// `IoBasisBuiltins.OnlyACompiledInSetIsNamed`: the canonical spelling,
/// whatever asked, kept by `primitive_normalized`; a file unnamed even when
/// its document claims a compiled-in name.
#[test]
fn only_a_compiled_in_set_is_named() {
    let sto = resolve_basis("sto3g").unwrap();
    assert_eq!(sto.name(), "STO-3G");
    assert_eq!(sto.primitive_normalized().name(), "STO-3G");

    let path = scratch("named").join("claims-sto-3g.json");
    std::fs::write(&path, format!(r#"{{"name": "STO-3G", {}"#, &BASIS[1..])).unwrap();
    assert_eq!(resolve_basis(path.to_str().unwrap()).unwrap().name(), "");
    assert_eq!(read_basis(&path).unwrap().name(), "");
}

/// The issue's fourth refusal, `test/gtest/io_basis_gpu.cpp`'s shape with
/// no device: a compiled-in set placed onto a molecule with an element it
/// lacks (potassium in cc-pVDZ) is `InvalidConfig`, naming the Z.
#[test]
fn an_element_the_set_lacks_is_invalid_config() {
    let k = Molecule::new(
        vec![Atom {
            position: Double3::default(),
            element: Element::K,
        }],
        0,
        2,
    );
    let error = resolve_basis("cc-pvdz").unwrap().place(&k).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    assert!(error.message.contains("Z=19"), "{}", error.message);
    assert!(resolve_basis("sto-3g").unwrap().place(&k).is_ok());
}

/// `test/gtest/io_basis_gpu.cpp`'s `water`, positions in bohr.
fn water() -> Molecule {
    let at = |y: f64, z: f64, element: Element| Atom {
        position: Double3 { x: 0.0, y, z },
        element,
    };
    Molecule::new(
        vec![
            at(0.0, 0.2217, Element::O),
            at(1.4309, -0.8867, Element::H),
            at(-1.4309, -0.8867, Element::H),
        ],
        0,
        1,
    )
}

/// `IoBasis.ParseAndAssignLaysTheBasisOverTheMolecule`, a GPU case in the
/// C++ for its `Molecule`, through `parse_basis` and `place`.
#[test]
fn parse_and_assign_lays_the_basis_over_the_molecule() {
    let basis = parse_basis(BASIS).unwrap().place(&water()).unwrap();
    // O: 6 shells (s, s, p, s, s, d), then one s per H.
    assert_eq!(basis.nshells(), 8);
    assert_eq!(basis.nao_cart(), 1 + 1 + 3 + 1 + 1 + 6 + 1 + 1);
    assert_eq!(basis.nao_sphe(), basis.nao_cart() - 1); // d: 5, not 6
    assert_eq!(
        fb_basis::spherical_transform_matrix(&basis).len(),
        basis.nao_cart() as usize * basis.nao_sphe() as usize
    );
    assert_eq!(basis.shells()[6].center().y, 1.4309);
    assert_eq!(basis.shells()[7].center().y, -1.4309);
}

/// `IoBasis.UncoveredElementIsInvalidConfig`, a GPU case in the C++.
#[test]
fn uncovered_element_is_invalid_config() {
    let co = Molecule::new(
        vec![
            Atom {
                position: Double3::default(),
                element: Element::C,
            },
            Atom {
                position: Double3 {
                    x: 0.0,
                    y: 0.0,
                    z: 2.1,
                },
                element: Element::O,
            },
        ],
        0,
        1,
    );
    let error = parse_basis(BASIS).unwrap().place(&co).unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    assert!(error.message.contains("Z=6"), "{}", error.message);
}

// The compiled-in fitting sets (`data/aux_basis/`, their own table) and
// `resolve_aux_basis`: `test/io/basis_host.cpp`'s `IoBasisAux` cases, then
// `test/gtest/io_basis_gpu.cpp`'s, which need no device here.

/// [`check_table`] over `data/aux_basis/`.
#[test]
fn the_aux_table_is_data_aux_basis() {
    check_table(&BUILTIN_AUX, "aux_basis");
}

/// The issue's second acceptance line: every set of `data/aux_basis/`
/// resolves by its name, parses, and comes back named with it.
#[test]
fn every_compiled_in_fitting_set_resolves_by_name_and_parses() {
    for (name, _) in BUILTIN_AUX {
        let stamp = resolve_aux_basis(name).unwrap_or_else(|e| panic!("{name}: {}", e.message));
        assert_eq!(stamp.name(), name);
        assert!(!stamp.element_shells().is_empty(), "{name}");
    }
}

/// The highest angular momentum among `shells`.
fn top_momentum(shells: &[ShellStamp]) -> i32 {
    shells
        .iter()
        .map(ShellStamp::angular_momentum)
        .max()
        .unwrap()
}

/// `IoBasisAux.BothJkfitSetsResolveByAnySpelling`: both are
/// `tools/jkfit_from_pyscf.py`'s, PySCF's 19 elements, H, B-Ne, Al-Ar,
/// Ga-Kr.
#[test]
fn both_jkfit_sets_resolve_by_any_spelling() {
    for (spec, canonical) in [
        ("cc-pVDZ-JKFIT", "cc-pVDZ-JKFIT"),
        ("ccpvdzjkfit", "cc-pVDZ-JKFIT"),
        ("cc-pVTZ-JKFIT", "cc-pVTZ-JKFIT"),
        ("CC_PVTZ_JKFIT", "cc-pVTZ-JKFIT"),
    ] {
        let stamp = resolve_aux_basis(spec).unwrap_or_else(|e| panic!("{spec}: {}", e.message));
        assert_eq!(stamp.name(), canonical);
        assert_eq!(stamp.element_shells().len(), 19, "{spec}");
        assert!(stamp.shells_for(36).is_some(), "{spec}");
        assert!(stamp.shells_for(2).is_none(), "{spec}");
    }
}

/// Every element of the orbital set `orbital` has shells in `aux`, which
/// covers no more elements, none past momentum `lmax`.
fn covers(aux: &BasisStamp, orbital: &str, lmax: i32) {
    let orbital = resolve_basis(orbital).unwrap();
    assert_eq!(aux.element_shells().len(), orbital.element_shells().len());
    for z in orbital.element_shells().keys() {
        let fit = aux.shells_for(*z).unwrap_or_else(|| panic!("Z={z}"));
        assert!(top_momentum(fit) <= lmax, "Z={z}");
    }
}

/// `IoBasisAux.TheAccdSetCoversCcPvdzAndStopsAtG`:
/// `tools/atomic_cd/aux_set.py` over `data/basis/cc-pVDZ.json`, `--max-l
/// 4`.
#[test]
fn the_accd_set_covers_cc_pvdz_and_stops_at_g() {
    let aux = resolve_aux_basis("cc-pvdz-accd").unwrap();
    assert_eq!(aux.name(), "cc-pVDZ-acCD");
    covers(&aux, "cc-pvdz", 4);
}

/// `IoBasisAux.EveryAccdSixSixSetCoversItsOrbitalSetAndStopsAtI`:
/// `tools/atomic_cd/aux_set.py` at `--threshold 1e-6 --contract-threshold
/// 1e-6 --max-l 6`.
#[test]
fn every_accd_six_six_set_covers_its_orbital_set_and_stops_at_i() {
    for (aux_name, orbital) in [
        ("cc-pVDZ-acCD-6-6", "cc-pVDZ"),
        ("cc-pVTZ-acCD-6-6", "cc-pVTZ"),
        ("aug-cc-pVDZ-acCD-6-6", "aug-cc-pVDZ"),
        ("aug-cc-pVTZ-acCD-6-6", "aug-cc-pVTZ"),
        ("def2-SVP-acCD-6-6", "def2-SVP"),
        ("def2-TZVP-acCD-6-6", "def2-TZVP"),
        ("6-31G**-acCD-6-6", "6-31G**"),
    ] {
        let aux = resolve_aux_basis(aux_name).unwrap();
        assert_eq!(aux.name(), aux_name);
        covers(&aux, orbital, 6);
    }
}

/// `IoBasisAux.TheDef2SetIsUniversalJkfitOverHToKr`: PySCF's
/// def2-universal-jkfit, the one fitting set it pairs with every def2
/// orbital set.
#[test]
fn the_def2_set_is_universal_jkfit_over_h_to_kr() {
    let aux = resolve_aux_basis("def2-universal-jkfit").unwrap();
    assert_eq!(aux.name(), "def2-universal-JKFIT");
    assert_eq!(aux.element_shells().len(), 36);
    assert!(aux.shells_for(1).is_some());
    assert!(aux.shells_for(36).is_some());
}

/// `IoBasisAux.TheFittingAndOrbitalTablesAreDisjoint`: a fitting set is no
/// orbital set, and an orbital set's name is no fitting set's.
#[test]
fn the_fitting_and_orbital_tables_are_disjoint() {
    for aux in ["cc-pVDZ-JKFIT", "cc-pvtz-jkfit"] {
        assert!(!is_builtin_basis(aux), "{aux}");
        assert!(is_builtin_aux_basis(aux), "{aux}");
        assert_eq!(resolve_basis(aux).unwrap_err().kind, ErrorKind::Io, "{aux}");
    }
    assert!(!is_builtin_aux_basis("cc-pvdz"));
    assert!(!is_builtin_aux_basis("sto-3g"));
    assert_eq!(
        resolve_aux_basis("cc-pvdz").unwrap_err().kind,
        ErrorKind::Io
    );
    // And no name in one table folds onto a name in the other.
    for (aux, _) in BUILTIN_AUX {
        assert!(!is_builtin_basis(aux), "{aux}");
    }
}

/// `IoBasisAux.AuxNamesIgnoreCaseHyphensAndUnderscores`.
#[test]
fn aux_names_ignore_case_hyphens_and_underscores() {
    for spelling in [
        "cc-pVDZ-JKFIT",
        "ccpvdzjkfit",
        "CC_PVTZ_JKFIT",
        "cc pvtz jkfit",
    ] {
        assert!(is_builtin_aux_basis(spelling), "{spelling}");
    }
    assert!(!is_builtin_aux_basis("cc-pvqz-jkfit"));
    assert!(!is_builtin_aux_basis("cc-pvdz-jkfit.json"));
    assert!(!is_builtin_aux_basis(""));
}

/// `IoBasisAux.AnUnknownNameReadsAsAPathAndSaysWhatTheNamesAre`: the
/// fitting sets' names, not the orbital ones.
#[test]
fn an_unknown_aux_name_reads_as_a_path_and_says_what_the_names_are() {
    let error = resolve_aux_basis("cc-pvqz-jkfit").unwrap_err();
    assert_eq!(error.kind, ErrorKind::Io);
    assert!(error.message.contains("cc-pVDZ-JKFIT"), "{}", error.message);
    assert!(!error.message.contains("STO-3G"), "{}", error.message);
}

/// `IoBasisAux.APathIsStillReadAndUnnamed`.
#[test]
fn an_aux_path_is_still_read_and_unnamed() {
    let path = scratch("aux").join("custom-fit.json");
    std::fs::write(&path, BASIS).unwrap();
    let stamp = resolve_aux_basis(path.to_str().unwrap()).unwrap();
    assert_eq!(stamp.element_shells().len(), 2);
    assert_eq!(stamp.name(), "");
}

/// `IoBasisAux.TheTripleZetaSetIsTheDoubleZetaSetPlusItsTopMomentum`:
/// Molpro's cc-pVDZ-JKFIT is cc-pVTZ-JKFIT with each element's highest
/// angular momentum dropped (`data/aux_basis/README.md`).
#[test]
fn the_triple_zeta_set_is_the_double_zeta_set_plus_its_top_momentum() {
    let dz = resolve_aux_basis("cc-pvdz-jkfit").unwrap();
    let tz = resolve_aux_basis("cc-pvtz-jkfit").unwrap();
    for (z, tz_shells) in tz.element_shells() {
        let dz_shells = dz.shells_for(*z).unwrap_or_else(|| panic!("Z={z}"));
        let top = top_momentum(tz_shells);
        let kept: Vec<&ShellStamp> = tz_shells
            .iter()
            .filter(|s| s.angular_momentum() != top)
            .collect();
        assert_eq!(kept.len(), dz_shells.len(), "Z={z}");
        for (shell, matched) in kept.into_iter().zip(dz_shells) {
            assert_eq!(
                matched.angular_momentum(),
                shell.angular_momentum(),
                "Z={z}"
            );
            assert_eq!(matched.primitives(), shell.primitives(), "Z={z}");
        }
    }
}

/// `IoBasisAux.EachAugJkfitSetIsItsCcSetPlusDiffuseFunctions`: Molpro's
/// `weigend_jkfit_aug.libmol`, at each angular momentum the cc set's
/// primitives plus one even-tempered diffuse one, `e1^2/e2` from the two
/// most diffuse exponents already there, or `e1/2.5` where there is one
/// (`data/aux_basis/README.md`). Checked on H and C, as in the C++.
#[test]
fn each_aug_jkfit_set_is_its_cc_set_plus_diffuse_functions() {
    // An element's uncontracted exponents at momentum `l`, ascending.
    let exponents = |shells: &[ShellStamp], l: i32| {
        let mut out: Vec<f64> = shells
            .iter()
            .filter(|s| s.angular_momentum() == l)
            .map(|s| {
                assert_eq!(s.nprimitives(), 1);
                s.primitives()[0].exponent
            })
            .collect();
        out.sort_by(f64::total_cmp);
        out
    };
    for (aug_name, cc_name) in [
        ("aug-cc-pVDZ-JKFIT", "cc-pVDZ-JKFIT"),
        ("aug-cc-pVTZ-JKFIT", "cc-pVTZ-JKFIT"),
    ] {
        assert!(is_builtin_aux_basis(aug_name));
        assert!(!is_builtin_basis(aug_name));
        let aug = resolve_aux_basis(aug_name).unwrap();
        let cc = resolve_aux_basis(cc_name).unwrap();
        assert_eq!(aug.name(), aug_name);
        assert_eq!(aug.element_shells().len(), 19);
        assert!(aug.shells_for(2).is_none());
        assert!(aug.shells_for(36).is_some());
        for z in [1, 6] {
            let a = aug.shells_for(z).unwrap();
            let c = cc.shells_for(z).unwrap();
            let top = top_momentum(c);
            assert_eq!(a.len(), c.len() + top as usize + 1, "{aug_name} Z={z}");
            for l in 0..=top {
                let have = exponents(c, l);
                let mut with = exponents(a, l);
                assert!(!have.is_empty(), "{aug_name} Z={z} l={l}");
                assert_eq!(with.len(), have.len() + 1, "{aug_name} Z={z} l={l}");
                let added = with.remove(0);
                assert_eq!(with, have, "{aug_name} Z={z} l={l}");
                let tempered = if have.len() >= 2 {
                    have[0] * have[0] / have[1]
                } else {
                    have[0] / 2.5
                };
                // Molpro prints eight decimals, so the rule holds to their
                // rounding.
                assert!(
                    (added - tempered).abs() <= 1e-7 * tempered,
                    "{aug_name} Z={z} l={l}: {added} against {tempered}"
                );
            }
        }
    }
}

/// `IoBasisAux.AJkfitSetPlacesOverWater`, a GPU case in the C++ for its
/// `Molecule`: the counts pin `data/aux_basis/` and match PySCF's `naux`
/// for H2O, 116 and 139. O: 10s 7p 5d 2f in DZ, one g more in TZ; H: 4s 3p
/// 2d, one f more in TZ.
#[test]
fn a_jkfit_set_places_over_water() {
    for (name, nshells, nao_sphe, oxygen_lmax) in [
        ("cc-pVDZ-JKFIT", 24 + 2 * 9, 70 + 2 * 23, 3),
        ("cc-pVTZ-JKFIT", 25 + 2 * 10, 79 + 2 * 30, 4),
    ] {
        let aux = resolve_aux_basis(name)
            .unwrap()
            .primitive_normalized()
            .place(&water())
            .unwrap();
        assert_eq!(aux.nshells(), nshells, "{name}");
        assert_eq!(aux.nao_sphe(), nao_sphe, "{name}");
        let oxygen = aux.atom_shell_offset(0) as usize..aux.atom_shell_offset(1) as usize;
        let lmax = aux.shells()[oxygen]
            .iter()
            .map(|s| s.angular_momentum())
            .max()
            .unwrap();
        assert_eq!(lmax, oxygen_lmax, "{name}");
    }
}

/// `IoBasisAux.AnElementTheSetLacksIsInvalidConfig`, a GPU case in the C++:
/// the JKFIT sets have H, then B onward, so Li is refused naming its Z.
#[test]
fn an_element_the_fitting_set_lacks_is_invalid_config() {
    let lih = Molecule::new(
        vec![
            Atom {
                position: Double3::default(),
                element: Element::Li,
            },
            Atom {
                position: Double3 {
                    x: 0.0,
                    y: 0.0,
                    z: 3.0,
                },
                element: Element::H,
            },
        ],
        0,
        1,
    );
    let error = resolve_aux_basis("cc-pvdz-jkfit")
        .unwrap()
        .primitive_normalized()
        .place(&lih)
        .unwrap_err();
    assert_eq!(error.kind, ErrorKind::InvalidConfig);
    assert!(error.message.contains("Z=3"), "{}", error.message);
}
