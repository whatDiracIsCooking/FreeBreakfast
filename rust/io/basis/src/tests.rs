//! `test/io/basis_host.cpp`'s orbital-set cases, and
//! `test/gtest/io_basis_gpu.cpp`'s, which need no device here, one test per
//! case, each named after it; and the checks the C++ build makes by
//! construction: the table's names against the documents, and every
//! `data/basis/*.json` in it. Unit tests, so the table stays private.

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

/// What the C++ build gets by construction: each name in the table is its
/// document's own `name`, and every `data/basis/*.json` is in the table.
#[test]
fn the_table_is_data_basis() {
    for (name, json) in BUILTIN {
        let document: Value = serde_json::from_str(json).unwrap();
        assert_eq!(document["name"], name);
    }
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../data/basis");
    let mut files = 0;
    for entry in std::fs::read_dir(&dir).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "json") {
            files += 1;
            let text = std::fs::read_to_string(&path).unwrap();
            assert!(
                BUILTIN.iter().any(|(_, json)| *json == text),
                "{} is not compiled in",
                path.display()
            );
        }
    }
    assert_eq!(files, BUILTIN.len());
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
