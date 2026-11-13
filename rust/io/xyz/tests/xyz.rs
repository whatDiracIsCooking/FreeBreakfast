//! `test/gtest/io_xyz_gpu.cpp`, with no device, one test per case, each
//! named after it. A C++ `Molecule` owns its nuclei on the device, so even
//! a refusal there needs a GPU; an `fb_molecule::Molecule` is host data.
//!
//! An integration test rather than unit tests, for `CARGO_TARGET_TMPDIR`,
//! the scratch directory the disk cases write into.

use std::path::PathBuf;

use fb_atom::Element;
use fb_core_constants::BOHR_PER_ANGSTROM;
use fb_core_error_handling::ErrorKind;
use fb_io_xyz::{parse_xyz, read_xyz};

const WATER: &str = "3\n\
                     water, Angstrom\n\
                     O   0.000000   0.000000   0.117300\n\
                     H   0.000000   0.757200  -0.469200\n\
                     H   0.000000  -0.757200  -0.469200\n";

/// A fresh scratch directory of this binary's, `ScratchDir`.
fn scratch(name: &str) -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join(name);
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// `IoXyz.ParsesAtomsAndConvertsAngstromToBohr`. The C++ compares within
/// four ulps; the product here is the same one, so it is compared exactly.
#[test]
fn parses_atoms_and_converts_angstrom_to_bohr() {
    let molecule = parse_xyz(WATER, 0, 1).unwrap();
    assert_eq!(molecule.natoms(), 3);
    assert_eq!(molecule.charge(), 0);
    assert_eq!(molecule.multiplicity(), 1);

    let atoms = molecule.atoms();
    assert_eq!(atoms[0].element, Element::O);
    assert_eq!(atoms[1].element, Element::H);
    assert_eq!(atoms[2].element, Element::H);
    assert_eq!(atoms[0].position.z, 0.1173 * BOHR_PER_ANGSTROM);
    assert_eq!(atoms[1].position.y, 0.7572 * BOHR_PER_ANGSTROM);
    assert_eq!(atoms[2].position.z, -0.4692 * BOHR_PER_ANGSTROM);
    assert_eq!(molecule.electron_count(), 10);
}

/// `IoXyz.AttachesChargeAndMultiplicity`.
#[test]
fn attaches_charge_and_multiplicity() {
    let molecule = parse_xyz(WATER, 1, 2).unwrap();
    assert_eq!(molecule.charge(), 1);
    assert_eq!(molecule.multiplicity(), 2);
    assert_eq!(molecule.electron_count(), 9);
}

/// `IoXyz.ToleratesCrlfTrailingBlanksAndSymbolCase`.
#[test]
fn tolerates_crlf_trailing_blanks_and_symbol_case() {
    let molecule = parse_xyz("2\r\ncomment\r\nCL 0 0 0\r\nna 0 0 2.5\r\n\r\n   \n", 0, 1).unwrap();
    assert_eq!(molecule.natoms(), 2);
    assert_eq!(molecule.atoms()[0].element, Element::Cl);
    assert_eq!(molecule.atoms()[1].element, Element::Na);
}

/// `IoXyz.EmptyGeometryIsIoError`: a `Molecule` needs at least one atom, so
/// the reader refuses before building one.
#[test]
fn empty_geometry_is_io_error() {
    let error = parse_xyz("0\nnothing here\n", 0, 1).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Io);
}

/// `IoXyz.MalformedInputIsIoError`, and below its cases the ones where a
/// whole-token field refuses what `operator>>` would read a prefix of (the
/// crate doc's "Each field is the whole token").
#[test]
fn malformed_input_is_io_error() {
    for text in [
        "3\n",                            // no comment line
        "three\ncomment\n",               // count not a number
        "-1\ncomment\n",                  // negative count
        "1\ncomment\nH 0 0\n",            // missing coordinate
        "1\ncomment\nH 0 zero 0\n",       // non-numeric coordinate
        "1\ncomment\nXx 0 0 0\n",         // unknown element
        "1\ncomment\nHel 0 0 0\n",        // symbol too long
        "2\ncomment\nH 0 0 0\n",          // count too high
        "1\ncomment\nH 0 0 0\nH 0 0 1\n", // count too low
        "",                               // nothing at all
        "1abc\ncomment\nH 0 0 0\n",       // count with a tail
        "1\ncomment\nH 0 0 0.5x\n",       // coordinate with a tail
        "1\ncomment\nH 0 nan 0\n",        // not finite
        "1\ncomment\nH 0 0 1e400\n",      // past f64's range
    ] {
        let error = parse_xyz(text, 0, 1).unwrap_err();
        assert_eq!(error.kind, ErrorKind::Io, "{text:?}");
    }
}

/// `IoXyz.ReadsFromDisk`.
#[test]
fn reads_from_disk() {
    let path = scratch("io_xyz_reads").join("water.xyz");
    std::fs::write(&path, WATER).unwrap();
    let molecule = read_xyz(&path, 0, 1).unwrap();
    assert_eq!(molecule.natoms(), 3);
}

/// `IoXyz.ReadOfMissingFileIsIoError`.
#[test]
fn read_of_missing_file_is_io_error() {
    let path = scratch("io_xyz_missing").join("absent.xyz");
    let error = read_xyz(&path, 0, 1).unwrap_err();
    assert_eq!(error.kind, ErrorKind::Io);
    assert!(error.message.contains("absent.xyz"), "{}", error.message);
}
