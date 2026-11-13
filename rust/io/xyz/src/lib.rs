//! `src/io/xyz` (module `free_breakfast.io.xyz`): a standard XYZ geometry
//! read into a [`Molecule`].
//!
//! Line 1 is the atom count, line 2 a free-form comment, and each line after
//! them `symbol x y z`, the coordinates in Angstrom. A [`Molecule`] holds
//! bohr, so every coordinate is multiplied by [`BOHR_PER_ANGSTROM`] on read,
//! the one physical assumption here. XYZ carries no charge or spin, so the
//! caller supplies both.
//!
//! | C++ (`free_breakfast`) | Here |
//! |---|---|
//! | `parse_xyz(res, text, charge = 0, multiplicity = 1)` | [`parse_xyz`]`(text, charge, multiplicity)` |
//! | `read_xyz(res, path, charge = 0, multiplicity = 1)` | [`read_xyz`]`(path, charge, multiplicity)` |
//! | the `DeviceResources` both take | nothing: an `fb_molecule::Molecule` is host data |
//! | `io/base`'s `read_file`, which `read_xyz` reads through | [`std::fs::read_to_string`], its failure an [`ErrorKind::Io`](fb_core_error_handling::ErrorKind::Io) error |
//!
//! Every refusal is an [`ErrorKind::Io`](fb_core_error_handling::ErrorKind::Io)
//! error, as in the C++: fewer than two lines, a count that is not a
//! positive integer, an atom line short of four fields, an unknown symbol, a
//! coordinate that is not a finite number, and a count the atom lines
//! disagree with. A carriage return ending a line, and blank lines, are
//! tolerated rather than counted as atoms; fields past the fourth are
//! ignored.
//!
//! **Each field is the whole token.** The C++ reads the count and the
//! coordinates with `operator>>`, which stops at the first character that
//! cannot continue the number, so `3abc` reads as 3 and `0.5x` as 0.5. Here
//! a field is a whitespace-separated token and must parse whole, so both are
//! refused. A coordinate must also be finite: `nan`, `inf` and a literal
//! past `f64`'s range, which `operator>>` fails on, are refused too.

#![forbid(unsafe_code)]

use std::path::Path;

use fb_atom::{Atom, Element};
use fb_core_constants::BOHR_PER_ANGSTROM;
use fb_core_error_handling::{Error, Result};
use fb_core_types::Double3;
use fb_molecule::Molecule;

/// Parse the XYZ document `text` into a [`Molecule`] of `charge` and
/// `multiplicity`, its positions converted from Angstrom to bohr:
/// `parse_xyz`.
///
/// Every refusal is an [`ErrorKind::Io`](fb_core_error_handling::ErrorKind::Io)
/// error (the crate doc lists them), so a [`Molecule`], which aborts on no
/// atoms, is built only from a count of at least one that the atom lines
/// match.
pub fn parse_xyz(text: &str, charge: i32, multiplicity: i32) -> Result<Molecule> {
    // A trailing '\n' leaves an empty last line, as the C++ split does; a
    // blank line is skipped below rather than counted.
    let lines: Vec<&str> = text
        .split('\n')
        .map(|line| line.strip_suffix('\r').unwrap_or(line))
        .collect();
    if lines.len() < 2 {
        return Err(Error::io(
            "XYZ input is too short: expected a count line and a comment line",
        ));
    }

    let declared = lines[0]
        .split_whitespace()
        .next()
        .and_then(|token| token.parse::<i32>().ok())
        .filter(|&count| count >= 1)
        .ok_or_else(|| {
            Error::io(format!(
                "XYZ line 1 is not a positive atom count: '{}'",
                lines[0]
            ))
        })?;

    let mut atoms = Vec::with_capacity(declared as usize);
    for (i, line) in lines.iter().enumerate().skip(2) {
        let mut fields = line.split_whitespace();
        let Some(symbol) = fields.next() else {
            continue; // a blank line is not an atom
        };
        let mut coordinate = || {
            fields
                .next()
                .and_then(|token| token.parse::<f64>().ok())
                .filter(|value| value.is_finite())
        };
        let (Some(x), Some(y), Some(z)) = (coordinate(), coordinate(), coordinate()) else {
            return Err(Error::io(format!(
                "XYZ atom line {} is malformed: '{line}'",
                i + 1
            )));
        };
        let element = Element::from_symbol(symbol);
        if element == Element::Sentinel {
            return Err(Error::io(format!(
                "XYZ atom line {} has unknown element '{symbol}'",
                i + 1
            )));
        }
        atoms.push(Atom {
            position: Double3 {
                x: x * BOHR_PER_ANGSTROM,
                y: y * BOHR_PER_ANGSTROM,
                z: z * BOHR_PER_ANGSTROM,
            },
            element,
        });
    }

    if atoms.len() != declared as usize {
        return Err(Error::io(format!(
            "XYZ declares {declared} atoms but {} were found",
            atoms.len()
        )));
    }
    Ok(Molecule::new(atoms, charge, multiplicity))
}

/// Read the XYZ file at `path` into a [`Molecule`] through [`parse_xyz`]:
/// `read_xyz`. A file that cannot be read is an
/// [`ErrorKind::Io`](fb_core_error_handling::ErrorKind::Io) error naming it.
pub fn read_xyz(path: &Path, charge: i32, multiplicity: i32) -> Result<Molecule> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::io(format!("cannot open '{}': {e}", path.display())))?;
    parse_xyz(&text, charge, multiplicity)
}
