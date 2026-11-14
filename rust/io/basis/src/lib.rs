//! `src/io/basis` (module `free_breakfast.io.basis`): a Basis Set Exchange
//! (BSE) JSON document read into a geometry-free [`BasisStamp`], and the
//! orbital sets of `data/basis/` and the density-fitting sets of
//! `data/aux_basis/` compiled in and resolved by name.
//!
//! The document's shape:
//!
//! ```text
//! { "elements": {
//!     "1": { "electron_shells": [ ... ] },   // keyed by Z, a decimal string
//!     "8": { "electron_shells": [ ... ] } } }
//! ```
//!
//! Each `electron_shells` block has `angular_momentum` (a list of integers,
//! `[0, 1]` for an sp block), `exponents` (decimal *strings*), `coefficients`
//! (columns of decimal strings, each matched one to one to the exponents)
//! and an optional `function_type`. Momentum `i` takes column `i` (the sp /
//! spd convention), so the two lists have one length, except that a single
//! momentum with several columns is a general contraction, each column a
//! shell of that momentum. Everything stays in atomic units; nothing is
//! converted.
//!
//! **Only spherical shells are read**, because every [`ShellStamp`] is
//! spherical. `function_type` is BSE's one statement of the harmonic type:
//! `gto_spherical` is read at any `l`; `gto_cartesian` refuses at any `l`;
//! bare `gto`, or no `function_type`, is read for `l <= 1`, where the two
//! conventions coincide, and refuses from d up as ambiguous; `sto` or
//! anything else refuses. A refusal anywhere refuses the whole document, so
//! one Cartesian shell makes a set unreadable, as in the C++.
//!
//! | C++ (`free_breakfast`) | Here |
//! |---|---|
//! | `parse_basis` | [`parse_basis`] |
//! | `read_basis` | [`read_basis`] |
//! | `is_builtin_basis` | [`is_builtin_basis`] |
//! | `resolve_basis` | [`resolve_basis`] |
//! | `parse_and_assign_basis(text, molecule)` | nothing: `parse_basis(text)?.place(&molecule)` |
//! | `is_builtin_aux_basis` | [`is_builtin_aux_basis`] |
//! | `resolve_aux_basis` | [`resolve_aux_basis`] |
//! | `builtin_basis.cpp.in` and its CMake generation | the tables `BUILTIN` and `BUILTIN_AUX`, each document `include_str!`'d from `data/basis/` or `data/aux_basis/` |
//! | `io/base`'s `read_file`, `io/json`'s `parse_json` | [`std::fs::read_to_string`] and [`serde_json`] |
//!
//! **Errors.** JSON that does not parse, and a file that cannot be read,
//! are [`ErrorKind::Io`]. A document
//! that parses but says something this reader cannot take (a missing field,
//! one of the wrong JSON type, a decimal string that is not a number, a
//! Cartesian or ambiguous shell) is
//! [`ErrorKind::InvalidConfig`].
//! An element the set lacks is refused by [`BasisStamp::place`], as
//! `InvalidConfig` naming its Z.
//!
//! **A negative angular momentum is refused.** The C++ passes it to the
//! `ShellStamp` constructor, which aborts; here it is `InvalidConfig` like
//! every other bad shell.
//!
//! **A decimal string must be one whole number.** It is parsed by
//! [`str::parse`], correctly rounded as `std::from_chars` is, so each
//! exponent and coefficient is the C++ double. A leading `+`, which
//! `from_chars` refuses and `str::parse` would take, is refused, and so is
//! one on an element key.
//!
//! **The tables name their sets, and a test pins them to the documents.**
//! The C++ build reads each document's `name` into its table; here the names
//! are written beside the `include_str!`, and a unit test checks each
//! against its document's `name` and that every `data/basis/*.json` and
//! `data/aux_basis/*.json` is in its table.
//!
//! **The fitting sets are a table of their own, and compiled in.** Every
//! orbital set is one the SAD guess carries stamps for (`data/sad/`), and a
//! SAD density in a fitting basis means nothing, so [`resolve_basis`] never
//! sees a fitting set and [`resolve_aux_basis`] never sees an orbital one.
//! They are compiled in, about 16 MB of them, for the reason the orbital
//! sets are and the C++ gives: this tree installs nothing, so a data
//! directory read at run time would need an install rule and a search path
//! that can work on one machine and not the next, while a name that resolves
//! inside the binary cannot be misplaced. The path form keeps a set the
//! build does not carry a file away, with no rebuild.

#![forbid(unsafe_code)]

use std::collections::BTreeMap;
use std::path::Path;

use fb_basis::{BasisStamp, ShellStamp};
use fb_core_error_handling::{Error, ErrorKind, Result, Status};
use serde_json::Value;

/// The orbital basis sets compiled into this crate, from `data/basis/`:
/// each set's canonical BSE name, and its document's bytes verbatim.
/// `kBuiltinBases`. Reached through [`resolve_basis`] alone, which takes a
/// path too.
const BUILTIN: [(&str, &str); 11] = [
    ("6-31G", include_str!("../../../../data/basis/6-31g.json")),
    (
        "6-31G*",
        include_str!("../../../../data/basis/6-31gstar.json"),
    ),
    (
        "6-31G**",
        include_str!("../../../../data/basis/6-31gstarstar.json"),
    ),
    (
        "aug-cc-pVDZ",
        include_str!("../../../../data/basis/aug-cc-pVDZ.json"),
    ),
    (
        "aug-cc-pVTZ",
        include_str!("../../../../data/basis/aug-cc-pVTZ.json"),
    ),
    (
        "cc-pVDZ",
        include_str!("../../../../data/basis/cc-pVDZ.json"),
    ),
    (
        "cc-pVTZ",
        include_str!("../../../../data/basis/cc-pVTZ.json"),
    ),
    (
        "def2-SVP",
        include_str!("../../../../data/basis/def2-SVP.json"),
    ),
    (
        "def2-TZVP",
        include_str!("../../../../data/basis/def2-TZVP.json"),
    ),
    ("MINAO", include_str!("../../../../data/basis/minao.json")),
    ("STO-3G", include_str!("../../../../data/basis/sto-3g.json")),
];

/// The density-fitting (auxiliary) basis sets compiled into this crate,
/// from `data/aux_basis/`, in [`BUILTIN`]'s shape: `kBuiltinAuxBases`.
/// Reached through [`resolve_aux_basis`] alone; disjoint from [`BUILTIN`].
const BUILTIN_AUX: [(&str, &str); 13] = [
    (
        "6-31G**-acCD-6-6",
        include_str!("../../../../data/aux_basis/6-31gstarstar-acCD-6-6.json"),
    ),
    (
        "aug-cc-pVDZ-JKFIT",
        include_str!("../../../../data/aux_basis/aug-cc-pVDZ-JKFIT.json"),
    ),
    (
        "aug-cc-pVDZ-acCD-6-6",
        include_str!("../../../../data/aux_basis/aug-cc-pVDZ-acCD-6-6.json"),
    ),
    (
        "aug-cc-pVTZ-JKFIT",
        include_str!("../../../../data/aux_basis/aug-cc-pVTZ-JKFIT.json"),
    ),
    (
        "aug-cc-pVTZ-acCD-6-6",
        include_str!("../../../../data/aux_basis/aug-cc-pVTZ-acCD-6-6.json"),
    ),
    (
        "cc-pVDZ-JKFIT",
        include_str!("../../../../data/aux_basis/cc-pVDZ-JKFIT.json"),
    ),
    (
        "cc-pVDZ-acCD",
        include_str!("../../../../data/aux_basis/cc-pVDZ-acCD.json"),
    ),
    (
        "cc-pVDZ-acCD-6-6",
        include_str!("../../../../data/aux_basis/cc-pVDZ-acCD-6-6.json"),
    ),
    (
        "cc-pVTZ-JKFIT",
        include_str!("../../../../data/aux_basis/cc-pVTZ-JKFIT.json"),
    ),
    (
        "cc-pVTZ-acCD-6-6",
        include_str!("../../../../data/aux_basis/cc-pVTZ-acCD-6-6.json"),
    ),
    (
        "def2-SVP-acCD-6-6",
        include_str!("../../../../data/aux_basis/def2-SVP-acCD-6-6.json"),
    ),
    (
        "def2-TZVP-acCD-6-6",
        include_str!("../../../../data/aux_basis/def2-TZVP-acCD-6-6.json"),
    ),
    (
        "def2-universal-JKFIT",
        include_str!("../../../../data/aux_basis/def2-universal-JKFIT.json"),
    ),
];

/// Parse the BSE JSON document `text` into a geometry-free [`BasisStamp`],
/// unnamed: `parse_basis`.
///
/// JSON that does not parse is an [`ErrorKind::Io`] error; a document that
/// parses but cannot be read as a spherical basis is
/// [`ErrorKind::InvalidConfig`] (the crate doc has both lists).
pub fn parse_basis(text: &str) -> Result<BasisStamp> {
    let document: Value =
        serde_json::from_str(text).map_err(|e| Error::io(format!("malformed JSON: {e}")))?;
    let Value::Object(document) = document else {
        return Err(Error::config("BSE document is not a JSON object"));
    };
    let Some(elements) = document.get("elements") else {
        return Err(Error::config("BSE document has no 'elements'"));
    };
    let Value::Object(elements) = elements else {
        return Err(Error::config("BSE 'elements' is not an object"));
    };

    let mut table = BTreeMap::new();
    for (key, element) in elements {
        let z = key
            .parse::<i32>()
            .ok()
            .filter(|&z| z > 0 && !key.starts_with('+'))
            .ok_or_else(|| {
                Error::config(format!("BSE element key '{key}' is not an atomic number"))
            })?;
        let Value::Object(element) = element else {
            return Err(Error::config(format!(
                "BSE element '{key}' is not an object"
            )));
        };
        let Some(blocks) = element.get("electron_shells") else {
            return Err(Error::config(format!(
                "BSE element '{key}' has no electron_shells"
            )));
        };
        let Value::Array(blocks) = blocks else {
            return Err(Error::config(format!(
                "BSE element '{key}' electron_shells is not an array"
            )));
        };
        let mut shells = Vec::new();
        for block in blocks {
            shells.extend(shells_from_block(block)?);
        }
        table.insert(z, shells);
    }
    Ok(BasisStamp::new(table, ""))
}

/// Read the BSE JSON file at `path` through [`parse_basis`], unnamed:
/// `read_basis`. A file that cannot be read is an [`ErrorKind::Io`] error
/// naming it.
pub fn read_basis(path: &Path) -> Result<BasisStamp> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::io(format!("cannot open '{}': {e}", path.display())))?;
    parse_basis(&text)
}

/// Whether `spec` names an orbital set compiled into this crate, never a
/// fitting one: `is_builtin_basis`.
///
/// The spelling is folded first, lowercased with `-`, `_` and spaces
/// dropped, so `STO-3G`, `sto3g` and `STO_3G` are one name, and `6-31G*`
/// stays apart from `6-31G`. A front end asks this before it resolves a
/// relative path, since a name must not be joined to a config file's
/// directory; [`resolve_basis`] asks it again itself.
pub fn is_builtin_basis(spec: &str) -> bool {
    find_builtin(&BUILTIN, spec).is_some()
}

/// The basis `spec` asks for, by compiled-in name or else by path:
/// `resolve_basis`.
///
/// A name ([`is_builtin_basis`]) wins, so a file called `sto-3g` is reached
/// as `./sto-3g`. A compiled-in set comes back named with its canonical
/// spelling, whatever spelling asked; a file comes back unnamed, whatever its
/// document calls itself, since only the compiled-in bytes are known to be
/// the set a name promises. A spec that is neither is [`read_basis`]'s
/// [`ErrorKind::Io`] error, its origin kept, with the compiled-in names
/// appended, since a misspelt name is the likelier mistake.
pub fn resolve_basis(spec: &str) -> Result<BasisStamp> {
    resolve_from(&BUILTIN, spec)
}

/// Whether `spec` names a density-fitting set compiled into this crate:
/// `is_builtin_aux_basis`. [`is_builtin_basis`] over the fitting sets of
/// `data/aux_basis/` instead of the orbital ones, the spelling folded the
/// same way; the question a front end asks before it resolves a relative
/// aux-basis path.
///
/// ```
/// use fb_io_basis::{is_builtin_aux_basis, is_builtin_basis};
///
/// assert!(is_builtin_aux_basis("cc_pvdz_jkfit"));
/// assert!(!is_builtin_aux_basis("cc-pvdz"));
/// assert!(!is_builtin_basis("cc-pvdz-jkfit"));
/// ```
pub fn is_builtin_aux_basis(spec: &str) -> bool {
    find_builtin(&BUILTIN_AUX, spec).is_some()
}

/// The density-fitting (auxiliary) basis `spec` asks for, by compiled-in
/// name or else by path: `resolve_aux_basis`.
///
/// [`resolve_basis`]'s rules exactly (name first, spelling folded, a path
/// otherwise, only a compiled-in set named, [`ErrorKind::Io`] for neither,
/// with the fitting sets' names appended) over the sets of
/// `data/aux_basis/`. The caller places the stamp over the orbital basis's
/// molecule, `stamp.primitive_normalized().place(&molecule)`, where an
/// element the set lacks is [`ErrorKind::InvalidConfig`] naming its Z.
/// Fitting sets run past the orbital ceiling (cc-pVTZ-JKFIT has g on O, the
/// acCD-6-6 sets reach i); holding those shells is supported, and a kernel
/// that cannot take one refuses it itself.
///
/// ```
/// let stamp = fb_io_basis::resolve_aux_basis("ccpvdzjkfit").unwrap();
/// assert_eq!(stamp.name(), "cc-pVDZ-JKFIT");
/// assert!(stamp.shells_for(8).is_some());
/// ```
pub fn resolve_aux_basis(spec: &str) -> Result<BasisStamp> {
    resolve_from(&BUILTIN_AUX, spec)
}

/// `spec` from `table` by name, else read from disk as a path:
/// `resolve_from`, what [`resolve_basis`] and [`resolve_aux_basis`] share.
fn resolve_from(table: &[(&'static str, &'static str)], spec: &str) -> Result<BasisStamp> {
    if let Some((name, json)) = find_builtin(table, spec) {
        let stamp = parse_basis(json)?;
        return Ok(BasisStamp::new(stamp.element_shells().clone(), name));
    }
    read_basis(Path::new(spec)).map_err(|mut error| {
        if error.kind == ErrorKind::Io {
            let names: Vec<&str> = table.iter().map(|(name, _)| *name).collect();
            error.message += &format!(" (and not a compiled-in basis: {})", names.join(", "));
        }
        error
    })
}

/// `spec` lowercased with `-`, `_` and spaces dropped:
/// `normalized_basis_name`.
fn normalized(spec: &str) -> String {
    spec.chars()
        .filter(|c| !matches!(c, '-' | '_' | ' '))
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

/// The entry of `table` that `spec` names: `find_builtin_basis`.
fn find_builtin(
    table: &[(&'static str, &'static str)],
    spec: &str,
) -> Option<(&'static str, &'static str)> {
    let key = normalized(spec);
    table
        .iter()
        .copied()
        .find(|(name, _)| normalized(name) == key)
}

/// One BSE decimal string, `field`'s, as a double: `parse_decimal`.
fn parse_decimal(node: &Value, field: &str) -> Result<f64> {
    let Value::String(text) = node else {
        return Err(Error::config(format!(
            "BSE '{field}' entry is not a string"
        )));
    };
    text.parse::<f64>()
        .ok()
        .filter(|_| !text.starts_with('+'))
        .ok_or_else(|| Error::config(format!("BSE '{field}' entry '{text}' is not a number")))
}

/// Admit a shell of momentum `l` from `block` only if its `function_type`
/// says it is spherical (the crate doc has the table): `check_spherical`.
fn check_spherical(block: &serde_json::Map<String, Value>, l: i32) -> Status {
    let ambiguous = || {
        Err(Error::config(format!(
            "BSE shell of angular_momentum={l} >= 2 has no function_type or a bare \"gto\", \
             ambiguous between Cartesian and spherical; only \"gto_spherical\" is supported"
        )))
    };
    match block.get("function_type") {
        None if l >= 2 => ambiguous(),
        None => Ok(()),
        Some(Value::String(t)) if t == "gto_spherical" => Ok(()),
        Some(Value::String(t)) if t == "gto_cartesian" => Err(Error::config(format!(
            "BSE shell function_type \"gto_cartesian\" (angular_momentum={l}): Cartesian \
             basis functions are not supported; only \"gto_spherical\" is"
        ))),
        Some(Value::String(t)) if t == "gto" && l >= 2 => ambiguous(),
        Some(Value::String(t)) if t == "gto" => Ok(()),
        Some(Value::String(t)) => Err(Error::config(format!(
            "BSE shell function_type \"{t}\" is not a supported Gaussian type"
        ))),
        Some(_) => Err(Error::config("BSE shell function_type is not a string")),
    }
}

/// One `electron_shells` block as its shells, one per coefficient column:
/// `shells_from_block`. Every shape [`ShellStamp::zipped`] aborts on is
/// refused first.
fn shells_from_block(block: &Value) -> Result<Vec<ShellStamp>> {
    let Value::Object(block) = block else {
        return Err(Error::config("BSE electron_shells entry is not an object"));
    };
    let (Some(momenta), Some(exponents), Some(columns)) = (
        block.get("angular_momentum"),
        block.get("exponents"),
        block.get("coefficients"),
    ) else {
        return Err(Error::config(
            "BSE shell is missing angular_momentum, exponents or coefficients",
        ));
    };
    let (Value::Array(momenta), Value::Array(exponents), Value::Array(columns)) =
        (momenta, exponents, columns)
    else {
        return Err(Error::config(
            "BSE shell angular_momentum, exponents and coefficients must be arrays",
        ));
    };

    let momenta = momenta
        .iter()
        .map(|l| {
            l.as_i64()
                .and_then(|l| i32::try_from(l).ok())
                .filter(|&l| l >= 0)
                .ok_or_else(|| {
                    Error::config(format!(
                        "BSE angular_momentum entry {l} is not a non-negative integer"
                    ))
                })
        })
        .collect::<Result<Vec<i32>>>()?;
    if momenta.is_empty() {
        return Err(Error::config(
            "BSE shell has an empty angular_momentum list",
        ));
    }
    let exponents = exponents
        .iter()
        .map(|e| parse_decimal(e, "exponents"))
        .collect::<Result<Vec<f64>>>()?;
    if exponents.is_empty() {
        return Err(Error::config("BSE shell has no exponents"));
    }
    if columns.is_empty() {
        return Err(Error::config("BSE shell has no coefficient columns"));
    }
    // One momentum per column (sp/spd), or one momentum over several
    // columns (a general contraction); anything else is ambiguous.
    let per_column = momenta.len() == columns.len();
    if !per_column && momenta.len() != 1 {
        return Err(Error::config(format!(
            "BSE shell has {} angular momenta but {} coefficient columns",
            momenta.len(),
            columns.len()
        )));
    }

    let mut shells = Vec::with_capacity(columns.len());
    for (col, column) in columns.iter().enumerate() {
        let Value::Array(column) = column else {
            return Err(Error::config("BSE coefficients column is not an array"));
        };
        if column.len() != exponents.len() {
            return Err(Error::config(format!(
                "BSE coefficients column has {} entries but there are {} exponents",
                column.len(),
                exponents.len()
            )));
        }
        let coefficients = column
            .iter()
            .map(|c| parse_decimal(c, "coefficients"))
            .collect::<Result<Vec<f64>>>()?;
        let l = if per_column { momenta[col] } else { momenta[0] };
        check_spherical(block, l)?;
        shells.push(ShellStamp::zipped(l, &exponents, &coefficients));
    }
    Ok(shells)
}

#[cfg(test)]
mod tests;
