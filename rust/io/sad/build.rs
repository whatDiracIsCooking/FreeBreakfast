//! Generate the table of compiled-in SAD stamps, `$OUT_DIR/builtin_sad.rs`:
//! one `include_bytes!` per `data/sad/<basis>/<Z>.npy`, as
//! `src/io/sad/CMakeLists.txt` generates `builtin_sad.cpp` at configure time.
//!
//! The same files, in a fixed order: every `*.npy` one directory below
//! `data/sad/`, the directories and the files each sorted by name. A stem that
//! is not a three-digit Z fails the build, as it fails the C++ configure. The
//! other configure check, that each directory names a `data/basis` set, needs
//! the documents' JSON, so it is the crate's unit test
//! `every_directory_names_a_compiled_in_basis` instead. An empty `data/sad/`
//! is allowed: a binary that computes every density.
//!
//! Cargo rescans a directory named in `rerun-if-changed` whole, so a stamp
//! added, removed or rewritten reruns this script.

use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};

fn sorted_entries(dir: &Path) -> Vec<PathBuf> {
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|entry| entry.unwrap().path())
        .collect();
    entries.sort();
    entries
}

fn main() {
    let sad_dir = Path::new(&std::env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("../../../data/sad")
        .canonicalize()
        .expect("data/sad");
    println!("cargo:rerun-if-changed={}", sad_dir.display());
    println!("cargo:rerun-if-changed=build.rs");

    let mut entries = String::new();
    let mut count = 0_usize;
    for basis_dir in sorted_entries(&sad_dir).iter().filter(|p| p.is_dir()) {
        let basis = basis_dir.file_name().unwrap().to_str().unwrap();
        for file in sorted_entries(basis_dir)
            .iter()
            .filter(|p| p.extension().is_some_and(|e| e == "npy"))
        {
            let stem = file.file_stem().unwrap().to_str().unwrap();
            assert!(
                stem.len() == 3 && stem.bytes().all(|b| b.is_ascii_digit()),
                "{}: expected a three-digit Z",
                file.display()
            );
            let z: i32 = stem.parse().unwrap();
            writeln!(
                entries,
                "    ({basis:?}, {z}, include_bytes!({:?})),",
                file.display().to_string()
            )
            .unwrap();
            count += 1;
        }
    }
    let table = format!(
        "/// The stamps compiled into this crate, from `data/sad/`: `kBuiltinSad`.\n\
         const BUILTIN: [(&str, i32, &[u8]); {count}] = [\n{entries}];\n"
    );
    let out = Path::new(&std::env::var("OUT_DIR").unwrap()).join("builtin_sad.rs");
    fs::write(&out, table).unwrap_or_else(|e| panic!("{}: {e}", out.display()));
}
