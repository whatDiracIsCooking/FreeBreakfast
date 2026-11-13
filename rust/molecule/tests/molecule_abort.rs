//! `MoleculeDeathTest.AbortsWithNoAtoms`, with no device: a molecule of no
//! atoms aborts at the caller with a critical line.
//!
//! A binary of its own, because the case re-runs it as a child (selected by
//! `FB_MOLECULE_CHILD`), as `core/arena/tests/arena_block.rs` does: what it
//! checks, the line flushed and the abort, is what survives the process
//! ending.

use std::os::unix::process::ExitStatusExt;
use std::process::Command;

use fb_molecule::Molecule;

/// What the child does, by `FB_MOLECULE_CHILD`; nothing when unset.
#[test]
fn child_entry() {
    if std::env::var("FB_MOLECULE_CHILD").as_deref() == Ok("no_atoms") {
        let _molecule = Molecule::new(Vec::new(), 0, 1); // MARK no_atoms
    }
}

#[test]
fn aborts_with_no_atoms() {
    let out = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "child_entry", "--nocapture", "--test-threads=1"])
        .env("FB_MOLECULE_CHILD", "no_atoms")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.signal(), Some(libc::SIGABRT), "{:?}", out.status);
    let line = include_str!("molecule_abort.rs")
        .lines()
        .position(|l| l.contains("MARK no_atoms") && !l.contains("contains("))
        .unwrap()
        + 1;
    for needle in [
        format!("[critical] [molecule_abort.rs:{line}]"),
        "Molecule: a molecule of 0 atoms".to_owned(),
    ] {
        assert!(stdout.contains(&needle), "{needle:?} not in {stdout}");
    }
}
