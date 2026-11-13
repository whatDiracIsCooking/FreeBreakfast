//! The death tests of `test/basis/`: `BasisShellDeathTest`,
//! `BasisShellStampDeathTest` and the two `BasisSetDeathTest` cases. Each
//! abort lands at the caller with a critical line.
//!
//! A binary of its own, because each case re-runs it as a child (selected by
//! `FB_BASIS_CHILD`), as `molecule/tests/molecule_abort.rs` does: what it
//! checks, the line flushed and the abort, is what survives the process
//! ending. The C++ partition case is under `FBF_SKIP_IF_FAST_TESTS()`; six
//! child processes here take milliseconds, so it always runs.

use std::os::unix::process::ExitStatusExt;
use std::process::Command;

use fb_basis::{BasisSet, Primitive, Shell, ShellStamp};
use fb_core_types::Double3;

const UNIT: Primitive = Primitive {
    exponent: 1.0,
    coefficient: 1.0,
};

fn two_s_shells() -> Vec<Shell> {
    vec![
        Shell::new(0, vec![UNIT], Double3::default()),
        Shell::new(0, vec![UNIT], Double3::default()),
    ]
}

/// What the child does, by `FB_BASIS_CHILD`; nothing when unset.
#[test]
fn child_entry() {
    let Ok(case) = std::env::var("FB_BASIS_CHILD") else {
        return;
    };
    let origin = Double3::default();
    match case.as_str() {
        "shell_negative_l" => drop(Shell::new(-1, vec![UNIT], origin)),
        "shell_no_primitives" => drop(Shell::new(0, Vec::new(), origin)),
        "shell_count_mismatch" => drop(Shell::zipped(0, &[1.0, 2.0], &[1.0], origin)),
        "stamp_negative_l" => drop(ShellStamp::new(-1, vec![UNIT])),
        "stamp_no_primitives" => drop(ShellStamp::new(0, Vec::new())),
        "stamp_count_mismatch" => drop(ShellStamp::zipped(0, &[1.0, 2.0], &[1.0])),
        "no_trailing_total" => drop(BasisSet::with_atoms(two_s_shells(), vec![], "")),
        "not_from_zero" => drop(BasisSet::with_atoms(two_s_shells(), vec![1, 2], "")),
        "not_to_nshells" => drop(BasisSet::with_atoms(two_s_shells(), vec![0, 1], "")),
        "decreasing" => drop(BasisSet::with_atoms(two_s_shells(), vec![0, 2, 1, 2], "")),
        "atom_shell_offset" => {
            BasisSet::with_atoms(two_s_shells(), vec![0, 2], "").atom_shell_offset(1);
        }
        "atom_shell_count" => {
            BasisSet::with_atoms(two_s_shells(), vec![0, 2], "").atom_shell_count(1);
        }
        "ao_offset_cart" => {
            BasisSet::new(vec![Shell::new(0, vec![UNIT], origin)]).ao_offset_cart(1);
        }
        "ao_offset_sphe" => {
            BasisSet::new(vec![Shell::new(0, vec![UNIT], origin)]).ao_offset_sphe(1);
        }
        other => panic!("no child case {other}"),
    }
}

/// Re-runs this binary as the child `case`, and checks it aborted with a
/// critical line located in this file and carrying `message`.
#[track_caller]
fn expect_abort(case: &str, message: &str) {
    let out = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "child_entry", "--nocapture", "--test-threads=1"])
        .env("FB_BASIS_CHILD", case)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(
        out.status.signal(),
        Some(libc::SIGABRT),
        "{case}: {:?}",
        out.status
    );
    for needle in ["[critical] [basis_abort.rs:", message] {
        assert!(
            stdout.contains(needle),
            "{case}: {needle:?} not in {stdout}"
        );
    }
}

/// `BasisShellDeathTest.ShellAbortsOnBadShape`.
#[test]
fn shell_aborts_on_bad_shape() {
    expect_abort(
        "shell_negative_l",
        "Shell: angular momentum must be non-negative",
    );
    expect_abort(
        "shell_no_primitives",
        "Shell: a shell needs at least one primitive",
    );
    expect_abort(
        "shell_count_mismatch",
        "Shell: exponent and coefficient counts must match",
    );
}

/// `BasisShellStampDeathTest.AbortsOnBadShape`.
#[test]
fn shell_stamp_aborts_on_bad_shape() {
    expect_abort(
        "stamp_negative_l",
        "ShellStamp: angular momentum must be non-negative",
    );
    expect_abort(
        "stamp_no_primitives",
        "ShellStamp: a shell needs at least one primitive",
    );
    expect_abort(
        "stamp_count_mismatch",
        "ShellStamp: exponent and coefficient counts must match",
    );
}

/// `BasisSetDeathTest.AtomShellOffsetsAbortOnBadPartition`.
#[test]
fn atom_shell_offsets_abort_on_bad_partition() {
    let partition = "BasisSet: atom shell offsets do not partition the shells";
    for case in [
        "no_trailing_total",
        "not_from_zero",
        "not_to_nshells",
        "decreasing",
    ] {
        expect_abort(case, partition);
    }
    expect_abort(
        "atom_shell_offset",
        "BasisSet::atom_shell_offset: atom index out of range",
    );
    expect_abort(
        "atom_shell_count",
        "BasisSet::atom_shell_count: atom index out of range",
    );
}

/// `BasisSetDeathTest.AoOffsetAbortsOutOfRange`, with its in-range read.
#[test]
fn ao_offset_aborts_out_of_range() {
    let basis = BasisSet::new(vec![Shell::new(0, vec![UNIT], Double3::default())]);
    assert_eq!(basis.ao_offset_cart(0), 0);
    expect_abort(
        "ao_offset_cart",
        "BasisSet::ao_offset_cart: shell index out of range",
    );
    expect_abort(
        "ao_offset_sphe",
        "BasisSet::ao_offset_sphe: shell index out of range",
    );
}
