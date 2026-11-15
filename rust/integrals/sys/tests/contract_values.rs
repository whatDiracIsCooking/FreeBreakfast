//! A tripped always-on contract's panic carries its operands' values, and
//! the location of the call that broke it (`docs/testing.md`, "Contracts
//! are part of the spec"; `docs/diagnostics.md`, "Panics and profiles").
//!
//! [`AuxTable::new`]'s contract is written with `fits_i32!`, so the panic
//! hook ends its line with the square it recomputed, and the function is
//! `#[track_caller]`, so the line is located at the call here, not in the
//! crate. A `#[should_panic]` test matches the contract's message alone, so
//! the parent re-runs it as a child at level off, where the hook writes its
//! one line to stderr, and reads that line. Host-only: no device is touched.

#![forbid(unsafe_code)]

use std::process::Command;

use fb_core_log as log;
use fb_integrals_sys::AuxTable;

/// The line `needle` is on in this file, 1-based.
fn line_of(needle: &str) -> u32 {
    include_str!("contract_values.rs")
        .lines()
        .position(|l| l.contains(needle) && !l.contains("line_of("))
        .expect("needle not in this file") as u32
        + 1
}

/// i32's limit is 2,147,483,647, and 46,341² = 2,147,488,281 passes it. Run
/// by the harness, a plain `#[should_panic]` test; run as a child
/// (`FB_CONTRACT_CHILD` set), at level off, so the hook's line goes to
/// stderr.
#[test]
#[should_panic(expected = "Pre-condition of new violated: n_cart² fits the kernels' 32-bit index")]
fn a_metric_past_the_index_panics() {
    if std::env::var_os("FB_CONTRACT_CHILD").is_some() {
        log::set_level(log::LogLevel::Off);
    }
    AuxTable::new(46_341); // MARK new
}

#[test]
fn the_panic_carries_the_square_at_the_callers_line() {
    let name = "a_metric_past_the_index_panics";
    let out = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture", "--test-threads=1"])
        .env("FB_CONTRACT_CHILD", "1")
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{:?}\n{stdout}{stderr}", out.status);
    assert!(stdout.contains("1 passed"), "{stdout}");
    let want = format!(
        "thread '{name}' panicked at {}:{}:5: Pre-condition of new violated: n_cart² fits the \
         kernels' 32-bit index: fits_i32! (n_cart.checked_mul(n_cart)) \
         [n_cart.checked_mul(n_cart) = Some(2147488281)]",
        file!(),
        line_of("MARK new")
    );
    assert!(
        stderr.lines().any(|l| l == want),
        "{stderr}\n  wanted the line {want}"
    );
}
