//! The panic hook (`docs/diagnostics.md`, "Panics and profiles"): a panic is
//! a critical line in the file sink, located at the panic, flushed before
//! an abort; and a `#[should_panic]` test still passes through it.
//!
//! A binary of its own because the hook and the file sink are
//! process-global. Each case re-runs this binary as a child (selected by
//! `FB_PANIC_CHILD`, with the sink's path in `FB_PANIC_SINK`) and reads the
//! sink, because what it checks is what survives the process ending.
//!
//! A test build unwinds, so the abort case aborts the way a release build
//! does, without the release profile: its panic leaves an `extern "C" fn`,
//! which cannot unwind, so the runtime aborts after the hook returns, as
//! under `panic = "abort"`, and runs no exit handlers. (The runtime's own
//! "panic in a function that cannot unwind" goes through the hook too, as a
//! second critical line.)

#![forbid(unsafe_code)]

use std::os::unix::process::ExitStatusExt;
use std::path::PathBuf;
use std::process::{Command, ExitStatus};

use fb_core_log as log;

/// The line `needle` is on in this file, 1-based.
fn line_of(needle: &str) -> u32 {
    include_str!("panic_hook.rs")
        .lines()
        .position(|l| l.contains(needle) && !l.contains("line_of("))
        .expect("needle not in this file") as u32
        + 1
}

/// The sink a child writes, when it is one.
fn child_sink() -> Option<PathBuf> {
    std::env::var_os("FB_PANIC_SINK").map(PathBuf::from)
}

/// Panic out of a function that cannot unwind: the runtime aborts once
/// the hook has run, as a release build does for any panic.
extern "C" fn panic_where_no_unwind_can_go() {
    panic!("the worker lost its device"); // MARK abort
}

/// What a child does, by `FB_PANIC_CHILD`; nothing when unset.
fn child() {
    if std::env::var("FB_PANIC_CHILD").as_deref() == Ok("abort") {
        log::add_file_sink(child_sink().unwrap()).unwrap();
        ::log::info!("buffered before the panic");
        panic_where_no_unwind_can_go();
    }
}

#[test]
fn child_entry() {
    child();
}

/// Run the test `name` alone in a fresh process with `FB_PANIC_CHILD=what`,
/// and return how it ended and what its file sink holds.
/// `RUST_BACKTRACE` is set for the child: a test build unwinds, so std's
/// hook, which follows ours, prints the backtrace when asked.
fn run_child(name: &str, what: &str) -> (ExitStatus, String, String) {
    let dir = std::env::temp_dir().join(format!(
        "fb-core-log-panic-hook-{}-{what}",
        std::process::id()
    ));
    let sink = dir.join("run.log");
    let _ = std::fs::remove_dir_all(&dir);
    let out = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", name, "--nocapture", "--test-threads=1"])
        .env("FB_PANIC_CHILD", what)
        .env("FB_PANIC_SINK", &sink)
        .env("RUST_BACKTRACE", "1")
        .output()
        .unwrap();
    let text = std::fs::read_to_string(&sink).unwrap_or_default();
    let _ = std::fs::remove_dir_all(dir);
    let output = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    (out.status, text, output)
}

#[test]
fn a_panic_is_a_critical_line_in_the_file_sink_before_the_abort() {
    let (status, sink, output) = run_child("child_entry", "abort");
    assert_eq!(status.signal(), Some(libc::SIGABRT), "{status:?}\n{output}");
    let want = format!(
        "[critical] [panic_hook.rs:{}] thread 'child_entry' panicked: the worker lost its device",
        line_of("MARK abort")
    );
    // The abort ran no exit handler, so the line is there because the hook
    // flushed it, and the buffered line before it with it.
    assert!(sink.contains(&want), "{sink}\n  wanted {want}\n{output}");
    assert!(sink.contains("buffered before the panic"), "{sink}");
    assert!(
        output.contains("stack backtrace:") && output.contains("panic_where_no_unwind_can_go"),
        "{output}"
    );
}

/// Trips a check as a contract test does, through the hook. Run by the
/// harness, it is a plain `#[should_panic]` test; run as a child, its panic
/// is also written to the child's sink.
#[test]
#[should_panic(expected = "a basis has shells")]
fn a_should_panic_test_through_the_hook() {
    log::init();
    if let Some(sink) = child_sink() {
        log::add_file_sink(sink).unwrap();
    }
    panic!("a basis has shells"); // MARK should_panic
}

#[test]
fn the_hook_leaves_should_panic_tests_passing() {
    let (status, sink, output) = run_child("a_should_panic_test_through_the_hook", "should_panic");
    assert!(status.success(), "{status:?}\n{output}");
    assert!(output.contains("1 passed"), "{output}");
    let want = format!(
        "[critical] [panic_hook.rs:{}] thread 'a_should_panic_test_through_the_hook' panicked: a basis has shells",
        line_of("MARK should_panic")
    );
    assert!(sink.contains(&want), "{sink}\n  wanted {want}");
}
