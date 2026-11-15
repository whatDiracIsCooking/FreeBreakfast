//! The panic hook (`docs/diagnostics.md`, "Panics and profiles"): a panic is
//! a critical line in the file sink, located at the panic, flushed before
//! an abort; at level off, where that line is dropped, its message goes to
//! stderr instead; and a `#[should_panic]` test still passes through it.
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
//!
//! A contract written with the check macros (`fb_core_log::check`) ends the
//! hook's line with its operands' text and values, at level off on stderr as
//! at info in the sink, and a check that held leaves nothing for the next
//! panic: the `values` children, which catch each panic and go on.

#![forbid(unsafe_code)]

use std::os::unix::process::ExitStatusExt;
use std::panic::catch_unwind;
use std::path::PathBuf;
use std::process::{Command, ExitStatus};

use contracts::requires;
use fb_core_log as log;
use fb_core_log::{eq, fits_i32, le, lt};

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

#[requires(le!(count, data.len()), "count is within data")]
fn head(count: usize, data: &[f64]) -> &[f64] {
    &data[..count]
}

#[requires(lt!(i, n), "i is below n")]
fn below(i: usize, n: usize) -> usize {
    n - i
}

#[requires(eq!(dst.len(), src.len()), "the two slices are one length")]
fn copy(dst: &mut [f64], src: &[f64]) {
    dst.copy_from_slice(src);
}

#[requires(fits_i32!(n.checked_mul(n)), "n squared fits an i32")]
fn square(n: usize) -> usize {
    n * n
}

/// Trip each check, then pass one and panic plainly; every panic caught.
fn trip_each_check() {
    assert!(catch_unwind(|| head(5, &[0.0; 4]).len()).is_err());
    assert!(catch_unwind(|| below(3, 3)).is_err());
    assert!(catch_unwind(|| copy(&mut [0.0; 2], &[1.0; 3])).is_err());
    assert!(catch_unwind(|| square(46_341)).is_err());
    assert_eq!(head(1, &[2.0]), [2.0]);
    assert!(catch_unwind(|| panic!("a plain panic after a check that held")).is_err()); // MARK plain
}

/// What a child does, by `FB_PANIC_CHILD`; nothing when unset. `off`
/// panics as `abort` does, with every line silenced. `values` and
/// `values_off` trip each check macro's contract, at info into the sink
/// and at off.
fn child() {
    match std::env::var("FB_PANIC_CHILD").as_deref() {
        Ok("values") => {
            log::add_file_sink(child_sink().unwrap()).unwrap();
            trip_each_check();
        }
        Ok("values_off") => {
            log::set_level(log::LogLevel::Off);
            trip_each_check();
        }
        Ok("abort") => {
            log::add_file_sink(child_sink().unwrap()).unwrap();
            ::log::info!("buffered before the panic");
            panic_where_no_unwind_can_go();
        }
        Ok("off") => {
            log::set_level(log::LogLevel::Off);
            log::add_file_sink(child_sink().unwrap()).unwrap();
            panic_where_no_unwind_can_go();
        }
        _ => {}
    }
}

#[test]
fn child_entry() {
    child();
}

/// Run the test `name` alone in a fresh process with `FB_PANIC_CHILD=what`,
/// and return how it ended, what its file sink holds, its stdout and stderr
/// together, and its stderr alone.
/// `RUST_BACKTRACE` is set for the child: a test build unwinds, so std's
/// hook, which follows ours, prints the backtrace when asked.
fn run_child(name: &str, what: &str) -> (ExitStatus, String, String, String) {
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
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let output = format!("{}{stderr}", String::from_utf8_lossy(&out.stdout));
    (out.status, text, output, stderr)
}

#[test]
fn a_panic_is_a_critical_line_in_the_file_sink_before_the_abort() {
    let (status, sink, output, stderr) = run_child("child_entry", "abort");
    assert_eq!(status.signal(), Some(libc::SIGABRT), "{status:?}\n{output}");
    let want = format!(
        "[critical] [panic_hook.rs:{}] thread 'child_entry' panicked: the worker lost its device",
        line_of("MARK abort")
    );
    // The abort ran no exit handler, so the line is there because the hook
    // flushed it, and the buffered line before it with it.
    assert_eq!(
        sink.matches(&want).count(),
        1,
        "{sink}\n  wanted {want}\n{output}"
    );
    assert!(sink.contains("buffered before the panic"), "{sink}");
    // The line was written, so the hook adds no stderr copy of it. std's
    // hook, which follows ours in a test build, writes its own, with the
    // message on the next line.
    let stderr_line = format!(
        "panic_hook.rs:{}:5: the worker lost its device",
        line_of("MARK abort")
    );
    assert!(!stderr.contains(&stderr_line), "{stderr}");
    assert!(
        output.contains("stack backtrace:") && output.contains("panic_where_no_unwind_can_go"),
        "{output}"
    );
}

#[test]
fn at_level_off_a_panic_is_still_written_to_stderr() {
    let (status, sink, output, stderr) = run_child("child_entry", "off");
    assert_eq!(status.signal(), Some(libc::SIGABRT), "{status:?}\n{output}");
    assert!(!sink.contains("[critical]"), "{sink}");
    // One line, unlike std's hook, which puts the message on the next.
    let want = format!(
        "thread 'child_entry' panicked at core/log/tests/panic_hook.rs:{}:5: the worker lost its device",
        line_of("MARK abort")
    );
    assert!(stderr.contains(&want), "{stderr}\n  wanted {want}");
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
    let (status, sink, output, _) =
        run_child("a_should_panic_test_through_the_hook", "should_panic");
    assert!(status.success(), "{status:?}\n{output}");
    assert!(output.contains("1 passed"), "{output}");
    let want = format!(
        "[critical] [panic_hook.rs:{}] thread 'a_should_panic_test_through_the_hook' panicked: a basis has shells",
        line_of("MARK should_panic")
    );
    assert!(sink.contains(&want), "{sink}\n  wanted {want}");
}

/// Each check macro's tripped contract, as the hook ends its line.
const TRIPPED: [&str; 4] = [
    "Pre-condition of head violated: count is within data: le! (count, data.len()) \
     [count = 5, data.len() = 4]",
    "Pre-condition of below violated: i is below n: lt! (i, n) [i = 3, n = 3]",
    "Pre-condition of copy violated: the two slices are one length: \
     eq! (dst.len(), src.len()) [dst.len() = 2, src.len() = 3]",
    "Pre-condition of square violated: n squared fits an i32: \
     fits_i32! (n.checked_mul(n)) [n.checked_mul(n) = Some(2147488281)]",
];

#[test]
fn a_tripped_check_ends_the_critical_line_with_its_operands() {
    let (status, sink, output, _) = run_child("child_entry", "values");
    assert!(status.success(), "{status:?}\n{output}");
    for want in TRIPPED {
        let line = format!("thread 'child_entry' panicked: {want}");
        assert_eq!(sink.matches(&line).count(), 1, "{sink}\n  wanted {line}");
    }
    // The check that held stored nothing, and the hook cleared the slot
    // after each panic, so the plain panic's line ends at its message.
    let plain = sink
        .lines()
        .find(|l| l.contains("a plain panic"))
        .unwrap_or_else(|| panic!("no plain panic in\n{sink}"));
    let want = format!(
        "[panic_hook.rs:{}] thread 'child_entry' panicked: a plain panic after a check that held",
        line_of("MARK plain")
    );
    assert!(plain.ends_with(&want), "{plain}\n  wanted {want}");
}

#[test]
fn at_level_off_a_tripped_check_has_its_operands_on_stderr() {
    let (status, sink, output, stderr) = run_child("child_entry", "values_off");
    assert!(status.success(), "{status:?}\n{output}");
    assert!(!sink.contains("[critical]"), "{sink}");
    for want in TRIPPED {
        let line = stderr
            .lines()
            .find(|l| l.starts_with("thread 'child_entry' panicked at ") && l.ends_with(want));
        assert!(line.is_some(), "{stderr}\n  wanted a line ending {want}");
    }
    let plain = stderr
        .lines()
        .find(|l| l.starts_with("thread 'child_entry' panicked at ") && l.contains("a plain panic"))
        .unwrap_or_else(|| panic!("no plain panic in\n{stderr}"));
    assert!(
        plain.ends_with(": a plain panic after a check that held"),
        "{plain}"
    );
}

#[requires(fits_i32!(n.checked_mul(n)), "n squared fits an i32")]
fn square_u32(n: u32) -> u32 {
    n * n
}

/// i32's limit is 2^31 - 1 = 2,147,483,647: 46,340² = 2,147,395,600 fits
/// it and 46,341² = 2,147,488,281 does not; a product that wrapped, `None`,
/// never fits.
#[test]
fn fits_i32_holds_at_i32s_limit() {
    log::init();
    assert_eq!(square(46_340), 2_147_395_600);
    assert!(catch_unwind(|| square(46_341)).is_err());
    assert_eq!(square_u32(46_340), 2_147_395_600);
    assert!(catch_unwind(|| square_u32(65_536)).is_err());
}
