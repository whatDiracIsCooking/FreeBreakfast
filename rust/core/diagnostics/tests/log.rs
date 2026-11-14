//! `core::log`, `core::diagnostics` and the abort tier, through the real
//! sinks. A binary of its own because the logger is process-global: one
//! level, one file sink, one counter. The in-process checks are one test, so
//! nothing beside them moves the level.
//!
//! The abort and exit cases re-run this binary as a child (selected by
//! `FB_LOG_CHILD`) and read its stdout, because what they check is what
//! survives the process ending.

use std::panic::Location;
use std::process::Command;

use fb_core_diagnostics::{critical, debug, error, gpu_must, info, trace, warn};
use fb_core_error_handling::{Error, check_or_abort};
use fb_core_log::{self as log, LogLevel};
use fb_core_wwr::gpu_log_error_at;
use wwr_sys::wwr_status;

/// The line `needle` is on in this file, 1-based.
fn line_of(needle: &str) -> u32 {
    include_str!("log.rs")
        .lines()
        .position(|l| l.contains(needle) && !l.contains("line_of("))
        .expect("needle not in this file") as u32
        + 1
}

#[test]
fn levels_format_counter_and_file_sink() {
    let dir = std::env::temp_dir().join(format!("fb-core-diagnostics-log-{}", std::process::id()));
    let path = dir.join("nested/run.log");
    log::add_file_sink(&path).unwrap();
    log::add_file_sink(&path).unwrap(); // the same path: nothing
    log::set_level(LogLevel::Debug);
    log::reset_warning_count();

    assert!(log::should_log(LogLevel::Debug));
    assert!(!log::should_log(LogLevel::Trace));
    assert!(!log::should_log(LogLevel::Off));

    trace!("trace is filtered out");
    debug!("iteration {} residual {:.3e}", 3, 1.5e-7); // MARK debug
    info!("plain info"); // MARK info
    warn!("first warning"); // MARK warn
    critical!("critical through the macro"); // MARK critical
    log::add_file_sink(dir.join("other.log")).unwrap(); // MARK second sink
    gpu_log_error_at(wwr_status::WWR_STATUS_INVALID_VALUE, Location::caller()); // MARK destroy

    // A filtered warning reaches no sink, so it does not count.
    log::set_level(LogLevel::Error);
    warn!("filtered warning");
    log::set_level(LogLevel::Critical);
    error!("filtered error");
    log::set_level(LogLevel::Off);
    critical!("off silences critical too");
    log::set_level(LogLevel::Info);
    log::flush();

    // The sink's warning and the counted warn!.
    assert_eq!(log::warning_count(), 2);

    let text = std::fs::read_to_string(&path).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    assert_eq!(lines.len(), 6, "{text}");

    // [YYYY-MM-DD HH:MM:SS.mmm] [tid] [level] [file:line] message
    // SAFETY: gettid takes nothing and cannot fail.
    let tid = unsafe { libc::gettid() }.to_string();
    let expect = [
        (
            "debug",
            line_of("MARK debug"),
            "iteration 3 residual 1.500e-7",
        ),
        ("info", line_of("MARK info"), "plain info"),
        ("warning", line_of("MARK warn"), "first warning"),
        (
            "critical",
            line_of("MARK critical"),
            "critical through the macro",
        ),
        (
            "warning",
            line_of("MARK second sink"),
            "file sink already set to",
        ),
        (
            "error",
            line_of("MARK destroy"),
            "WWR_STATUS_INVALID_VALUE (invalid argument)",
        ),
    ];
    for (line, (level, number, message)) in lines.iter().zip(expect) {
        let (stamp, rest) = line.split_at(25);
        let b = stamp.as_bytes();
        assert!(
            b[0] == b'[' && b[5] == b'-' && b[11] == b' ' && b[20] == b'.' && b[24] == b']',
            "{line}"
        );
        let want = format!(" [{tid}] [{level}] [log.rs:{number}] ");
        assert!(rest.starts_with(&want), "{line}\n  wanted {want}");
        assert!(rest.contains(message), "{line}");
    }
    let _ = std::fs::remove_dir_all(dir);
}

/// What a child does, by `FB_LOG_CHILD`; nothing when unset.
fn child() {
    match std::env::var("FB_LOG_CHILD").as_deref() {
        Ok("exit") => {
            log::init();
            info!("buffered until exit");
            std::process::exit(0);
        }
        Ok("check_or_abort") => {
            log::init();
            info!("buffered before the abort");
            check_or_abort::<()>(Err(Error::numerical("singular matrix")));
        }
        Ok("gpu_must") => {
            log::init();
            gpu_must(wwr_status::WWR_STATUS_NO_DEVICE); // MARK gpu_must
        }
        _ => {}
    }
}

#[test]
fn child_entry() {
    child();
}

/// Run [`child_entry`] in a fresh process with `FB_LOG_CHILD=what`.
fn run_child(what: &str) -> (std::process::ExitStatus, String) {
    let out = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "child_entry", "--nocapture", "--test-threads=1"])
        .env("FB_LOG_CHILD", what)
        .output()
        .unwrap();
    (
        out.status,
        String::from_utf8_lossy(&out.stdout).into_owned(),
    )
}

#[test]
fn a_normal_exit_flushes_the_buffered_tail() {
    let (status, stdout) = run_child("exit");
    assert!(status.success());
    assert!(
        stdout.contains("[info] ") && stdout.contains("buffered until exit"),
        "{stdout}"
    );
}

#[test]
fn check_or_abort_logs_at_critical_and_aborts() {
    use std::os::unix::process::ExitStatusExt;
    let (status, stdout) = run_child("check_or_abort");
    assert_eq!(status.signal(), Some(libc::SIGABRT), "{status:?}");
    // The critical line flushes the buffered one ahead of it.
    assert!(stdout.contains("buffered before the abort"), "{stdout}");
    assert!(stdout.contains("[critical] [log.rs:"), "{stdout}");
    assert!(stdout.contains("singular matrix"), "{stdout}");
}

#[test]
fn gpu_must_aborts_at_the_caller() {
    use std::os::unix::process::ExitStatusExt;
    let (status, stdout) = run_child("gpu_must");
    assert_eq!(status.signal(), Some(libc::SIGABRT), "{status:?}");
    let want = format!(
        "[critical] [log.rs:{}] WWR_STATUS_NO_DEVICE (no GPU device is available)",
        line_of("MARK gpu_must")
    );
    assert!(stdout.contains(&want), "{stdout}\n  wanted {want}");
}
