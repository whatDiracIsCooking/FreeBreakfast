//! `src/core/log` (module `free_breakfast.core.log`): the logger behind the
//! [`log`](https://crates.io/crates/log) facade.
//!
//! Call sites use `log`'s macros (`log::info!`, `log::warn!`, ...), or the
//! same macros through [`crate::core::diagnostics`], which adds
//! [`critical!`](crate::critical). This module is the one backend they
//! reach, as spdlog is the one backend behind the C++ module, and nothing
//! outside it names how a line is written.
//!
//! | C++ (`free_breakfast::log`) | Here |
//! |---|---|
//! | `LogLevel` | [`LogLevel`], the same seven levels and values |
//! | `trace` / `debug` / `info` / `warn` / `error` | `log::trace!` ... `log::error!` |
//! | `critical` | [`critical!`](crate::critical), and [`log_at`] for a location already held |
//! | `set_level` / `should_log` / `flush` | [`set_level`] / [`should_log`] / [`flush`] |
//! | `add_file_sink` | [`add_file_sink`], which returns the open's failure |
//! | `warning_count` / `reset_warning_count` | [`warning_count`] / [`reset_warning_count`] |
//! | `get_logger()`'s first-use construction | [`init`], or any function here |
//!
//! Output goes to stdout, formatted
//! `[YYYY-MM-DD HH:MM:SS.mmm] [tid] [level] [file:line] message`, as the C++
//! pattern `[%Y-%m-%d %H:%M:%S.%e] [%t] [%l] [%s:%#] %v` writes it: local
//! time, the OS thread id (what gdb, `perf` and compute-sanitizer print, not
//! [`std::thread::ThreadId`]), and the file's base name. The level name is
//! coloured when stdout is a terminal, as spdlog's colour sink does; the file
//! sink never is. Error and critical flush every sink at once; lower levels
//! stay buffered until [`flush`], or the process exits.
//!
//! Three things differ from the C++, each because Rust does it for free or
//! cannot do it at all:
//!
//! - **No eager formatting.** `log::debug!("{x}")` checks the level before it
//!   formats anything, so the `FBF_LOG_*` rule (format only through the
//!   macros) is how every call already works.
//! - **The logger is installed, not built on first use.** `log` has a slot a
//!   backend fills once, and a C++ function-local static has no equivalent
//!   the facade will call. Until something here runs, `log`'s own macros
//!   discard everything. [`init`] installs it at level info, the C++ default;
//!   every other function here installs it too, so a binary that calls
//!   [`set_level`] first need not call [`init`].
//! - **The tail is flushed at exit by `atexit`.** A Rust static is never
//!   dropped, so a buffered sink in one would lose whatever was buffered
//!   when `main` returned, where the C++ runtime flushes `stdout` and closes
//!   the file. The handler runs on a normal exit and on
//!   [`std::process::exit`]; an abort skips it, which is why error and
//!   critical flush at once.
//!
//! **The warning counter counts warnings that were written.** It is bumped in
//! the backend, and `log::warn!` never reaches the backend when warn is
//! filtered out. That is how the C++ `FBF_LOG_WARN` counts. A bare C++
//! `log::warn` call counts even when filtered; there is no such call here.
//!
//! **A line written inside a task names it.** While a [`TaskScope`] is open on
//! a thread, every line that thread writes carries `[task@worker]` after its
//! location. The orchestrator opens one around each node it runs
//! ([`crate::core::thread_pool`]); a task does nothing for it.
//!
//! `log` has no critical level; it stops at error. A critical line goes
//! around the facade, straight to the sinks, labelled `critical`. It is
//! still filtered by [`set_level`].

use std::cell::RefCell;
use std::fmt;
use std::fs::File;
use std::io::{self, BufWriter, IsTerminal, Stdout, Write};
use std::marker::PhantomData;
use std::panic::Location;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicI32, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

thread_local! {
    /// The task this thread is running, as `task@worker`; see [`TaskScope`].
    static TASK: RefCell<Option<String>> = const { RefCell::new(None) };
}

/// The current-task record for the guard's lifetime: lines this thread
/// writes carry `[task@worker]`, and [`current_task`] returns it. The record
/// before it is restored on drop, so scopes nest.
///
/// Neither `Send` nor `Sync`: the record is the thread's, and restoring it
/// on another thread would set that thread's and leave this one's behind.
pub struct TaskScope {
    previous: Option<String>,
    not_send: PhantomData<*const ()>,
}

impl TaskScope {
    /// Name `task`, running on `worker`, as this thread's current task.
    pub fn enter(task: &str, worker: &str) -> Self {
        let previous = TASK.with_borrow_mut(|t| t.replace(format!("{task}@{worker}")));
        TaskScope {
            previous,
            not_send: PhantomData,
        }
    }
}

impl Drop for TaskScope {
    fn drop(&mut self) {
        let previous = self.previous.take();
        // A thread being torn down has no record left to restore.
        let _ = TASK.try_with(|t| *t.borrow_mut() = previous);
    }
}

/// This thread's current task, `task@worker`, if a [`TaskScope`] is open.
pub fn current_task() -> Option<String> {
    TASK.try_with(|t| t.borrow().clone()).ok().flatten()
}

/// Severity levels for log filtering: `LogLevel`, with the C++ (and
/// spdlog's) values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum LogLevel {
    /// Per-iteration / per-kernel, dev-only.
    Trace = 0,
    /// Algorithmic detail for diagnosing a wrong answer or slow convergence.
    Debug = 1,
    /// User-facing progress: what run this is, and how it went.
    Info = 2,
    /// Recoverable degradation: the run continues, not as requested.
    Warn = 3,
    /// A value-tier failure, logged once where it becomes an exit code.
    Error = 4,
    /// The abort tier: logged immediately before the process aborts.
    Critical = 5,
    /// Nothing is written.
    Off = 6,
}

impl LogLevel {
    fn from_log(level: ::log::Level) -> Self {
        match level {
            ::log::Level::Trace => Self::Trace,
            ::log::Level::Debug => Self::Debug,
            ::log::Level::Info => Self::Info,
            ::log::Level::Warn => Self::Warn,
            ::log::Level::Error => Self::Error,
        }
    }

    /// The facade's filter for this level. Critical and off both close it:
    /// the facade has nothing above error, and critical goes around it.
    fn filter(self) -> ::log::LevelFilter {
        match self {
            Self::Trace => ::log::LevelFilter::Trace,
            Self::Debug => ::log::LevelFilter::Debug,
            Self::Info => ::log::LevelFilter::Info,
            Self::Warn => ::log::LevelFilter::Warn,
            Self::Error => ::log::LevelFilter::Error,
            Self::Critical | Self::Off => ::log::LevelFilter::Off,
        }
    }

    /// spdlog's short names, which its `%l` writes.
    fn name(self) -> &'static str {
        match self {
            Self::Trace => "trace",
            Self::Debug => "debug",
            Self::Info => "info",
            Self::Warn => "warning",
            Self::Error => "error",
            Self::Critical => "critical",
            Self::Off => "off",
        }
    }

    /// spdlog's colour sink's escape for this level.
    fn colour(self) -> &'static str {
        match self {
            Self::Trace => "\x1b[37m",
            Self::Debug => "\x1b[36m",
            Self::Info => "\x1b[32m",
            Self::Warn => "\x1b[33m\x1b[1m",
            Self::Error => "\x1b[31m\x1b[1m",
            Self::Critical => "\x1b[1m\x1b[41m",
            Self::Off => "",
        }
    }
}

struct Sinks {
    stdout: BufWriter<Stdout>,
    colour: bool,
    file: Option<(PathBuf, BufWriter<File>)>,
}

struct Logger {
    level: AtomicI32,
    warnings: AtomicU64,
    sinks: Mutex<Sinks>,
}

impl Logger {
    fn enabled_at(&self, level: LogLevel) -> bool {
        level != LogLevel::Off && level as i32 >= self.level.load(Ordering::Relaxed)
    }

    fn write(&self, level: LogLevel, file: &str, line: u32, args: &fmt::Arguments<'_>) {
        if level == LogLevel::Warn {
            self.warnings.fetch_add(1, Ordering::Relaxed);
        }
        let time = jiff::Zoned::now().strftime("%Y-%m-%d %H:%M:%S%.3f");
        // SAFETY: gettid has no preconditions and cannot fail.
        let tid = unsafe { libc::gettid() };
        let base = file.rsplit('/').next().unwrap_or(file);
        let task = current_task()
            .map(|t| format!(" [{t}]"))
            .unwrap_or_default();
        // A poisoned lock means a write panicked mid-line; the sinks are
        // still usable, and losing the log over it would be worse.
        let mut sinks = self.sinks.lock().unwrap_or_else(|e| e.into_inner());
        let Sinks {
            stdout,
            colour,
            file,
        } = &mut *sinks;
        // A failed write has nowhere to be reported; the line is lost.
        let _ = if *colour {
            writeln!(
                stdout,
                "[{time}] [{tid}] [{}{}\x1b[0m] [{base}:{line}]{task} {args}",
                level.colour(),
                level.name()
            )
        } else {
            writeln!(
                stdout,
                "[{time}] [{tid}] [{}] [{base}:{line}]{task} {args}",
                level.name()
            )
        };
        if let Some((_, f)) = file {
            let _ = writeln!(
                f,
                "[{time}] [{tid}] [{}] [{base}:{line}]{task} {args}",
                level.name()
            );
        }
        if level >= LogLevel::Error {
            sinks.flush();
        }
    }
}

impl Sinks {
    fn flush(&mut self) {
        let _ = self.stdout.flush();
        if let Some((_, f)) = &mut self.file {
            let _ = f.flush();
        }
    }
}

impl ::log::Log for Logger {
    fn enabled(&self, metadata: &::log::Metadata<'_>) -> bool {
        self.enabled_at(LogLevel::from_log(metadata.level()))
    }

    fn log(&self, record: &::log::Record<'_>) {
        let level = LogLevel::from_log(record.level());
        if self.enabled_at(level) {
            let file = record.file().unwrap_or("?");
            self.write(level, file, record.line().unwrap_or(0), record.args());
        }
    }

    fn flush(&self) {
        self.sinks.lock().unwrap_or_else(|e| e.into_inner()).flush();
    }
}

/// The logger, once installed. A module-level static rather than one inside
/// [`installed`], so the exit handler can read it without installing one.
static LOGGER: OnceLock<Logger> = OnceLock::new();

/// The installed logger, installing it on first use.
fn installed() -> &'static Logger {
    let mut fresh = false;
    let lg = LOGGER.get_or_init(|| {
        fresh = true;
        Logger {
            level: AtomicI32::new(LogLevel::Info as i32),
            warnings: AtomicU64::new(0),
            sinks: Mutex::new(Sinks {
                stdout: BufWriter::new(io::stdout()),
                colour: io::stdout().is_terminal(),
                file: None,
            }),
        }
    });
    if fresh {
        // Another backend already installed means this process logs through
        // that one; this logger then serves only log_at and critical lines.
        if ::log::set_logger(lg).is_ok() {
            ::log::set_max_level(LogLevel::Info.filter());
        }
        // SAFETY: `flush_at_exit` is an `extern "C" fn()` that never unwinds.
        unsafe { libc::atexit(flush_at_exit) };
    }
    lg
}

/// Flush the sinks as the process exits. `try_lock`, not `lock`: a thread
/// still holding the sinks would otherwise hang the exit.
extern "C" fn flush_at_exit() {
    if let Some(lg) = LOGGER.get()
        && let Ok(mut sinks) = lg.sinks.try_lock()
    {
        sinks.flush();
    }
}

/// Install the logger at level info, if it is not installed yet. Every
/// other function here does this too.
pub fn init() {
    installed();
}

/// Set the global minimum log level: `set_level`.
pub fn set_level(level: LogLevel) {
    let lg = installed();
    lg.level.store(level as i32, Ordering::Relaxed);
    ::log::set_max_level(level.filter());
}

/// Would a line at `level` be written right now? `should_log`. A guard for
/// work done only to log, since the macros already guard their formatting.
pub fn should_log(level: LogLevel) -> bool {
    installed().enabled_at(level)
}

/// Flush every sink now: `flush`.
pub fn flush() {
    ::log::Log::flush(installed());
}

/// Write `args` at `level`, located at `location`: the C++ level functions'
/// explicit-location form, for a location already held (an error's origin,
/// a wrapper's construction site). Filtered by [`set_level`] like any line.
pub fn log_at(level: LogLevel, location: &'static Location<'static>, args: fmt::Arguments<'_>) {
    let lg = installed();
    if lg.enabled_at(level) {
        lg.write(level, location.file(), location.line(), &args);
    }
}

/// Add a file sink beside stdout: `add_file_sink`. The file is appended to,
/// and its directory created, as spdlog's `basic_file_sink_mt` does.
///
/// One file sink per process: a second call with another path warns, at the
/// caller, and is ignored; one with the same path does nothing. The C++ one
/// throws if the file cannot be opened; this returns that failure.
#[track_caller]
pub fn add_file_sink(path: impl AsRef<Path>) -> io::Result<()> {
    let path = path.as_ref();
    let lg = installed();
    let mut sinks = lg.sinks.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((registered, _)) = &sinks.file {
        if registered != path {
            let registered = registered.clone();
            drop(sinks);
            log_at(
                LogLevel::Warn,
                Location::caller(),
                format_args!(
                    "file sink already set to '{}', ignoring '{}'",
                    registered.display(),
                    path.display()
                ),
            );
        }
        return Ok(());
    }
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(dir)?;
    }
    let file = File::options().create(true).append(true).open(path)?;
    sinks.file = Some((path.to_path_buf(), BufWriter::new(file)));
    Ok(())
}

/// Warnings written so far this process: `warning_count`.
pub fn warning_count() -> u64 {
    installed().warnings.load(Ordering::Relaxed)
}

/// Reset the warning counter to zero, for test isolation:
/// `reset_warning_count`.
pub fn reset_warning_count() {
    installed().warnings.store(0, Ordering::Relaxed);
}
