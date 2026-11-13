//! `src/core/error_handling` (module `free_breakfast.core.error_handling`):
//! the error value, the `Result` aliases over it, and the abort tier.
//!
//! | C++ (`free_breakfast`) | Here |
//! |---|---|
//! | `:error`: `ErrorKind` | [`ErrorKind`] |
//! | `:error`: `Error`, its factories (`Error::io`, ...) | [`Error`], the same factories, `#[track_caller]` |
//! | `:error`: `Result<T>`, `Status` | [`Result<T>`], [`Status`] |
//! | `:error`: `to_exit_code` | [`ErrorKind::exit_code`] |
//! | `:error`: `err`, `LocatedFormat`, the `err_*` helpers | nothing: see below |
//! | `:abort`: both `check_or_abort` overloads | [`check_or_abort`], one generic function |
//!
//! Like the C++ module it does not name WarpWraps: the bridge from a GPU
//! status (`fb_core_wwr::gpu_status`) and the log line
//! a failed destroy gets (`fb_core_wwr::gpu_log_error_at`)
//! live in `fb_core_wwr`, which builds on this.
//!
//! **The `err_*` helpers have no counterpart.** They exist because a C++
//! defaulted `std::source_location` cannot follow a parameter pack, so the
//! location rides on the format string (`LocatedFormat`), and because
//! `std::unexpected` is wordy. A `#[track_caller]` factory has neither
//! problem: `Err(Error::io(format!("cannot open '{}'", path.display())))`
//! captures its caller wherever the `format!` is, and `Err` is short. `err`
//! itself is `Err`.
//!
//! **A location already held** (the C++ `Error::gpu(msg, loc)`) is a struct
//! literal: the fields are public, as the C++ aggregate's are.
//!
//! **Propagation is `?`.** It moves the `Error` through unchanged, as
//! `FBF_TRY` does, so [`Error::origin`] stays the detection site.

#![forbid(unsafe_code)]

use std::fmt;
use std::panic::Location;

use fb_core_log::{self as log, LogLevel};

/// Classification of errors: `ErrorKind`. Flat and small on purpose, a
/// dispatch tag for [`ErrorKind::exit_code`] and for callers that branch on
/// failure category. See the C++ module's README for the easily-confused
/// pairs (divergence vs max iterations, unsupported vs invalid config,
/// numerical vs GPU runtime) and for why parse errors fold into `Io`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// Host or device memory could not be obtained.
    Allocation,
    /// A file, stream or parse operation failed.
    Io,
    /// A GPU runtime or library call itself failed.
    GpuRuntime,
    /// Well-formed input that is semantically wrong.
    InvalidConfig,
    /// An iterative process went bad: the iterate itself, not the budget.
    Divergence,
    /// Still converging, but out of iteration budget.
    MaxIterationsExceeded,
    /// Well-formed input asking for a deliberately unimplemented feature.
    Unsupported,
    /// Completed, but the result is mathematically invalid.
    Numerical,
}

impl ErrorKind {
    /// The process exit code for this kind: `to_exit_code`.
    ///
    /// The numbers are pinned, and the same as the C++ ones: a shell script
    /// branching on `$?` is an API consumer, so a new kind extends the
    /// mapping rather than renumbering it. The match is exhaustive, so the
    /// C++ fallback (127) cannot be reached and has no counterpart.
    pub const fn exit_code(self) -> i32 {
        match self {
            Self::InvalidConfig => 1,
            Self::Allocation => 2,
            Self::GpuRuntime => 3,
            Self::Io => 4,
            Self::Divergence => 5,
            Self::MaxIterationsExceeded => 6,
            Self::Unsupported => 7,
            Self::Numerical => 8,
        }
    }
}

/// A structured error: kind, message, and where it was first constructed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    /// Failure category; decides the exit code.
    pub kind: ErrorKind,
    /// Human-readable description, including any detail `kind` does not
    /// distinguish.
    pub message: String,
    /// Where the error was first constructed. Propagation moves it through
    /// unchanged, so this stays the detection site.
    pub origin: &'static Location<'static>,
    /// What the error crossed on its way out, innermost first: the
    /// orchestrator pushes `task@worker` when an error leaves a task
    /// (`fb_core_thread_pool`). Empty at the detection site.
    pub context: Vec<String>,
}

impl Error {
    #[track_caller]
    fn at_caller(kind: ErrorKind, message: impl Into<String>) -> Self {
        Error {
            kind,
            message: message.into(),
            origin: Location::caller(),
            context: Vec::new(),
        }
    }

    /// Push `context` onto [`Error::context`], outermost so far, and return
    /// the error: `result.map_err(|e| e.with_context("j@stream0"))`.
    #[must_use]
    pub fn with_context(mut self, context: impl Into<String>) -> Self {
        self.context.push(context.into());
        self
    }

    /// An [`ErrorKind::Allocation`] error, located at the caller.
    #[track_caller]
    pub fn allocation(message: impl Into<String>) -> Self {
        Self::at_caller(ErrorKind::Allocation, message)
    }
    /// An [`ErrorKind::Io`] error, located at the caller.
    #[track_caller]
    pub fn io(message: impl Into<String>) -> Self {
        Self::at_caller(ErrorKind::Io, message)
    }
    /// An [`ErrorKind::GpuRuntime`] error, located at the caller.
    #[track_caller]
    pub fn gpu(message: impl Into<String>) -> Self {
        Self::at_caller(ErrorKind::GpuRuntime, message)
    }
    /// An [`ErrorKind::InvalidConfig`] error, located at the caller.
    #[track_caller]
    pub fn config(message: impl Into<String>) -> Self {
        Self::at_caller(ErrorKind::InvalidConfig, message)
    }
    /// An [`ErrorKind::Divergence`] error, located at the caller: the
    /// iterate blew up, not merely ran out of iterations.
    #[track_caller]
    pub fn divergence(message: impl Into<String>) -> Self {
        Self::at_caller(ErrorKind::Divergence, message)
    }
    /// An [`ErrorKind::MaxIterationsExceeded`] error, located at the caller:
    /// still converging, out of budget.
    #[track_caller]
    pub fn max_iterations_exceeded(message: impl Into<String>) -> Self {
        Self::at_caller(ErrorKind::MaxIterationsExceeded, message)
    }
    /// An [`ErrorKind::Unsupported`] error, located at the caller.
    #[track_caller]
    pub fn unsupported(message: impl Into<String>) -> Self {
        Self::at_caller(ErrorKind::Unsupported, message)
    }
    /// An [`ErrorKind::Numerical`] error, located at the caller.
    #[track_caller]
    pub fn numerical(message: impl Into<String>) -> Self {
        Self::at_caller(ErrorKind::Numerical, message)
    }
}

/// The message, then each [`Error::context`] entry in brackets, innermost
/// first. The origin is not printed: it is the log line's location.
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)?;
        for context in &self.context {
            write!(f, " [{context}]")?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {}

/// A failable return: `Result<T>`, `std::expected<T, Error>`.
pub type Result<T> = std::result::Result<T, Error>;

/// A failable return with no value: `Status`.
pub type Status = Result<()>;

/// Unwrap `result`, or log its error at critical against the error's origin
/// and abort: `check_or_abort`, both overloads (a [`Status`] is a
/// `Result<()>`).
///
/// The abort tier, and the one place an error is logged where it is
/// detected: there, detection and the process boundary are the same call.
/// A critical line flushes before the abort, which runs no exit handlers.
pub fn check_or_abort<T>(result: Result<T>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => {
            log::log_at(LogLevel::Critical, error.origin, format_args!("{error}"));
            std::process::abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exit_codes_are_pinned() {
        use ErrorKind::*;
        let pinned = [
            (InvalidConfig, 1),
            (Allocation, 2),
            (GpuRuntime, 3),
            (Io, 4),
            (Divergence, 5),
            (MaxIterationsExceeded, 6),
            (Unsupported, 7),
            (Numerical, 8),
        ];
        for (kind, code) in pinned {
            assert_eq!(kind.exit_code(), code, "{kind:?}");
        }
    }

    #[test]
    fn every_factory_makes_its_own_kind() {
        use ErrorKind::*;
        let made = [
            (Error::allocation("m"), Allocation),
            (Error::io("m"), Io),
            (Error::gpu("m"), GpuRuntime),
            (Error::config("m"), InvalidConfig),
            (Error::divergence("m"), Divergence),
            (Error::max_iterations_exceeded("m"), MaxIterationsExceeded),
            (Error::unsupported("m"), Unsupported),
            (Error::numerical("m"), Numerical),
        ];
        for (error, kind) in made {
            assert_eq!(error.kind, kind);
            assert_eq!(error.message, "m");
        }
    }

    fn detect(raw: i32) -> Result<i32> {
        if raw <= 0 {
            return Err(Error::config(format!("count {raw} must be positive")));
        }
        Ok(raw)
    }

    fn propagate(raw: i32) -> Result<i32> {
        let n = detect(raw)?;
        Ok(n * 2)
    }

    #[test]
    fn the_origin_is_the_detection_site_through_propagation() {
        let error = propagate(-1).unwrap_err();
        assert_eq!(error.kind, ErrorKind::InvalidConfig);
        assert_eq!(error.message, "count -1 must be positive");
        assert_eq!(error.origin.file(), file!());
        // The factory call in `detect`, not the `?` in `propagate`.
        let source = include_str!("lib.rs");
        let detect_line = source
            .lines()
            .position(|l| l.contains("return Err(Error::config(format!(\"count"))
            .unwrap() as u32
            + 1;
        assert_eq!(error.origin.line(), detect_line);
        assert_eq!(propagate(3), Ok(6));
    }

    #[test]
    fn check_or_abort_unwraps_success() {
        assert_eq!(check_or_abort(Ok::<_, Error>(7)), 7);
        check_or_abort(Status::Ok(()));
    }
}
