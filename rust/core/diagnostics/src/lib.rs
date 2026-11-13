//! `src/core/diagnostics` (`diagnostics.h`): the call-site idioms over
//! [`fb_core_error_handling`] and [`fb_core_log`].
//!
//! The C++ header is macros because a defaulted `std::source_location` and
//! a level check before `std::format` both need expanding at the call site,
//! and macros cannot cross a module boundary. Here most of it is already
//! the language or the `log` crate, and what is left is one function and one
//! macro:
//!
//! | `diagnostics.h` | Here |
//! |---|---|
//! | `FBF_TRY(expr)` | `expr?`: it moves the [`Error`](fb_core_error_handling::Error) through unchanged, so its origin stays the detection site |
//! | `FBF_GPU_MUST(call)` | [`gpu_must`], a `#[track_caller]` function |
//! | `FBF_LOG_TRACE` ... `FBF_LOG_ERROR` | [`trace!`] ... [`error!`], `log`'s own macros, re-exported |
//! | `FBF_LOG_CRITICAL` | [`critical!`] |
//!
//! Every macro here formats only when its level would be written, which is
//! what the `FBF_LOG_*` macros exist to enforce; `log`'s check the level
//! first, and [`critical!`] hands `format_args!` (formatted only when
//! written) to [`log_at`](fb_core_log::log_at).
//!
//! ```ignore
//! use fb_core_diagnostics::{critical, debug, gpu_must};
//! use fb_core_error_handling::Result;
//!
//! fn load(path: &str) -> Result<Config> {
//!     let bytes = read_file(path)?;            // FBF_TRY(read_file(path))
//!     validate(&bytes)?;                       // FBF_TRY(validate(bytes))
//!     parse(&bytes)
//! }
//!
//! gpu_must(unsafe { wwr_sys::wwr_set_device(0) });       // FBF_GPU_MUST
//! debug!("iteration {iter} residual {residual:.3e}");   // FBF_LOG_DEBUG
//! critical!("input {name} is empty");                  // FBF_LOG_CRITICAL
//! ```
//!
//! **Value-tier errors are logged once**, at the boundary that turns an
//! `Error` into an exit code
//! ([`ErrorKind::exit_code`](fb_core_error_handling::ErrorKind::exit_code)),
//! not at every `?` in between. The factories construct and do not log;
//! only the abort tier ([`check_or_abort`], and [`gpu_must`] over it) logs
//! where it detects, because there detection and the process boundary are
//! the same call.

#![forbid(unsafe_code)]

pub use ::log::{debug, error, info, trace, warn};
pub use fb_core_log::critical;

use fb_core_error_handling::check_or_abort;
use fb_core_wwr::common::ErrorCode;
use fb_core_wwr::gpu_status;

/// Check a GPU call's status, and on failure log it at critical against the
/// caller and abort: `FBF_GPU_MUST(call)`, which is
/// `check_or_abort(gpu_status(call))`. Does nothing else on success.
#[track_caller]
pub fn gpu_must<E: ErrorCode>(code: E) {
    check_or_abort(gpu_status(code));
}
