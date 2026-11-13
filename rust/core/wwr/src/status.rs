//! `free_breakfast.core.wwr:status`: a GPU status as a [`Status`], and what
//! a `Drop` does with a failed one.

use std::panic::Location;

use super::common::{ErrorCode, SourceLocation};
use fb_core_error_handling::{Error, ErrorKind, Status};
use fb_core_log::{self as log, LogLevel};

/// Convert a GPU status to a [`Status`], located at the caller:
/// `gpu_status`.
///
/// Takes anything registered as an [`ErrorCode`], `wwr_status` and the
/// library codes alike. A failure becomes [`ErrorKind::GpuRuntime`] carrying
/// the code's name and description.
#[track_caller]
pub fn gpu_status<E: ErrorCode>(code: E) -> Status {
    gpu_status_at(code, Location::caller())
}

/// [`gpu_status`] at a location the caller already holds: the C++ call that
/// passes its `loc` on, as a constructor does with the location its caller
/// gave it.
pub fn gpu_status_at<E: ErrorCode>(code: E, location: SourceLocation) -> Status {
    if code == E::SUCCESS {
        return Ok(());
    }
    Err(Error {
        kind: ErrorKind::GpuRuntime,
        message: format!("{} ({})", code.name(), code.description()),
        origin: location,
        context: Vec::new(),
    })
}

/// Log a failed GPU status at error, at `location`, and carry on: what a
/// `Drop` does with a failed destroy or free, the C++ `DestroyPolicy`.
/// Does nothing on success.
///
/// A destroy runs in a `Drop`, often during teardown after an earlier
/// failure, where aborting would bury the root cause under the knock-on
/// one. The C++ policy wraps its body in `try`/`catch (...)`, because
/// formatting can throw out of a `noexcept` function; here a failed write is
/// dropped inside the logger, and a failed allocation aborts in Rust
/// whatever the caller does, so there is nothing to catch.
pub fn gpu_log_error_at<E: ErrorCode>(code: E, location: SourceLocation) {
    if let Err(e) = gpu_status_at(code, location) {
        log::log_at(LogLevel::Error, location, format_args!("{e}"));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wwr_sys::wwr_status;

    #[test]
    fn success_is_ok_and_a_failure_is_a_located_gpu_error() {
        assert_eq!(gpu_status(wwr_status::WWR_STATUS_SUCCESS), Ok(()));

        let line = line!() + 1;
        let error = gpu_status(wwr_status::WWR_STATUS_NO_DEVICE).unwrap_err();
        assert_eq!(error.kind, ErrorKind::GpuRuntime);
        assert_eq!(
            error.message,
            "WWR_STATUS_NO_DEVICE (no GPU device is available)"
        );
        assert_eq!((error.origin.file(), error.origin.line()), (file!(), line));
    }
}
