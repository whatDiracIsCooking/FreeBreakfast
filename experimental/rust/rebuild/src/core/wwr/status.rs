//! `free_breakfast.core.wwr:status`: a GPU status as a [`Status`].

use std::panic::Location;

use super::common::{ErrorCode, SourceLocation};
use crate::core::error_handling::{Error, ErrorKind, Status};

/// Convert a GPU status to a [`Status`], located at the caller:
/// `gpu_status`.
///
/// Takes anything registered as an [`ErrorCode`], `wwr_status` and the
/// library codes alike. A failure becomes [`ErrorKind::GpuRuntime`] carrying
/// the code's name and description. Named `gpu_status`, not `gpu_check`, so
/// it does not collide with [`super::common::gpu_check`], which hands a
/// failure to a policy and returns `bool`.
#[track_caller]
pub fn gpu_status<E: ErrorCode>(code: E) -> Status {
    gpu_status_at(code, Location::caller())
}

/// [`gpu_status`] at a location the caller already holds: the C++ call that
/// passes its `loc` on, as the error policies do with the location a wrapper
/// captured at construction.
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
