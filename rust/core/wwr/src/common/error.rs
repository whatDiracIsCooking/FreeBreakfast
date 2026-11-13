//! `wwr.extension.common:error`: `wwrError_t` registered as an error code.

use super::ErrorCode;
use wwr_sys::wwr_status;

/// `wwrError_t`, as the C ABI hands it back: the vendor's code mapped onto
/// a frozen set of eight. The name is the enumerator's; wwr's would be the
/// vendor's (`cudaErrorInvalidDevice`), which the C ABI does not carry.
impl ErrorCode for wwr_status {
    const SUCCESS: Self = wwr_status::WWR_STATUS_SUCCESS;

    fn name(self) -> &'static str {
        match self {
            wwr_status::WWR_STATUS_SUCCESS => "WWR_STATUS_SUCCESS",
            wwr_status::WWR_STATUS_INVALID_VALUE => "WWR_STATUS_INVALID_VALUE",
            wwr_status::WWR_STATUS_INVALID_DEVICE => "WWR_STATUS_INVALID_DEVICE",
            wwr_status::WWR_STATUS_NO_DEVICE => "WWR_STATUS_NO_DEVICE",
            wwr_status::WWR_STATUS_NOT_READY => "WWR_STATUS_NOT_READY",
            wwr_status::WWR_STATUS_MEMORY_ALLOCATION => "WWR_STATUS_MEMORY_ALLOCATION",
            wwr_status::WWR_STATUS_NOT_SUPPORTED => "WWR_STATUS_NOT_SUPPORTED",
            wwr_status::WWR_STATUS_UNKNOWN => "WWR_STATUS_UNKNOWN",
        }
    }

    fn description(self) -> &'static str {
        match self {
            wwr_status::WWR_STATUS_SUCCESS => "no error",
            wwr_status::WWR_STATUS_INVALID_VALUE => "invalid argument",
            wwr_status::WWR_STATUS_INVALID_DEVICE => "invalid device ordinal",
            wwr_status::WWR_STATUS_NO_DEVICE => "no GPU device is available",
            wwr_status::WWR_STATUS_NOT_READY => "work has not completed yet",
            wwr_status::WWR_STATUS_MEMORY_ALLOCATION => "out of memory",
            wwr_status::WWR_STATUS_NOT_SUPPORTED => "operation not supported",
            wwr_status::WWR_STATUS_UNKNOWN => "a backend error with no stable code of its own",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_status_has_a_name_and_a_description() {
        use wwr_status::*;
        for s in [
            WWR_STATUS_SUCCESS,
            WWR_STATUS_INVALID_VALUE,
            WWR_STATUS_INVALID_DEVICE,
            WWR_STATUS_NO_DEVICE,
            WWR_STATUS_NOT_READY,
            WWR_STATUS_MEMORY_ALLOCATION,
            WWR_STATUS_NOT_SUPPORTED,
            WWR_STATUS_UNKNOWN,
        ] {
            assert_eq!(s.name(), format!("{s:?}"));
            assert!(!s.description().is_empty());
        }
    }
}
