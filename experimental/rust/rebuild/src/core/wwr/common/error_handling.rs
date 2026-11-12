//! wwr's `src/extension/error_handling` (module `wwr.extension.error_handling`),
//! which `wwr.extension.common` re-exports: its `:error_code`,
//! `:error_policy` and `:gpu_check` partitions, in one file.

use std::panic::Location;

use contracts::{contract_trait, debug_requires};

/// The source location an error is reported at: `std::source_location`.
pub type SourceLocation = &'static Location<'static>;

// ============================================================================
// :error_code
// ============================================================================

/// An error code the error-handling layer supports: `error_type`, with
/// `success_code<T>()`, `error_name(T)` and `error_string(T)` as its items.
///
/// As in wwr, implementing it *is* the registration, and a half-registered
/// type cannot exist: the compiler demands all three.
pub trait ErrorCode: Copy + Eq + std::fmt::Debug {
    /// The success value: `success_code<T>()`.
    const SUCCESS: Self;
    /// The code's name: `error_name`.
    fn name(self) -> &'static str;
    /// What the code means: `error_string`.
    fn description(self) -> &'static str;
}

// ============================================================================
// :error_policy
// ============================================================================

/// What to do with a failed call: `error_policy` and `typed_error_policy`
/// together, since a Rust policy always names the error type it handles.
///
/// A policy that works for any code is generic over it, as
/// [`super::kit::AbortPolicy`] is.
///
/// There is no separate `nothrow_error_policy`. That concept requires
/// `handle_error` to be `noexcept` in the destruction slots, because an
/// exception out of a destructor is `std::terminate`. A Rust policy reports
/// a failure by returning normally, by recording it, or by aborting; one that
/// panics in a `Drop` aborts the process if it is already unwinding, which is
/// the same outcome, and no trait can rule a panic out.
///
/// It is a `contracts` trait: every implementation carries
/// `#[contracts::contract_trait]` (one without it does not compile), so
/// the precondition on [`ErrorPolicy::handle_error`] is checked whichever
/// policy is called, in debug builds.
#[contract_trait]
pub trait ErrorPolicy {
    /// The error code this policy handles.
    type Error: ErrorCode;

    /// Handle `error`, reported at `location`. Never handed
    /// [`ErrorCode::SUCCESS`]: [`gpu_check`] calls it only on a failure, and
    /// a direct caller must do the same, so no policy re-checks.
    #[debug_requires(error != Self::Error::SUCCESS, "handle_error was handed SUCCESS")]
    fn handle_error(&mut self, error: Self::Error, location: SourceLocation);
}

/// A buffer's alloc/free policy pair for every error code it may meet:
/// `policy_map`, with its two member alias templates as generic associated
/// types.
///
/// wwr's concept is checked per error type, by the buffer suite that names
/// it; here the bounds on [`PolicyMap::Alloc`] and [`PolicyMap::Free`] are
/// checked once, for every `E`, where the map is implemented.
pub trait PolicyMap {
    /// The policy for the alloc calls on a buffer whose calls return `E`.
    type Alloc<E: ErrorCode>: ErrorPolicy<Error = E>;
    /// The policy for the free calls on a buffer whose calls return `E`.
    type Free<E: ErrorCode>: ErrorPolicy<Error = E>;
}

// ============================================================================
// :gpu_check
// ============================================================================

/// Check `error`, handing a failure to `policy` at the caller's location.
/// True on success.
///
/// `policy` is borrowed mutably, as wwr's forwarding reference passes a named
/// policy through, so a stateful policy records into the caller's instance.
#[track_caller]
pub fn gpu_check<P: ErrorPolicy>(error: P::Error, policy: &mut P) -> bool {
    gpu_check_at(error, policy, Location::caller())
}

/// [`gpu_check`] at a location the caller already holds: the C++ call that
/// passes its `location` argument on, as a wrapper does with the location
/// it captured at construction.
pub fn gpu_check_at<P: ErrorPolicy>(
    error: P::Error,
    policy: &mut P,
    location: SourceLocation,
) -> bool {
    if error != P::Error::SUCCESS {
        policy.handle_error(error, location);
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::super::kit::AbortPolicy;
    use super::super::tests::Recording;
    use super::*;
    use wwr_sys::wwr_status;

    #[test]
    fn gpu_check_passes_success_and_reports_failure_at_the_caller() {
        let mut policy = Recording::default();
        assert!(gpu_check(wwr_status::WWR_STATUS_SUCCESS, &mut policy));
        assert!(policy.0.is_empty());

        let line = line!() + 1;
        assert!(!gpu_check(wwr_status::WWR_STATUS_NO_DEVICE, &mut policy));
        assert_eq!(policy.0, [(wwr_status::WWR_STATUS_NO_DEVICE, line)]);
    }

    #[test]
    #[cfg(debug_assertions)] // the contract is compiled out of release
    #[should_panic(expected = "handle_error was handed SUCCESS")]
    fn handing_a_policy_success_breaks_its_contract() {
        Recording::default().handle_error(wwr_status::WWR_STATUS_SUCCESS, Location::caller());
    }

    #[test]
    fn policies_are_zero_sized() {
        assert_eq!(std::mem::size_of::<AbortPolicy<wwr_status>>(), 0);
        // What policy_slot exists for in C++: two same-type empty policies.
        struct TwoSlots(AbortPolicy<wwr_status>, AbortPolicy<wwr_status>);
        assert_eq!(std::mem::size_of::<TwoSlots>(), 0);
    }
}
