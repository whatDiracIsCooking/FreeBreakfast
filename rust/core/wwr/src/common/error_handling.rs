//! wwr's `src/extension/error_handling` (module `wwr.extension.error_handling`),
//! which `wwr.extension.common` re-exports: its `:error_code` partition. The
//! `:error_policy` and `:gpu_check` partitions have no counterpart; see
//! [`super`].

use std::panic::Location;

/// The source location an error is reported at: `std::source_location`.
pub type SourceLocation = &'static Location<'static>;

/// An error code the error-handling layer supports: `error_type`, with
/// `success_code<T>()`, `error_name(T)` and `error_string(T)` as its items.
///
/// As in wwr, implementing it *is* the registration, and a half-registered
/// type cannot exist: the compiler demands all three.
///
/// `wwr_status` and `wwr_blas_status` ([`crate::blas`]) are registered
/// today. The C ABI's other library surfaces each have their own
/// (`wwr_wrappers_solver_status`, ...), so each library handle's
/// [`HandleKind::Error`](crate::handle::HandleKind::Error) is another.
pub trait ErrorCode: Copy + Eq + std::fmt::Debug {
    /// The success value: `success_code<T>()`.
    const SUCCESS: Self;
    /// The code's name: `error_name`.
    fn name(self) -> &'static str;
    /// What the code means: `error_string`.
    fn description(self) -> &'static str;
}
