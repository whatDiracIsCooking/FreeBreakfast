//! wwr's `src/extension/common` (module `wwr.extension.common`), and the
//! `src/extension/error_handling` primitives it re-exports.
//!
//! | C++ (wwr) | Here |
//! |---|---|
//! | `error_handling:error_code`: `success_code` / `error_name` / `error_string`, `concept error_type` | [`ErrorCode`], one trait (`error_handling.rs`) |
//! | `error_handling:error_policy`: `error_policy`, `typed_error_policy` | [`ErrorPolicy`], one trait with the error type associated |
//! | `error_handling:error_policy`: `nothrow_error_policy` | nothing: see [`ErrorPolicy`] |
//! | `error_handling:error_policy`: `policy_map` | [`PolicyMap`] |
//! | `error_handling:gpu_check` | [`gpu_check`], and [`gpu_check_at`] for a location already held |
//! | `:error`, the `wwrError_t` specializations | `impl ErrorCode for wwr_status` (`error.rs`) |
//! | `:scoped_device_index` | [`ScopedDeviceIndex`] (`scoped_device_index.rs`) |
//! | `:abort_policy`, `kit::AbortPolicy` | [`kit::AbortPolicy`] (`kit.rs`, named for its namespace) |
//! | `:noncopyable` | nothing: see below |
//! | `:policy_slot`, `elided_policy` | nothing: see below |
//!
//! Two partitions have no Rust counterpart because the language already does
//! what they exist to force:
//!
//! - `NonCopyable` deletes copy and keeps move. A Rust type is move-only
//!   unless it implements `Clone`/`Copy`, so a wrapper simply does not.
//! - `policy_slot` collapses a second same-type empty policy to nothing,
//!   because `[intro.object]` forbids two same-type subobjects one address
//!   and `[[no_unique_address]]` cannot elide the second. Rust has no such
//!   rule: every zero-sized field takes zero bytes, however many share a
//!   type, so a stateless policy in four slots costs nothing as written.
//!
//! `std::source_location` becomes [`SourceLocation`], a
//! `&'static Location<'static>` captured by `#[track_caller]` where the C++
//! captures it by a defaulted argument. Like that argument it reaches through
//! every `#[track_caller]` frame to the first caller that is not one. It
//! carries file, line and column; there is no function name.

mod error;
mod error_handling;
pub mod kit;
mod scoped_device_index;

pub use error_handling::{
    ErrorCode, ErrorPolicy, PolicyMap, SourceLocation, gpu_check, gpu_check_at,
};
pub use scoped_device_index::ScopedDeviceIndex;

#[cfg(test)]
pub(crate) mod tests {
    use super::{ErrorPolicy, SourceLocation};
    use wwr_sys::wwr_status;

    /// Records every failure it is handed, with the line it was reported at.
    #[derive(Default)]
    pub(crate) struct Recording(pub Vec<(wwr_status, u32)>);

    #[contracts::contract_trait]
    impl ErrorPolicy for Recording {
        type Error = wwr_status;
        fn handle_error(&mut self, error: wwr_status, location: SourceLocation) {
            self.0.push((error, location.line()));
        }
    }
}
