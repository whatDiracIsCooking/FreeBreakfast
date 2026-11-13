//! wwr's `src/extension/common` (module `wwr.extension.common`), and the
//! `src/extension/error_handling` primitives it re-exports.
//!
//! | C++ (wwr) | Here |
//! |---|---|
//! | `error_handling:error_code`: `success_code` / `error_name` / `error_string`, `concept error_type` | [`ErrorCode`], one trait (`error_handling.rs`) |
//! | `error_handling:error_policy`: `error_policy`, `typed_error_policy`, `nothrow_error_policy`, `policy_map` | nothing: see below |
//! | `error_handling:gpu_check` | nothing: [`gpu_status`](crate::core::wwr::gpu_status), which returns a `Result` |
//! | `:error`, the `wwrError_t` specializations | `impl ErrorCode for wwr_status` (`error.rs`) |
//! | `:scoped_device_index` | [`ScopedDeviceIndex`] (`scoped_device_index.rs`) |
//! | `:abort_policy`, `kit::AbortPolicy` | nothing: see below |
//! | `:noncopyable` | nothing: see below |
//! | `:policy_slot`, `elided_policy` | nothing: see below |
//!
//! **The error policies and `kit`.** wwr's policies answer one question for
//! a library whose consumers differ: what happens when a GPU call in a
//! constructor or a destructor fails. A C++ constructor cannot return a
//! status, so the answer has to be injected, as a type parameter per slot,
//! and `kit` is wwr's ready-made answer for a consumer who has not chosen
//! one. In Rust a constructor returns `Result`, and this tree's answer is
//! fixed (`docs/diagnostics.md`): a failed create aborts, a failed destroy
//! logs. So it is two calls, not a parameter:
//! `check_or_abort(gpu_status_at(..))` where a wrapper is constructed, and
//! [`gpu_log_error_at`](crate::core::wwr::gpu_log_error_at) in a `Drop`. A
//! site that must recover from a failed create calls the fallible
//! constructor underneath and handles the `Err`; nothing tree-wide changes.
//! With no policies there are no policy slots, so `policy_slot` goes too.
//!
//! **`NonCopyable`** deletes copy and keeps move. A Rust type is move-only
//! unless it implements `Clone`/`Copy`, so a wrapper simply does not.
//!
//! `std::source_location` becomes [`SourceLocation`], a
//! `&'static Location<'static>` captured by `#[track_caller]` where the C++
//! captures it by a defaulted argument. Like that argument it reaches through
//! every `#[track_caller]` frame to the first caller that is not one. It
//! carries file, line and column; there is no function name.

mod error;
mod error_handling;
mod scoped_device_index;

pub use error_handling::{ErrorCode, SourceLocation};
pub use scoped_device_index::ScopedDeviceIndex;
