//! `src/core/wwr`: the layer over WarpWraps.
//!
//! - [`common`]: wwr's `src/extension/common` (and the
//!   `src/extension/error_handling` primitives it re-exports).
//! - [`handle`]: wwr's `src/extension/handle`, the RAII handles and views.
//! - [`runtime`]: wwr's `src/extension/runtime`, the streams, events,
//!   memory pools and graphs built on them.
//! - [`blas`]: wwr's `src/extension/blas`, the BLAS status as an error code,
//!   the BLAS handle bound to a [`Stream`](runtime::Stream), and the typed
//!   calls through it ([`blas::gemm`]).
//! - [`device_slice`]: no wwr counterpart. Device memory with its length,
//!   which a safe library call takes and checks in place of a raw pointer.
//! - [`solver`]: wwr's `src/extension/solver`, the dense solver's status as
//!   an error code and its handle, bound to a stream as the BLAS one is.
//!
//! And free_breakfast's own partitions of `free_breakfast.core.wwr`, over
//! [`fb_core_error_handling`] and [`fb_core_log`]:
//!
//! - `:status`: [`gpu_status`] (and [`gpu_status_at`]), a GPU status as a
//!   [`Status`](fb_core_error_handling::Status), and
//!   [`gpu_log_error_at`], what a `Drop` does with a failed one.
//! - `:policy`: nothing. The C++ tree's `CreatePolicy` / `DestroyPolicy`
//!   are its rule that a failed create aborts and a failed destroy logs;
//!   here that rule is two calls rather than two types ([`common`] has
//!   why).

pub mod blas;
pub mod common;
pub mod device_slice;
pub mod handle;
pub mod runtime;
pub mod solver;
mod status;

pub use status::{gpu_log_error_at, gpu_status, gpu_status_at};
