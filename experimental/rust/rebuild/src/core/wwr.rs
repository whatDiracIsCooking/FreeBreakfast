//! `src/core/wwr`: the layer over WarpWraps.
//!
//! - [`common`]: wwr's `src/extension/common` (and the
//!   `src/extension/error_handling` primitives it re-exports).
//! - [`handle`]: wwr's `src/extension/handle`, the RAII handles and views.
//! - [`runtime`]: wwr's `src/extension/runtime`, the streams, events,
//!   memory pools and graphs built on them.
//!
//! And two of free_breakfast's own partitions of `free_breakfast.core.wwr`,
//! over [`crate::core::error_handling`] and [`crate::core::log`]:
//!
//! - `:status`: [`gpu_status`] (and [`gpu_status_at`]), a GPU status as a
//!   [`Status`](crate::core::error_handling::Status).
//! - `:policy`: [`AbortPolicy`] / [`LogPolicy`], [`CreatePolicy`] /
//!   [`DestroyPolicy`] and [`PolicyMap`], the tree's policies for the
//!   wrappers.

pub mod common;
pub mod handle;
mod policy;
pub mod runtime;
mod status;

pub use policy::{AbortPolicy, CreatePolicy, DestroyPolicy, LogPolicy, PolicyMap};
pub use status::{gpu_status, gpu_status_at};
