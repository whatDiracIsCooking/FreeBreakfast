//! `src/core/wwr`: the layer over WarpWraps.
//!
//! - [`common`]: wwr's `src/extension/common` (and the
//!   `src/extension/error_handling` primitives it re-exports).
//! - [`handle`]: wwr's `src/extension/handle`, the RAII handles and views.
//! - [`runtime`]: wwr's `src/extension/runtime`, the streams, events,
//!   memory pools and graphs built on them.

pub mod common;
pub mod handle;
pub mod runtime;
