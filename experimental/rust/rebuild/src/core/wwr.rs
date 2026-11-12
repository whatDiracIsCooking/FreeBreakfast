//! `src/core/wwr`: the layer over WarpWraps.
//!
//! - [`common`]: wwr's `src/extension/common` (and the
//!   `src/extension/error_handling` primitives it re-exports).
//! - [`handle`]: wwr's `src/extension/handle`, the RAII handles and views.

pub mod common;
pub mod handle;
