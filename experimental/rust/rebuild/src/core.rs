//! `src/core`: the GPU, error-handling and logging vocabulary.
//!
//! - [`log`]: `src/core/log`, the logger behind the `log` facade.
//! - [`error_handling`]: `src/core/error_handling`, `Error` and the abort
//!   tier.
//! - [`diagnostics`]: `src/core/diagnostics`, what `diagnostics.h`'s macros
//!   become.
//! - [`wwr`]: `src/core/wwr`, the layer over WarpWraps.

pub mod diagnostics;
pub mod error_handling;
pub mod log;
pub mod wwr;
