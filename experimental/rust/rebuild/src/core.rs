//! `src/core`: the GPU, error-handling and logging vocabulary.
//!
//! - [`log`]: `src/core/log`, the logger behind the `log` facade.
//! - [`error_handling`]: `src/core/error_handling`, `Error` and the abort
//!   tier.
//! - [`diagnostics`]: `src/core/diagnostics`, what `diagnostics.h`'s macros
//!   become.
//! - [`wwr`]: `src/core/wwr`, the layer over WarpWraps.
//! - [`thread_pool`]: no C++ counterpart; the run's workers, which execute
//!   a placed DAG on a fixed set of threads.

pub mod diagnostics;
pub mod error_handling;
pub mod log;
pub mod thread_pool;
pub mod wwr;
