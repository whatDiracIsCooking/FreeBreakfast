//! free_breakfast's `src/` rebuilt in Rust, one layer at a time, over
//! WarpWraps' C ABI (the `wwr-sys` crate). The module tree mirrors the C++
//! one: `core::wwr` is the GPU vocabulary `src/core/wwr` holds.

pub mod core;
