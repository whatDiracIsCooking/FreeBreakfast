//! Point the link at `libfb_recipes_kernels.so` and WarpWraps' C ABI
//! libraries beside `libwwr_capi_runtime_api.so` (`rust/kernel_lib.rs`):
//! fb-recipes-sys's and wwr-sys's rpaths reach their own targets only, so
//! this crate's tests and its example carry their own. Under
//! `FB_RUST_DOC_ONLY=1` it links nothing.

include!("../capi_dir.rs");
include!("../kernel_lib.rs");

fn main() {
    link_kernel_lib("recipes");
    link_kernel_lib("integrals");
}
