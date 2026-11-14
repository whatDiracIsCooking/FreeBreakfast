//! Give the tests an rpath to `libfb_recipes_kernels.so` and to the C ABI
//! libraries beside `libwwr_capi_runtime_api.so` (`rust/kernel_lib.rs`):
//! fb-recipes-sys's and wwr-sys's rpaths reach their own targets only.

include!("../../capi_dir.rs");
include!("../../kernel_lib.rs");

fn main() {
    link_kernel_lib("recipes");
}
