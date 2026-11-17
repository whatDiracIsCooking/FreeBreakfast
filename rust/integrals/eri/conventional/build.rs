//! Give the tests an rpath to `libfb_integrals_kernels.so` (the tensor, the
//! gather, the packing), to `libfb_recipes_kernels.so` (the average of J and
//! K) and to the C ABI libraries beside `libwwr_capi_runtime_api.so`,
//! `libwwr_capi_blas.so` among them (`rust/kernel_lib.rs`): the `-sys`
//! crates' and wwr-sys's rpaths reach their own targets only.

include!("../../../capi_dir.rs");
include!("../../../kernel_lib.rs");

fn main() {
    link_kernel_lib("integrals");
    link_kernel_lib("recipes");
}
