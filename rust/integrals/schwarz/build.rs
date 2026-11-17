//! Give the projection test an rpath to `libfb_integrals_kernels.so`, which
//! the three-center integrals and the metric launch, to
//! `libfb_recipes_kernels.so`, which the fit's Cholesky factor launches, and
//! to the C ABI libraries beside `libwwr_capi_runtime_api.so`
//! (`rust/kernel_lib.rs`): the `-sys` crates' and wwr-sys's rpaths reach
//! their own targets only. The diagonal itself calls into neither.

include!("../../capi_dir.rs");
include!("../../kernel_lib.rs");

fn main() {
    link_kernel_lib("integrals");
    link_kernel_lib("recipes");
}
