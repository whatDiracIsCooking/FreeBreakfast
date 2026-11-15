//! Link `libfb_integrals_kernels.so` and `libfb_recipes_kernels.so` (the
//! lower triangle's mirror) (`rust/kernel_lib.rs`), and give the tests an
//! rpath to them and to the C ABI libraries beside
//! `libwwr_capi_runtime_api.so`, `libwwr_capi_blas.so` among them: the
//! `-sys` crates' and wwr-sys's rpaths reach their own targets only.

include!("../../../capi_dir.rs");
include!("../../../kernel_lib.rs");

fn main() {
    link_kernel_lib("integrals");
    link_kernel_lib("recipes");
}
