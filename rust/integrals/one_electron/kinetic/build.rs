//! Link `libfb_integrals_kernels.so` (`rust/kernel_lib.rs`), and give the
//! tests an rpath to it and to the C ABI libraries beside
//! `libwwr_capi_runtime_api.so`, `libwwr_capi_blas.so` among them:
//! fb-integrals-sys's and wwr-sys's rpaths reach their own targets only.

include!("../../../capi_dir.rs");
include!("../../../kernel_lib.rs");

fn main() {
    link_kernel_lib("integrals");
}
