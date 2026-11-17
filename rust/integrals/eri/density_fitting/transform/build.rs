//! Give the tests an rpath to `libfb_integrals_kernels.so`, which the
//! packed map's crate names through `RecordTables`, and to the C ABI
//! libraries beside it, `libwwr_capi_blas.so` among them
//! (`rust/kernel_lib.rs`): fb-integrals-sys's and wwr-sys's rpaths reach
//! their own targets only.

include!("../../../../capi_dir.rs");
include!("../../../../kernel_lib.rs");

fn main() {
    link_kernel_lib("integrals");
}
