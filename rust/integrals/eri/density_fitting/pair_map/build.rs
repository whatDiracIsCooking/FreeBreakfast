//! Give the tests an rpath to `libfb_integrals_kernels.so` and to the C ABI
//! libraries beside it (`rust/kernel_lib.rs`): fb-integrals-sys names the
//! library, whose `RecordTables` the map is built from, but its rpath
//! reaches its own targets only.

include!("../../../../capi_dir.rs");
include!("../../../../kernel_lib.rs");

fn main() {
    link_kernel_lib("integrals");
}
