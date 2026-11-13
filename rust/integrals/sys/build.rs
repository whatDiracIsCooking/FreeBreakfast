//! Link `libfb_integrals_kernels.so`, built by rust/'s CMake project
//! (`rust/kernel_lib.rs`).

include!("../../capi_dir.rs");
include!("../../kernel_lib.rs");

fn main() {
    link_kernel_lib("integrals");
}
