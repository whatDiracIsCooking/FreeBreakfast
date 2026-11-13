//! Link `libfb_integrals_kernels.so`, built by rust/'s CMake project, and
//! record the backend of that build for the load check (`src/lib.rs`): both
//! `rust/kernel_lib.rs`.

include!("../../capi_dir.rs");
include!("../../kernel_lib.rs");

fn main() {
    link_kernel_lib("integrals");
}
