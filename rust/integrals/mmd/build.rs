//! Link `libfb_integrals_kernels.so` (`rust/kernel_lib.rs`): fb-integrals-sys
//! names it, but its rpath reaches its own targets only, so this crate's
//! tests carry their own.

include!("../../capi_dir.rs");
include!("../../kernel_lib.rs");

fn main() {
    link_kernel_lib("integrals");
}
