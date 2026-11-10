//! Link the C ABI CMake builds from ffi/ by replaying the link line its probe,
//! free_breakfast_rust_gemm_linkprobe, recorded in fb_gemm.link. link_record.rs
//! has the how and the why.

include!("link_record.rs");

fn main() {
    replay("gemm", "fb_gemm.link", "free_breakfast_rust_gemm_linkprobe");
}
