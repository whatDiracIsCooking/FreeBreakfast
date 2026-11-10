//! Link the C ABI CMake builds from ffi/ by replaying the link line its probe,
//! free_breakfast_rust_s_overlap_linkprobe, recorded in fb_overlap.link. That
//! line carries rust/gemm's archive too, so this crate's binaries need nothing
//! from fb-gemm's own build script (whose link arguments apply to its own
//! targets only). ../gemm/link_record.rs has the how and the why.

include!("../gemm/link_record.rs");

fn main() {
    replay(
        "s_overlap",
        "fb_overlap.link",
        "free_breakfast_rust_s_overlap_linkprobe",
    );
}
