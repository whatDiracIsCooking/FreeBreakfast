//! Link the C ABI CMake builds from ffi/ by replaying the link line its probe,
//! free_breakfast_rust_s_overlap_linkprobe, recorded in fb_overlap.link
//! (link_record.rs has the how and the why). fb-gemm's build script points
//! the link at WarpWraps' C ABI libraries, but its rpath applies to its own
//! targets only, so this crate's binaries and tests get one here too.

include!("link_record.rs");
include!("../gemm/capi_dir.rs");

fn main() {
    replay(
        "s_overlap",
        "fb_overlap.link",
        "free_breakfast_rust_s_overlap_linkprobe",
    );
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", capi_dir().display());
}
