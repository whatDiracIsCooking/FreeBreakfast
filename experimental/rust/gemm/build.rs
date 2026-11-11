//! Point the link at WarpWraps' C ABI libraries, which CMake builds in-tree
//! (deps/WarpWraps/src/capi under the build directory); wwr-sys's own build.rs
//! names which ones to link.
//!
//! The directory is looked for under FB_RUST_BUILD_DIR, the CMake build
//! directory, if set, else under build-experimental at the checkout's root.
//! wwr-sys also reads WWR_CAPI_LIB_DIR, but the rpath it emits applies to its
//! own targets only, so this crate's binaries and tests carry their own.
//! A crate depending on this one (rust/s_overlap) gets the search path but not
//! the rpath, and emits its own from the same capi_dir.rs.

include!("capi_dir.rs");

fn main() {
    let dir = capi_dir();
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
}
