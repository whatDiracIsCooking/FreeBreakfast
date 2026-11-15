//! Point the link at WarpWraps' runtime_api C ABI library
//! (`rust/capi_dir.rs`). wwr-sys's own build.rs names the library; its rpath
//! applies to its own targets only, so this crate's tests carry their own.
//!
//! It also hands the crate the backend of the library's build as
//! `FB_WWR_BACKEND`, `CUDA` or `HIP`, read from that build's CMakeCache.txt
//! as `rust/kernel_lib.rs` reads it, since wwr's C ABI has no query that
//! names its backend: [`runtime::stack_growth`] is 0 under HIP without a
//! device call, as the C++ `Device::stack_growth` is. Under
//! `FB_RUST_DOC_ONLY=1` it links nothing and names CUDA.

include!("../../capi_dir.rs");

fn main() {
    if doc_only() {
        println!("cargo:rustc-env=FB_WWR_BACKEND=CUDA");
        return;
    }
    let dir = capi_dir();
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    // capi_dir() is <build>/deps/WarpWraps/src/capi.
    let cache = dir.ancestors().nth(4).unwrap().join("CMakeCache.txt");
    println!("cargo:rerun-if-changed={}", cache.display());
    let text = std::fs::read_to_string(&cache)
        .unwrap_or_else(|e| panic!("{} is unreadable: {e}", cache.display()));
    let backend = text
        .lines()
        .find_map(|l| l.strip_prefix("FREE_BREAKFAST_GPU_BACKEND:STRING="))
        .unwrap_or_else(|| panic!("{} sets no FREE_BREAKFAST_GPU_BACKEND", cache.display()));
    assert!(
        matches!(backend, "CUDA" | "HIP"),
        "{}: FREE_BREAKFAST_GPU_BACKEND is '{backend}', not CUDA or HIP",
        cache.display()
    );
    println!("cargo:rustc-env=FB_WWR_BACKEND={backend}");
}
