//! Link `libfb_recipes_kernels.so`, built by rust/'s CMake project
//! (`rust/kernel_lib.rs`), and record the backend of that build for the
//! load check (`src/lib.rs`).

include!("../../capi_dir.rs");
include!("../../kernel_lib.rs");

fn main() {
    link_kernel_lib("recipes");
    // The backend expected of the library: the one the CMake build whose C
    // ABI library wwr-sys links was configured for, from its cache. wwr's C
    // ABI has no query that names its backend.
    let cache = capi_dir()
        .ancestors()
        .nth(4)
        .unwrap()
        .join("CMakeCache.txt");
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
    println!("cargo:rustc-env=FB_KERNELS_BACKEND={backend}");
}
