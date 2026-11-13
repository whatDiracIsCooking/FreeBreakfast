// Link one of rust/'s kernel shared objects, libfb_<module>_kernels.so: the
// helper every kernel library's -sys crate shares (docs/gpu_layer.md,
// "Build"), included by its build.rs after capi_dir.rs, whose choice of
// <build> it reuses so the kernel library and libwwr_capi_runtime_api.so
// always come from the same CMake build. The library is at
// <build>/<module>/kernels, where rust/CMakeLists.txt's add_subdirectory puts
// it. The test binaries get an rpath to both directories: the kernel library
// NEEDs the C ABI's, and wwr-sys's own rpath reaches its own targets only.
//
// It also hands the crate the backend expected of the library, for its load
// check (`check_library()`, compared by fb-kernel-abi): the one <build> was
// configured for, read from its CMakeCache.txt, since wwr's C ABI has no
// query that names its backend. The crate reads it as
// `env!("FB_KERNELS_BACKEND")`, `CUDA` or `HIP`.

/// Links `libfb_<module>_kernels.so` and the C ABI library beside it, and
/// sets `FB_KERNELS_BACKEND` to the backend of their build.
fn link_kernel_lib(module: &str) {
    let capi = capi_dir();
    // capi_dir() is <build>/deps/WarpWraps/src/capi.
    let build = capi.ancestors().nth(4).unwrap();
    let dir = build.join(module).join("kernels");
    let lib = dir.join(format!("libfb_{module}_kernels.so"));
    if !lib.exists() {
        panic!(
            "{} is missing: build it first, from rust/, `cmake --preset cuda && \
             cmake --build --preset cuda` (or hip), into the CMake build \
             directory that holds libwwr_capi_runtime_api.so",
            lib.display()
        );
    }
    println!("cargo:rerun-if-changed={}", lib.display());
    for d in [&dir, &capi] {
        println!("cargo:rustc-link-search=native={}", d.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", d.display());
    }
    println!("cargo:rustc-link-lib=dylib=fb_{module}_kernels");
    let cache = build.join("CMakeCache.txt");
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
