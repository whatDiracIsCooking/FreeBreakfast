// Link one of rust/'s kernel shared objects, libfb_<module>_kernels.so: the
// helper every kernel library's -sys crate shares (docs/gpu_layer.md,
// "Build"), included by its build.rs after capi_dir.rs, whose choice of
// <build> it reuses so the kernel library and libwwr_capi_runtime_api.so
// always come from the same CMake build. The library is at
// <build>/<module>/kernels, where rust/CMakeLists.txt's add_subdirectory puts
// it. The test binaries get an rpath to both directories: the kernel library
// NEEDs the C ABI's, and wwr-sys's own rpath reaches its own targets only.

/// Links `libfb_<module>_kernels.so` and the C ABI library beside it.
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
}
