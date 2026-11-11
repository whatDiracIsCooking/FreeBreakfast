// Where CMake put WarpWraps' C ABI libraries (libwwr_capi_<unit>.so): shared
// by the build.rs of each crate under experimental/rust/ that links them, by
// `include!`.

/// `<build>/deps/WarpWraps/src/capi`, where `<build>` is FB_RUST_BUILD_DIR if
/// set, else build-experimental at the checkout's root.
fn capi_dir() -> std::path::PathBuf {
    println!("cargo:rerun-if-env-changed=FB_RUST_BUILD_DIR");
    let build = std::env::var_os("FB_RUST_BUILD_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let manifest =
                std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
            manifest.join("../../../build-experimental")
        });
    let dir = build.join("deps/WarpWraps/src/capi");
    for unit in ["runtime_api", "blas"] {
        let lib = dir.join(format!("libwwr_capi_{unit}.so"));
        if !lib.exists() {
            panic!(
                "{} is missing: build it first, `cmake --build build-experimental \
                 --target wwr.capi.runtime_api wwr.capi.blas`, or point \
                 FB_RUST_BUILD_DIR at the CMake build directory holding it",
                lib.display()
            );
        }
        println!("cargo:rerun-if-changed={}", lib.display());
    }
    dir.canonicalize().unwrap_or(dir)
}
