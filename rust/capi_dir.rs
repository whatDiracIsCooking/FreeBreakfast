// Where CMake put WarpWraps' runtime_api C ABI library
// (libwwr_capi_runtime_api.so): shared by the build.rs of each crate here
// whose tests load it, by `include!`. <build> is FB_RUST_BUILD_DIR if set,
// else whichever of rust/'s own CMake project's build-cuda and build-hip
// holds the library; with both built, the backend has no default here either,
// and FB_RUST_BUILD_DIR must name one. Only the library chosen is watched: a
// second backend built later is not noticed until FB_RUST_BUILD_DIR changes or
// the target directory is cleaned.

/// `<build>/deps/WarpWraps/src/capi`, checked to hold the library.
fn capi_dir() -> std::path::PathBuf {
    const LIB: &str = "deps/WarpWraps/src/capi/libwwr_capi_runtime_api.so";
    println!("cargo:rerun-if-env-changed=FB_RUST_BUILD_DIR");
    let build = match std::env::var_os("FB_RUST_BUILD_DIR") {
        Some(dir) => std::path::PathBuf::from(dir),
        None => {
            let rust = std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap())
                .join("../..");
            let built: Vec<_> = ["build-cuda", "build-hip"]
                .iter()
                .map(|b| rust.join(b))
                .filter(|b| b.join(LIB).exists())
                .collect();
            match built.as_slice() {
                [one] => one.clone(),
                [] => panic!(
                    "libwwr_capi_runtime_api.so is built in neither rust/build-cuda \
                     nor rust/build-hip: build it first, from rust/, `cmake --preset \
                     cuda && cmake --build --preset cuda` (or hip), or point \
                     FB_RUST_BUILD_DIR at the CMake build directory holding it"
                ),
                _ => panic!(
                    "libwwr_capi_runtime_api.so is built in both rust/build-cuda and \
                     rust/build-hip: set FB_RUST_BUILD_DIR to the one to link, with a \
                     CARGO_TARGET_DIR of its own"
                ),
            }
        }
    };
    let lib = build.join(LIB);
    if !lib.exists() {
        panic!(
            "{} is missing: build it first, from rust/, `cmake --preset cuda && \
             cmake --build --preset cuda` (or hip), or point FB_RUST_BUILD_DIR at \
             the CMake build directory holding it",
            lib.display()
        );
    }
    println!("cargo:rerun-if-changed={}", lib.display());
    let dir = lib.parent().unwrap().to_owned();
    dir.canonicalize().unwrap_or(dir)
}
