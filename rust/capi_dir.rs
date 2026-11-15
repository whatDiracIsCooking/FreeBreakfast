// Where CMake put WarpWraps' runtime_api C ABI library
// (libwwr_capi_runtime_api.so): shared by the build.rs of each crate here
// whose tests load it, by `include!`. <build> is FB_RUST_BUILD_DIR if set,
// else whichever of rust/'s own CMake project's build-cuda and build-hip
// holds the library; with both built, the backend has no default here either,
// and FB_RUST_BUILD_DIR must name one. Only the library chosen is watched: a
// second backend built later is not noticed until FB_RUST_BUILD_DIR changes or
// the target directory is cleaned.
//
// FB_RUST_DOC_ONLY=1 is the one way round it: for `cargo doc` on a host with
// no CMake build (the Pages workflow, `.github/workflows/docs.yml`), every
// build.rs here asks `doc_only()` first and links nothing. rustdoc never
// links, so the docs build; a test or binary built so fails at the link.
// The crates' env!() backends are set to CUDA then (`rust/kernel_lib.rs`,
// `core/wwr/build.rs`), which no rendered page shows.

/// Whether `FB_RUST_DOC_ONLY` is `1`: document, and link nothing.
fn doc_only() -> bool {
    println!("cargo:rerun-if-env-changed=FB_RUST_DOC_ONLY");
    std::env::var_os("FB_RUST_DOC_ONLY").is_some_and(|v| v == "1")
}

/// `<build>/deps/WarpWraps/src/capi`, checked to hold the library.
fn capi_dir() -> std::path::PathBuf {
    const LIB: &str = "deps/WarpWraps/src/capi/libwwr_capi_runtime_api.so";
    println!("cargo:rerun-if-env-changed=FB_RUST_BUILD_DIR");
    let build = match std::env::var_os("FB_RUST_BUILD_DIR") {
        Some(dir) => std::path::PathBuf::from(dir),
        None => {
            // rust/ is the nearest ancestor holding this file: a crate sits
            // two levels below it or three (integrals/one_electron/overlap).
            let manifest = std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
            let rust = manifest
                .ancestors()
                .find(|d| d.join("capi_dir.rs").exists())
                .expect("a crate under rust/")
                .to_owned();
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
