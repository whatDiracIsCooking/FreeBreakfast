//! Point the link at WarpWraps' runtime_api C ABI library, which CMake builds
//! in-tree under <build>/deps/WarpWraps/src/capi. <build> is FB_RUST_BUILD_DIR
//! if set, else build-experimental at the checkout's root.
//! wwr-sys's own build.rs names the library; its rpath applies to its own
//! targets only, so this crate's tests carry their own.

fn main() {
    println!("cargo:rerun-if-env-changed=FB_RUST_BUILD_DIR");
    let build = std::env::var_os("FB_RUST_BUILD_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            let manifest =
                std::path::PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
            manifest.join("../build-experimental")
        });
    let dir = build.join("deps/WarpWraps/src/capi");
    let lib = dir.join("libwwr_capi_runtime_api.so");
    if !lib.exists() {
        panic!(
            "{} is missing: build it first, `cmake --build build-experimental \
             --target wwr.capi.runtime_api`, or point FB_RUST_BUILD_DIR at the \
             CMake build directory holding it",
            lib.display()
        );
    }
    println!("cargo:rerun-if-changed={}", lib.display());
    let dir = dir.canonicalize().unwrap_or(dir);
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
}
