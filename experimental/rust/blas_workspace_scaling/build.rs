//! Link and rpath WarpWraps' C ABI libraries from rust/'s own CMake project:
//! `<build>/deps/WarpWraps/src/capi`, where `<build>` is FB_RUST_BUILD_DIR if
//! set, else rust/build-cuda. run_all.sh sets it per backend, with a target
//! directory per backend, since the rpath is baked into the binary.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=FB_RUST_BUILD_DIR");
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let build = std::env::var_os("FB_RUST_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("../../../rust/build-cuda"));
    let dir = build.join("deps/WarpWraps/src/capi");
    for unit in ["runtime_api", "blas", "solver"] {
        let lib = dir.join(format!("libwwr_capi_{unit}.so"));
        assert!(
            lib.exists(),
            "{} is missing: from rust/, `cmake --preset cuda && cmake --build build-cuda \
             --target wwr.capi.runtime_api wwr.capi.blas wwr.capi.solver` (or hip), or set \
             FB_RUST_BUILD_DIR",
            lib.display()
        );
        println!("cargo:rerun-if-changed={}", lib.display());
    }
    let dir = dir.canonicalize().unwrap();
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
}
