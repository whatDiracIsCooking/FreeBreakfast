//! Link and rpath the kernel libraries the plan launches, and wwr's C ABI
//! libraries beside them. rust/'s `-sys` crates emit rpaths for their own
//! targets only, so this binary carries its own (as rust/'s test build
//! scripts do, `rust/kernel_lib.rs`).
//!
//! The CMake build directory is FB_RUST_BUILD_DIR if set, else
//! rust/build-cuda of this checkout (`cmake --preset cuda && cmake --build
//! --preset cuda` from rust/).

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=FB_RUST_BUILD_DIR");
    let build = std::env::var_os("FB_RUST_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap())
                .join("../../../../rust/build-cuda")
        });
    let build = build.canonicalize().unwrap_or_else(|e| {
        panic!(
            "{}: {e}; build rust/'s libraries first (`cmake --preset cuda && \
             cmake --build --preset cuda` from rust/) or set FB_RUST_BUILD_DIR",
            build.display()
        )
    });
    for dir in [
        build.join("deps/WarpWraps/src/capi"),
        build.join("integrals/kernels"),
        build.join("recipes/kernels"),
    ] {
        assert!(dir.exists(), "{} is missing", dir.display());
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
}
