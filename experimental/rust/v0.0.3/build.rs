//! The rpaths of this crate's tests and examples. The `-sys` crates' build
//! scripts name the libraries and their search paths, which cargo passes on
//! to a dependent; an rpath it does not, so each linking crate emits its own.
//!
//! `<build>` is `FB_RUST_BUILD_DIR` if set, else whichever of `rust/`'s
//! `build-cuda` and `build-hip` exists, as `rust/capi_dir.rs` chooses.

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=FB_RUST_BUILD_DIR");
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let rust = manifest.join("../../../rust");
    let build = match std::env::var_os("FB_RUST_BUILD_DIR") {
        Some(dir) => PathBuf::from(dir),
        None => ["build-cuda", "build-hip"]
            .iter()
            .map(|b| rust.join(b))
            .find(|b| b.join("CMakeCache.txt").exists())
            .expect("rust/build-cuda or rust/build-hip: build rust/'s CMake project first"),
    };
    for dir in [
        "deps/WarpWraps/src/capi",
        "recipes/kernels",
        "integrals/kernels",
    ] {
        let dir = build.join(dir);
        let dir = dir.canonicalize().unwrap_or(dir);
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
}
