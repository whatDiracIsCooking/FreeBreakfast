//! The link search path of the `ssperi3` kernel library, and the rpaths of
//! this crate's tests and examples. The `-sys` crates' build scripts name
//! rust/'s libraries and their search paths, which cargo passes on to a
//! dependent; an rpath it does not, so each linking crate emits its own.
//!
//! rust/'s CMake build directory is `FB_RUST_BUILD_DIR` if set, else
//! whichever of `rust/`'s `build-cuda` and `build-hip` exists. The
//! `ssperi3` library's is `FBX_SSPERI3_BUILD_DIR` if set, else
//! `../df_batched/single_gpu_spherical/kernels/build`: build it there
//! first (`cmake -S kernels -B kernels/build -G Ninja && cmake --build
//! kernels/build`).

use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-env-changed=FB_RUST_BUILD_DIR");
    println!("cargo:rerun-if-env-changed=FBX_SSPERI3_BUILD_DIR");
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
    let ssperi3 = std::env::var_os("FBX_SSPERI3_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join("../df_batched/single_gpu_spherical/kernels/build"));
    let ssperi3 = ssperi3.canonicalize().unwrap_or_else(|e| {
        panic!(
            "{}: {e}; build the ssperi3 library first, or set FBX_SSPERI3_BUILD_DIR",
            ssperi3.display()
        )
    });
    println!("cargo:rustc-link-search=native={}", ssperi3.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", ssperi3.display());
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
