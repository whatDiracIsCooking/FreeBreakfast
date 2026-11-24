//! Link and rpath the kernel libraries the plan launches: rust/'s (as
//! ../single_gpu/build.rs does) and ../single_gpu_spherical's
//! libfbx_ssperi3_kernels.so (its kernels/CMakeLists.txt), which this
//! experiment shares.
//!
//! rust/'s CMake build directory is FB_RUST_BUILD_DIR if set, else
//! rust/build-cuda of this checkout; the ssperi3 library's is
//! FBX_SSPERI3_BUILD_DIR if set, else
//! ../single_gpu_spherical/kernels/build.

use std::path::PathBuf;

fn dir(var: &str, default: &str, how: &str) -> PathBuf {
    println!("cargo:rerun-if-env-changed={var}");
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let path = std::env::var_os(var)
        .map(PathBuf::from)
        .unwrap_or_else(|| manifest.join(default));
    path.canonicalize()
        .unwrap_or_else(|e| panic!("{}: {e}; {how}, or set {var}", path.display()))
}

fn main() {
    let rust = dir(
        "FB_RUST_BUILD_DIR",
        "../../../../rust/build-cuda",
        "build rust/'s libraries first (`cmake --preset cuda && cmake --build --preset cuda` \
         from rust/)",
    );
    let ssperi3 = dir(
        "FBX_SSPERI3_BUILD_DIR",
        "../single_gpu_spherical/kernels/build",
        "build the ssperi3 library first (`cmake -S kernels -B kernels/build -G Ninja && cmake \
         --build kernels/build` from ../single_gpu_spherical)",
    );
    println!(
        "cargo:rerun-if-changed={}",
        ssperi3.join("libfbx_ssperi3_kernels.so").display()
    );
    for dir in [
        rust.join("deps/WarpWraps/src/capi"),
        rust.join("integrals/kernels"),
        rust.join("recipes/kernels"),
        ssperi3,
    ] {
        assert!(dir.exists(), "{} is missing", dir.display());
        println!("cargo:rustc-link-search=native={}", dir.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
    }
}
