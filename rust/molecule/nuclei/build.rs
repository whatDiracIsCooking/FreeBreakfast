//! Point the link at WarpWraps' runtime_api C ABI library
//! (`rust/capi_dir.rs`): wwr-sys's own build.rs names the library, but its
//! rpath applies to its own targets only, so this crate's tests carry their
//! own.

include!("../../capi_dir.rs");

fn main() {
    let dir = capi_dir();
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
}
