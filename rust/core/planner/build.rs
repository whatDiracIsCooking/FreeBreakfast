//! Point the link at WarpWraps' runtime_api C ABI library
//! (`rust/capi_dir.rs`), with the `device` feature only: without it nothing
//! in this crate's graph links the library, and the library need not exist.
//! wwr-sys's own build.rs names the library; its rpath applies to its own
//! targets only, so this crate's tests carry their own.

include!("../../capi_dir.rs");

fn main() {
    if std::env::var_os("CARGO_FEATURE_DEVICE").is_none() {
        return;
    }
    let dir = capi_dir();
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
}
