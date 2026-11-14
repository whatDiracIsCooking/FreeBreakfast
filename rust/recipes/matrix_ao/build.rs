//! Point the link at WarpWraps' C ABI libraries (`rust/capi_dir.rs`), and
//! give the tests an rpath to them, `libwwr_capi_blas.so` among them:
//! wwr-sys's own rpath reaches its own targets only.

include!("../../capi_dir.rs");

fn main() {
    let dir = capi_dir();
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
}
