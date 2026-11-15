//! Give the tests an rpath to WarpWraps' C ABI libraries (`rust/capi_dir.rs`)
//! and to `libfb_integrals_kernels.so` and `libfb_recipes_kernels.so`, which
//! the overlap the device test builds beside the guess links (its records
//! and classes, and its lower triangle's mirror): wwr-sys's and the `-sys`
//! crates' rpaths reach their own targets only.

include!("../../capi_dir.rs");

fn main() {
    if doc_only() {
        return;
    }
    let capi = capi_dir();
    // capi_dir() is <build>/deps/WarpWraps/src/capi; the kernel libraries
    // are at <build>/<module>/kernels (`rust/kernel_lib.rs`).
    let build = capi.ancestors().nth(4).unwrap();
    println!("cargo:rustc-link-search=native={}", capi.display());
    for d in [
        capi.clone(),
        build.join("integrals/kernels"),
        build.join("recipes/kernels"),
    ] {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", d.display());
    }
}
