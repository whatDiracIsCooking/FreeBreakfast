//! Give the tests an rpath to WarpWraps' C ABI libraries (`rust/capi_dir.rs`)
//! and to `libfb_integrals_kernels.so`, which the overlap the device test
//! builds beside the guess links: wwr-sys's and fb-integrals-sys's rpaths
//! reach their own targets only.

include!("../../capi_dir.rs");

fn main() {
    let capi = capi_dir();
    // capi_dir() is <build>/deps/WarpWraps/src/capi; the kernel library is
    // at <build>/integrals/kernels (`rust/kernel_lib.rs`).
    let kernels = capi.ancestors().nth(4).unwrap().join("integrals/kernels");
    println!("cargo:rustc-link-search=native={}", capi.display());
    for d in [&capi, &kernels] {
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", d.display());
    }
}
