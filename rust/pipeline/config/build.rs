//! Give the binary and the tests an rpath to WarpWraps' C ABI libraries
//! (`rust/capi_dir.rs`) and to both kernel libraries a plan reaches,
//! `libfb_integrals_kernels.so` (the integrals) and
//! `libfb_recipes_kernels.so` (the orthogonalizer, the Cholesky factor,
//! DIIS): wwr-sys's and the `-sys` crates' rpaths reach their own targets
//! only, and `free-breakfast` is run from wherever cargo put it.

include!("../../capi_dir.rs");

fn main() {
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
