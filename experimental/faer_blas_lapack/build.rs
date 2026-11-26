//! The LAPACK build links the image's LP64 OpenBLAS (`libopenblas-dev`),
//! which carries LAPACK and the `openblas_*` thread calls in one library.

fn main() {
    if std::env::var_os("CARGO_FEATURE_LAPACK").is_some() {
        println!("cargo:rustc-link-lib=dylib=openblas");
    }
}
