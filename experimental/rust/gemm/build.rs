//! Link the C ABI CMake builds from ffi/, by replaying the link line CMake
//! recorded for it.
//!
//! The C ABI is a static archive over a stack of others (the core and wwr
//! module libraries, spdlog, taskflow) plus the vendor runtime and BLAS and
//! libc++, and only CMake knows that whole set. Its probe executable,
//! free_breakfast_rust_gemm_linkprobe, links the archive through
//! ffi/record_link.sh, which writes the command to fb_gemm.link. This reads
//! the file and hands every argument but the probe's own (its object, its
//! output, its dependency file and the compile-only flags) to rustc's link.
//!
//! The file is looked for in FB_RUST_GEMM_FFI_DIR if set, else where the
//! `experimental` preset puts it: build-experimental/experimental/rust/gemm
//! under the checkout's root. The recorded rpath comes with it, so the demo
//! and the tests find the vendor libraries without LD_LIBRARY_PATH.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    println!("cargo:rerun-if-env-changed=FB_RUST_GEMM_FFI_DIR");
    let dir = env::var_os("FB_RUST_GEMM_FFI_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
            manifest.join("../../../build-experimental/experimental/rust/gemm")
        });
    let record = dir.join("fb_gemm.link");
    let text = fs::read_to_string(&record).unwrap_or_else(|e| {
        panic!(
            "cannot read {} ({e}): build the C ABI first, `cmake --build \
             build-experimental --target free_breakfast_rust_gemm_linkprobe`, \
             or point FB_RUST_GEMM_FFI_DIR at the directory holding it",
            record.display()
        )
    });
    println!("cargo:rerun-if-changed={}", record.display());

    let mut lines = text.lines();
    let cwd = PathBuf::from(lines.next().expect("fb_gemm.link is empty"));
    let _linker = lines.next().expect("fb_gemm.link names no linker");
    let mut args = lines.peekable();
    while let Some(arg) = args.next() {
        match arg {
            // The probe's output.
            "-o" => {
                args.next();
            }
            // The probe's dependency file; any other -Xlinker pair is kept.
            "-Xlinker"
                if args
                    .peek()
                    .is_some_and(|next| next.starts_with("--dependency-file=")) =>
            {
                args.next();
            }
            // Compile-only, or for the C++ driver only: rustc links through
            // cc with -nodefaultlibs, so libc++ is named outright below.
            "-O0" | "-O1" | "-O2" | "-O3" | "-Os" | "-DNDEBUG" | "-stdlib=libc++" => {}
            a if a.ends_with("linkprobe.cpp.o") => {}
            a => {
                let resolved = resolve(&cwd, a);
                if resolved.ends_with(".a") || resolved.ends_with(".so") {
                    println!("cargo:rerun-if-changed={resolved}");
                }
                println!("cargo:rustc-link-arg={resolved}");
            }
        }
    }
    println!("cargo:rustc-link-arg=-lc++");
}

/// A relative path that names a file under the link's working directory, made
/// absolute; anything else (a flag, an absolute path) as it is.
fn resolve(cwd: &Path, arg: &str) -> String {
    if !arg.starts_with('-') && !arg.starts_with('/') {
        let p = cwd.join(arg);
        if p.exists() {
            return p.display().to_string();
        }
    }
    arg.to_string()
}
