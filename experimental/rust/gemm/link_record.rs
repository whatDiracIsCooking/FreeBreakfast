// Replay a link line CMake recorded for a C ABI, into rustc's link. Shared by
// the crates under experimental/rust/: each build.rs `include!`s this file and
// calls `replay`. (A build-dependency crate would be the cargo way; one file is
// less machinery for two callers.)
//
// A C ABI here is a static archive over a stack of others (the core, wwr and
// pipeline module libraries, spdlog, taskflow) plus the vendor runtime and
// BLAS and libc++, and only CMake knows that whole set. Each has a probe
// executable that links the archive through gemm/ffi/record_link.sh, which
// writes the command to a `.link` file. `replay` reads it and hands every
// argument but the probe's own (its object, its output, its dependency file and
// the compile-only flags) to rustc. The recorded rpath comes with it, so
// binaries and tests find the vendor libraries without LD_LIBRARY_PATH.
//
// The file is looked for under FB_RUST_BUILD_DIR, the CMake build directory, if
// set, else under build-experimental at the checkout's root: what the
// `experimental` preset configures.

use std::env;
use std::fs;
use std::path::{Path, PathBuf};

/// Replay `<build>/experimental/rust/<subdir>/<record>`, which CMake target
/// `probe` writes.
fn replay(subdir: &str, record: &str, probe: &str) {
    println!("cargo:rerun-if-env-changed=FB_RUST_BUILD_DIR");
    let build = env::var_os("FB_RUST_BUILD_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
            manifest.join("../../../build-experimental")
        });
    let path = build.join("experimental/rust").join(subdir).join(record);
    let text = fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!(
            "cannot read {} ({e}): build the C ABI first, `cmake --build \
             build-experimental --target {probe}`, or point FB_RUST_BUILD_DIR \
             at the CMake build directory holding it",
            path.display()
        )
    });
    println!("cargo:rerun-if-changed={}", path.display());

    let mut lines = text.lines();
    let cwd = PathBuf::from(lines.next().expect("the link record is empty"));
    let _linker = lines.next().expect("the link record names no linker");
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
