//! Every kernel library's committed `-sys` bindings, and the shared
//! vocabulary's in `fb-kernel-abi`, are what `devtools/rust-bindings.py`
//! generates from its C header now (`docs/gpu_layer.md`, "Build", "From
//! bridge to C ABI"): a header edited without regenerating fails here, with
//! the diff. Tier 3: it reads files and runs the script, which needs
//! python3 and nothing else.

use std::path::Path;
use std::process::Command;

#[test]
fn committed_bindings_match_their_headers() {
    let repo = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = Command::new("python3")
        .arg(repo.join("devtools/rust-bindings.py"))
        .arg("--check")
        .output()
        .expect("python3 runs devtools/rust-bindings.py");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(out.status.success(), "{stdout}{stderr}");
    // The shared header's home and at least the recipes library, so a glob
    // that found nothing, or a shared header dropped from the script, fails.
    for file in [
        "rust/kernel_abi/src/bindings.rs",
        "rust/recipes/sys/src/bindings.rs",
    ] {
        assert!(stdout.contains(&format!("{file} is current")), "{stdout}");
    }
}

/// The names `bindings` defines at its top level: each `pub const`,
/// `pub type` and `pub struct`.
fn defined(bindings: &str) -> Vec<&str> {
    bindings
        .lines()
        .filter_map(|l| {
            ["pub const ", "pub type ", "pub struct "]
                .iter()
                .find_map(|p| l.strip_prefix(p))
        })
        .map(|rest| rest.split([':', ' ', '=']).next().unwrap())
        .collect()
}

/// `fb_launch.h` is bound once, in `fb-kernel-abi`: no `-sys` crate's
/// bindings define a name of it again, and each re-exports every one, so a
/// shared header the script inlined into a module, or a `-sys` crate that
/// lost one of the names its tests write, fails.
#[test]
fn the_shared_vocabulary_is_bound_once() {
    let rust = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let shared = std::fs::read_to_string(rust.join("kernel_abi/src/bindings.rs")).unwrap();
    let names = defined(&shared);
    assert!(names.contains(&"FB_LAUNCH_SUCCESS"), "{names:?}");
    let mut libraries = 0;
    for entry in std::fs::read_dir(&rust).unwrap() {
        let file = entry.unwrap().path().join("sys/src/bindings.rs");
        let Ok(text) = std::fs::read_to_string(&file) else {
            continue;
        };
        libraries += 1;
        let own = defined(&text);
        let reexport = text
            .split_once("pub use fb_kernel_abi::{")
            .and_then(|(_, rest)| rest.split_once("};"))
            .map(|(list, _)| list)
            .unwrap_or_else(|| panic!("{}: no pub use of fb_kernel_abi", file.display()));
        let reexported: Vec<&str> = reexport.split(',').map(str::trim).collect();
        for name in &names {
            assert!(
                !own.contains(name),
                "{} defines {name} again",
                file.display()
            );
            assert!(
                reexported.contains(name),
                "{} does not re-export {name}",
                file.display()
            );
        }
    }
    assert!(libraries >= 2, "found {libraries} -sys bindings");
}
