//! Every kernel library's committed `-sys` bindings are what
//! `devtools/rust-bindings.py` generates from its C header now
//! (`docs/gpu_layer.md`, "Build"): a header edited without regenerating
//! fails here, with the diff. Tier 3: it reads files and runs the script,
//! which needs python3 and nothing else.

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
    // At least the recipes library, so a glob that found nothing fails.
    assert!(
        stdout.contains("rust/recipes/sys/src/bindings.rs is current"),
        "{stdout}"
    );
}
