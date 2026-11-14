//! Release aborts on a panic and dev and test unwind (`docs/diagnostics.md`,
//! "Panics and profiles"), as the workspace manifest says: its
//! `[profile.release]` sets `panic = "abort"`, and no other profile sets
//! `panic` at all. Cargo ignores `panic` for test targets, so a test profile
//! that asked for abort would be silently overridden; one that is not
//! written cannot mislead. Tier 3: it reads one file.

#![forbid(unsafe_code)]

/// Each `panic` key of `manifest`, with the table it is under.
fn panic_keys(manifest: &str) -> Vec<(String, String)> {
    let mut table = String::new();
    let mut out = Vec::new();
    for line in manifest.lines() {
        let t = line.split('#').next().unwrap().trim();
        if t.starts_with('[') {
            table = t.to_owned();
        } else if let Some((key, value)) = t.split_once('=')
            && key.trim() == "panic"
        {
            out.push((table.clone(), value.trim().to_owned()));
        }
    }
    out
}

#[test]
fn the_release_profile_alone_aborts_on_a_panic() {
    let manifest = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../Cargo.toml"))
        .expect("the workspace manifest");
    assert_eq!(
        panic_keys(&manifest),
        [("[profile.release]".to_owned(), "\"abort\"".to_owned())]
    );
}

#[test]
fn panic_keys_reads_tables_and_skips_comments() {
    let manifest = "[profile.dev]\n# panic = \"abort\"\nopt-level = 1\n\
                    [profile.release]\npanic = \"abort\" # why\n";
    assert_eq!(
        panic_keys(manifest),
        [("[profile.release]".to_owned(), "\"abort\"".to_owned())]
    );
}
