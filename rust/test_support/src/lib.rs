//! What more than one crate's unit tests use: a dev-dependency only.

#![forbid(unsafe_code)]

/// Compare `text` with the committed golden `<manifest_dir>/tests/golden/<file>`,
/// or rewrite it when `FB_BLESS` is set (`docs/testing.md`, "Performance").
/// `manifest_dir` is the calling crate's `env!("CARGO_MANIFEST_DIR")`, so the
/// golden lives beside the tests that read it.
///
/// # Panics
/// If `text` differs from the golden, or the golden cannot be read (or,
/// under `FB_BLESS`, written).
#[track_caller]
pub fn golden(manifest_dir: &str, file: &str, text: &str) {
    let path = format!("{manifest_dir}/tests/golden/{file}");
    if std::env::var_os("FB_BLESS").is_some() {
        std::fs::write(&path, text).expect("write the golden");
    }
    let golden = std::fs::read_to_string(&path).expect("the golden; FB_BLESS=1 writes it");
    assert_eq!(text, golden, "{file} changed; FB_BLESS=1 to accept");
}

/// Run `f`, which must panic, and return its message.
///
/// # Panics
/// If `f` returns, or panics with a payload that is not a string.
pub fn panic_message(f: impl FnOnce()) -> String {
    let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_err();
    e.downcast_ref::<String>()
        .cloned()
        .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap()
}
