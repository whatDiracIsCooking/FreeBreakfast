//! What more than one module's unit tests use.

/// Compare `text` with the committed golden `tests/golden/<file>`, or
/// rewrite it when `FB_BLESS` is set (`docs/testing.md`, "Performance").
#[track_caller]
pub(crate) fn golden(file: &str, text: &str) {
    let path = format!("{}/tests/golden/{file}", env!("CARGO_MANIFEST_DIR"));
    if std::env::var_os("FB_BLESS").is_some() {
        std::fs::write(&path, text).expect("write the golden");
    }
    let golden = std::fs::read_to_string(&path).expect("the golden; FB_BLESS=1 writes it");
    assert_eq!(text, golden, "{file} changed; FB_BLESS=1 to accept");
}

/// Run `f`, which must panic, and return its message.
pub(crate) fn panic_message(f: impl FnOnce()) -> String {
    let e = std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)).unwrap_err();
    e.downcast_ref::<String>()
        .cloned()
        .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
        .unwrap()
}
