//! Three rules of the design docs that a scan of the Rust sources holds,
//! each failing with the file and line of what breaks it:
//!
//! - **Nothing between the detection site and `main` logs an error**
//!   (`docs/diagnostics.md`, "The one boundary", "`Drop` cannot fail"):
//!   every line logged at error or critical is in [`ERROR_LOGS`], which
//!   names the boundary, the destroys that cannot fail, and the abort tier,
//!   each with its count and its reason.
//! - **Rust never learns the warp width** (`docs/gpu_layer.md`, "Launch
//!   shape and dispatch"): no Rust source names it or a launch shape, in an
//!   identifier or in a string it passes on ([`LAUNCH_SHAPE`]).
//! - **`Allocation` is a value only at admission** (`docs/diagnostics.md`,
//!   "Allocation is a planning-time error"): an `Allocation` error is made
//!   only where [`ALLOCATIONS`] says, and the block's failed allocation
//!   makes one only to abort with it.
//!
//! Every member's `.rs` is read, and the two at `rust/`'s top that the
//! `build.rs` files include. Tier 3: it reads files and nothing else.

#![forbid(unsafe_code)]

use fb_doc_check::{code_only, is_ident, rust_root, without_comments, workspace_sources};

/// The places a line is logged at error or critical, as `(file, how many,
/// why)`. A file not listed may log none, and a listed one exactly its
/// count, so a second line in an allowed file is a new site to justify, and
/// an entry that matches nothing fails, so the list only shrinks.
const ERROR_LOGS: &[(&str, usize, &str)] = &[
    (
        "pipeline/config/src/main.rs",
        1,
        "the binary's `main`, the one boundary (docs/diagnostics.md, \"The one boundary\")",
    ),
    (
        "core/wwr/src/status.rs",
        1,
        "`gpu_log_error_at`, what a `Drop` does with a failed destroy (docs/diagnostics.md, \"`Drop` cannot fail\")",
    ),
    (
        "core/wwr/src/handle/base_handle.rs",
        1,
        "a handle's `Drop`, through `gpu_log_error_at` (docs/diagnostics.md, \"`Drop` cannot fail\")",
    ),
    (
        "core/wwr/src/blas.rs",
        1,
        "the BLAS handle's `Drop`, whose drain failed, through `gpu_log_error_at` (docs/diagnostics.md, \"`Drop` cannot fail\")",
    ),
    (
        "core/arena/src/kind/device.rs",
        2,
        "the device and pinned frees, from a block's `Drop`, through `gpu_log_error_at` (docs/diagnostics.md, \"`Drop` cannot fail\")",
    ),
    (
        "core/arena/src/block.rs",
        1,
        "a block's `Drop`, whose device could not be made current, frees anyway (docs/diagnostics.md, \"`Drop` cannot fail\")",
    ),
    (
        "core/error_handling/src/lib.rs",
        1,
        "`check_or_abort`, the abort tier, where detection and the process boundary are one call (docs/diagnostics.md, \"Levels\")",
    ),
    (
        "core/log/src/lib.rs",
        2,
        "the logger itself: the `critical!` macro and the panic hook (docs/diagnostics.md, \"Panics and profiles\")",
    ),
    (
        "core/diagnostics/tests/log.rs",
        4,
        "the logger's own test of each level",
    ),
];

/// Where an `Allocation` error may be made, as `(file, how many, why)`,
/// counted as [`ERROR_LOGS`] is. Test code (`tests/`, a `tests.rs`, a
/// `#[cfg(test)]` item) is not read: a test may make one to assert on.
/// `core/arena/src/block.rs` is not here: [`ABORTS`] holds it.
const ALLOCATIONS: &[(&str, usize, &str)] = &[
    (
        "core/arena/src/admit.rs",
        1,
        "admission refusing a plan, the one place the error is a value (docs/diagnostics.md, \"Allocation is a planning-time error\")",
    ),
    (
        "core/error_handling/src/lib.rs",
        1,
        "`Error::allocation`, the constructor admission calls",
    ),
    (
        "core/thread_pool/src/run.rs",
        1,
        "the timing record reading a kind back by name: an error already returned, not a new one",
    ),
];

/// The files whose `Allocation` errors are made only to abort with: each
/// must sit inside a `check_or_abort` call, with no statement between. The
/// block's one allocation at instantiation aborts when it fails, as every
/// create does (docs/diagnostics.md, "Allocation is a planning-time error").
const ABORTS: &[&str] = &["core/arena/src/block.rs"];

/// The names of a warp width or a launch shape, lowercased with `_`
/// removed, as a substring of an identifier or a word in a string. The
/// width is a configure-time define on the device side (`WWR_WARP_SIZE`,
/// `FJ_WAVEFRONT`, `kWarpSize`, `kFourWarps`), and the block and grid are
/// the wrapper's (CUDA's `blockDim`, `gridDim`, `dim3`, a block's
/// `threads_per_block`, `__launch_bounds__`).
const LAUNCH_SHAPE: &[&str] = &[
    "warpsize",
    "warpwidth",
    "wavefront",
    "fourwarps",
    "blockdim",
    "griddim",
    "dim3",
    "threadsperblock",
    "launchbounds",
];

/// The checks that name what they forbid: this file, and
/// `fb-device-check`'s scan of the C headers for a launch shape.
const CHECKS: &[&str] = &[
    "doc_check/tests/code_rules.rs",
    "device_check/tests/kernel_sources.rs",
];

/// Every member's `.rs`, and the ones at `rust/`'s top that every
/// `build.rs` includes (`capi_dir.rs`, `kernel_lib.rs`).
fn sources() -> Vec<(String, String)> {
    let mut out = workspace_sources();
    let root = rust_root();
    for entry in std::fs::read_dir(&root).unwrap() {
        let path = entry.unwrap().path();
        if path.is_file() && path.extension().is_some_and(|x| x == "rs") {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            out.push((name, std::fs::read_to_string(&path).unwrap()));
        }
    }
    // At least the arena's admission, so a walk that found nothing fails.
    assert!(
        out.iter().any(|(p, _)| p == "core/arena/src/admit.rs"),
        "no sources under {}",
        root.display()
    );
    out
}

/// The line of byte `offset` in `text`, from 1.
fn line_of(text: &str, offset: usize) -> usize {
    text[..offset].bytes().filter(|&c| c == b'\n').count() + 1
}

/// Each identifier in `text`, with its byte offset.
fn idents(text: &str) -> impl Iterator<Item = (usize, &str)> {
    let b = text.as_bytes();
    let mut i = 0;
    std::iter::from_fn(move || {
        while i < b.len() && !(is_ident(b[i]) && (i == 0 || !is_ident(b[i - 1]))) {
            i += 1;
        }
        if i >= b.len() {
            return None;
        }
        let start = i;
        while i < b.len() && is_ident(b[i]) {
            i += 1;
        }
        Some((start, &text[start..i]))
    })
}

/// The first byte after `offset` that is not whitespace.
fn next_char(text: &str, offset: usize) -> Option<(usize, u8)> {
    text.bytes()
        .enumerate()
        .skip(offset)
        .find(|(_, c)| !c.is_ascii_whitespace())
}

/// Whether the identifier before `offset` is `word`.
fn preceded_by(text: &str, offset: usize, word: &str) -> bool {
    text[..offset].trim_end().ends_with(word)
}

/// Each line `code` (from [`code_only`]) logs at error or critical, with
/// its line and the form it takes: `error!`, `critical!`, `log!` at either
/// level, `log_at` at either level, or `gpu_log_error_at`.
fn error_logs(code: &str) -> Vec<(usize, String)> {
    let mut found = Vec::new();
    for (at, word) in idents(code) {
        let end = at + word.len();
        let Some((q, next)) = next_char(code, end) else {
            continue;
        };
        let call = next == b'(' && !preceded_by(code, at, "fn");
        // The statement from the call on, for the level it names.
        let rest = &code[q..code[q..].find(';').map_or(code.len(), |e| q + e)];
        let first_arg = rest.split(',').next().unwrap_or("");
        let at_level = |text: &str| {
            idents(text).any(|(_, w)| w == "Error" || w == "Critical") && text.contains("Level")
        };
        let form = match word {
            "error" | "critical" if next == b'!' => Some(format!("{word}!")),
            "log" if next == b'!' && at_level(rest) => Some("log! at error".to_owned()),
            "log_at" if call && at_level(first_arg) => Some("log_at at error".to_owned()),
            "gpu_log_error_at" if call => Some("gpu_log_error_at".to_owned()),
            _ => None,
        };
        if let Some(form) = form {
            found.push((line_of(code, at), form));
        }
    }
    found
}

/// Each name of a warp width or a launch shape in `text` (from
/// [`without_comments`]), with its line.
fn launch_shape_names(text: &str) -> Vec<(usize, String)> {
    idents(text)
        .filter(|(_, w)| {
            let key = w.to_ascii_lowercase().replace('_', "");
            LAUNCH_SHAPE.iter().any(|n| key.contains(n))
        })
        .map(|(at, w)| (line_of(text, at), w.to_owned()))
        .collect()
}

/// `code` with every `#[cfg(test)]` item blanked, newlines kept.
fn without_test_items(code: &str) -> String {
    let mut out = code.to_owned().into_bytes();
    let mut from = 0;
    while let Some(at) = code[from..].find("#[cfg(test)]").map(|e| from + e) {
        from = at + 1;
        let Some(open) = code[at..].find(['{', ';']).map(|e| at + e) else {
            break;
        };
        if code.as_bytes()[open] == b';' {
            continue;
        }
        let mut depth = 0;
        let mut close = code.len();
        for (i, c) in code.bytes().enumerate().skip(open) {
            match c {
                b'{' => depth += 1,
                b'}' => {
                    depth -= 1;
                    if depth == 0 {
                        close = i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        for c in &mut out[at..close] {
            if *c != b'\n' {
                *c = b' ';
            }
        }
        from = close;
    }
    String::from_utf8(out).unwrap_or_default()
}

/// Whether `path` is test code as a whole: under a `tests/` directory, or a
/// `tests.rs` module.
fn is_test_file(path: &str) -> bool {
    path.starts_with("tests/") || path.contains("/tests/") || path.ends_with("/tests.rs")
}

/// Each `Allocation` error `code` (from [`code_only`], test items removed)
/// makes, as `(offset, line)`: `Error::allocation` or `ErrorKind::Allocation`.
fn allocations(code: &str) -> Vec<(usize, usize)> {
    let mut found = Vec::new();
    for pattern in ["ErrorKind::Allocation", "Error::allocation"] {
        let mut from = 0;
        while let Some(at) = code[from..].find(pattern).map(|e| from + e) {
            from = at + pattern.len();
            let bounded = (at == 0 || !is_ident(code.as_bytes()[at - 1]))
                && code.as_bytes().get(from).is_none_or(|&c| !is_ident(c));
            if bounded {
                found.push((at, line_of(code, at)));
            }
        }
    }
    found.sort_unstable();
    found
}

/// Whether the error at `offset` is inside a `check_or_abort` call: the
/// text since the last one has no `;`, so it is the call's argument.
fn inside_abort(code: &str, offset: usize) -> bool {
    code[..offset]
        .rfind("check_or_abort")
        .is_some_and(|c| !code[c..offset].contains(';'))
}

/// The failures of `hits`, a file's count of something, against `allowed`,
/// each `(file, count, why)`: a file not listed with any hit, a listed one
/// whose count differs, and an entry whose file has none.
fn against(
    rule: &str,
    hits: &[(String, usize, String)],
    allowed: &[(&str, usize, &str)],
) -> Vec<String> {
    let mut failures = Vec::new();
    let mut files: Vec<&str> = hits.iter().map(|(f, _, _)| f.as_str()).collect();
    files.dedup();
    for file in &files {
        let here: Vec<&(String, usize, String)> =
            hits.iter().filter(|(f, _, _)| f == file).collect();
        let lines = here
            .iter()
            .map(|(f, l, what)| format!("  {f}:{l}: {what}"))
            .collect::<Vec<_>>()
            .join("\n");
        match allowed.iter().find(|(f, _, _)| f == file) {
            None => failures.push(format!("{rule}, and {file} is not allowed any:\n{lines}")),
            Some((_, n, why)) if *n != here.len() => failures.push(format!(
                "{rule}; {file} is allowed {n} ({why}) and has {}:\n{lines}",
                here.len()
            )),
            Some(_) => {}
        }
    }
    for (file, n, _) in allowed {
        if !files.contains(file) {
            failures.push(format!(
                "{file}: allowed {n} but has none; drop its entry ({rule})"
            ));
        }
    }
    failures
}

fn error_log_failures(files: &[(String, String)]) -> Vec<String> {
    let hits: Vec<(String, usize, String)> = files
        .iter()
        .filter(|(p, _)| !CHECKS.contains(&p.as_str()))
        .flat_map(|(p, src)| {
            error_logs(&code_only(src))
                .into_iter()
                .map(move |(l, what)| (p.clone(), l, what))
        })
        .collect();
    against(
        "an error is logged once, at the boundary: propagate it as a value and \
         let `main` log it (docs/diagnostics.md, \"The one boundary\")",
        &hits,
        ERROR_LOGS,
    )
}

fn launch_shape_failures(files: &[(String, String)]) -> Vec<String> {
    let mut failures = Vec::new();
    for (path, src) in files.iter().filter(|(p, _)| !CHECKS.contains(&p.as_str())) {
        for (line, name) in launch_shape_names(&without_comments(src)) {
            failures.push(format!(
                "{path}:{line}: `{name}`: Rust passes dimensions, never a block size, and \
                 never learns the warp width; the wrapper picks the launch shape \
                 (docs/gpu_layer.md, \"Launch shape and dispatch\")"
            ));
        }
    }
    failures
}

fn allocation_failures(files: &[(String, String)]) -> Vec<String> {
    let mut hits = Vec::new();
    let mut failures = Vec::new();
    for (path, src) in files
        .iter()
        .filter(|(p, _)| !is_test_file(p) && !CHECKS.contains(&p.as_str()))
    {
        let code = without_test_items(&code_only(src));
        for (at, line) in allocations(&code) {
            if !ABORTS.contains(&path.as_str()) {
                hits.push((path.clone(), line, "an `Allocation` error".to_owned()));
            } else if !inside_abort(&code, at) {
                failures.push(format!(
                    "{path}:{line}: an `Allocation` error outside `check_or_abort`: the \
                     block's allocation aborts, it does not return the error \
                     (docs/diagnostics.md, \"Allocation is a planning-time error\")"
                ));
            }
        }
    }
    failures.extend(against(
        "`Allocation` is a value only at admission (docs/diagnostics.md, \
         \"Allocation is a planning-time error\")",
        &hits,
        ALLOCATIONS,
    ));
    failures
}

#[test]
fn nothing_but_the_boundary_logs_an_error() {
    let failures = error_log_failures(&sources());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn no_rust_source_names_a_warp_width_or_a_launch_shape() {
    let failures = launch_shape_failures(&sources());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn allocation_is_a_value_only_at_admission() {
    let failures = allocation_failures(&sources());
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The scan sees every form of an error line, and not a `use`, a comment,
/// a string, a definition, or a lower level.
#[test]
fn the_error_scan_sees_each_form() {
    let src = "\
        use log::error;\n\
        log::error!(\"x\");\n\
        .inspect_err(|e| error!(\"{e}\"))\n\
        critical!(\"y\");\n\
        log::log!(log::Level::Error, \"z\");\n\
        log_at(LogLevel::Error, here, args);\n\
        log::log_at(LogLevel::Critical, here, args);\n\
        gpu_log_error_at(status, here);\n\
        pub fn gpu_log_error_at(code: E) {}\n\
        pub fn log_at(level: LogLevel) {}\n\
        log_at(LogLevel::Warn, here, format_args!(\"Error\"));\n\
        log::warn!(\"error!(no)\");\n\
        // error!(\"in a comment\")\n";
    let got: Vec<(usize, String)> = error_logs(&code_only(src));
    let want: Vec<(usize, String)> = [
        (2, "error!"),
        (3, "error!"),
        (4, "critical!"),
        (5, "log! at error"),
        (6, "log_at at error"),
        (7, "log_at at error"),
        (8, "gpu_log_error_at"),
    ]
    .into_iter()
    .map(|(l, w)| (l, w.to_owned()))
    .collect();
    assert_eq!(got, want);
}

/// A site in an unlisted file, one too many in a listed file, and an entry
/// with nothing left each fail, naming the file and line.
#[test]
fn error_logs_are_held_to_their_counts() {
    let files = vec![
        (
            "pipeline/config/src/main.rs".to_owned(),
            "log::log_at(LogLevel::Error, o, a);".to_owned(),
        ),
        (
            "core/arena/src/block.rs".to_owned(),
            "error!(\"a\");\nerror!(\"b\");".to_owned(),
        ),
        (
            "recipes/blas/src/lib.rs".to_owned(),
            "fn f() {\n    log::error!(\"x\");\n}".to_owned(),
        ),
    ];
    let failures = error_log_failures(&files);
    let all = failures.join("\n");
    assert!(
        all.contains(
            "recipes/blas/src/lib.rs is not allowed any:\n  recipes/blas/src/lib.rs:2: error!"
        ),
        "{all}"
    );
    assert!(
        all.contains("core/arena/src/block.rs is allowed 1") && all.contains("has 2"),
        "{all}"
    );
    assert!(
        all.contains("core/log/src/lib.rs: allowed 2 but has none"),
        "{all}"
    );
    assert!(!all.contains("main.rs"), "{all}");
}

/// The width in an identifier or a string is seen; `WarpWraps`, an arena
/// `Block` and a comment are not.
#[test]
fn the_launch_shape_scan_sees_names_and_strings() {
    let src = "\
        let n = wwr_sys::WWR_DEV_ATTR_WARP_SIZE;\n\
        cmd.arg(\"-DFJ_WAVEFRONT=64\");\n\
        fn launch(block_dim: u32, threads_per_block: u32) {}\n\
        const K_FOUR_WARPS: u32 = 128;\n\
        // the warp_size is the wrapper's\n\
        let p = \"deps/WarpWraps/src/capi\";\n\
        let b = Block::new(0, table);\n";
    let got: Vec<(usize, String)> = launch_shape_names(&without_comments(src));
    let want: Vec<(usize, String)> = [
        (1, "WWR_DEV_ATTR_WARP_SIZE"),
        (2, "DFJ_WAVEFRONT"),
        (3, "block_dim"),
        (3, "threads_per_block"),
        (4, "K_FOUR_WARPS"),
    ]
    .into_iter()
    .map(|(l, w)| (l, w.to_owned()))
    .collect();
    assert_eq!(got, want);
}

/// An `Allocation` made outside admission fails, one in a test item or a
/// comparison of a kind read elsewhere does not, and the block's must be an
/// abort's argument.
#[test]
fn allocations_outside_admission_fail() {
    let files = vec![
        (
            "core/arena/src/admit.rs".to_owned(),
            "Err(Error::allocation(m))".to_owned(),
        ),
        (
            "core/error_handling/src/lib.rs".to_owned(),
            "Self::at_caller(ErrorKind::Allocation, m)\n\
             #[cfg(test)]\nmod tests {\n    fn t() { Error::allocation(\"m\"); }\n}"
                .to_owned(),
        ),
        (
            "core/thread_pool/src/run.rs".to_owned(),
            "const KINDS: [ErrorKind; 1] = [ErrorKind::Allocation];".to_owned(),
        ),
        (
            "core/arena/src/block.rs".to_owned(),
            "check_or_abort::<()>(Err(Error {\n    kind: ErrorKind::Allocation,\n}));\n\
             return Err(Error::allocation(m));"
                .to_owned(),
        ),
        (
            "core/planner/src/lib.rs".to_owned(),
            "fn f() -> Status {\n    Err(Error::allocation(\"late\"))\n}".to_owned(),
        ),
        (
            "core/planner/tests/t.rs".to_owned(),
            "Err(Error::allocation(\"a test\"))".to_owned(),
        ),
    ];
    let failures = allocation_failures(&files);
    assert_eq!(failures.len(), 2, "{failures:#?}");
    assert!(
        failures[0].starts_with("core/arena/src/block.rs:4: an `Allocation` error outside"),
        "{failures:#?}"
    );
    assert!(
        failures[1]
            .contains("core/planner/src/lib.rs is not allowed any:\n  core/planner/src/lib.rs:2:"),
        "{failures:#?}"
    );
}
