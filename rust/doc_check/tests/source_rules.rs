//! Eight rules of the register (`docs/rules.md`) that a scan of the sources
//! holds, each failing with the file and line of every departure:
//!
//! - **`#[ignore]` is a device case's, and says so**: every `ignore` in an
//!   attribute is `ignore = "tier 4: needs a GPU"` (`docs/testing.md`,
//!   "Selecting GPU tiers in cargo").
//! - **No Markdown beside a module**: the only `.md` under `rust/` are the
//!   workspace's `README.md` and `CLAUDE.md`, the design documents under
//!   `docs/`, a crate's front page that its `src/lib.rs` includes, and a
//!   golden's provenance, a `README.md` in a `tests/data/` directory
//!   (`docs/design.md`, "Where a design lives"; `docs/documentation.md`,
//!   "Code: rustdoc"; `docs/testing.md`, "Correctness").
//! - **A doc example's marker**: no Rust block in a doc comment is
//!   `ignore`, a `compile_fail` names its error code, and a block that runs
//!   calls no device type's associated function and no binding (`docs/documentation.md`,
//!   "Code: rustdoc").
//! - **Nothing crosses from Rust into C++**: no `no_mangle` or
//!   `export_name`, no `pub` `extern` function, and no `cdylib` or
//!   `staticlib` crate (`docs/gpu_layer.md`, "What the port changes").
//! - **The safe wrapper links down**: in a binding crate (a member whose
//!   `src/bindings.rs` declares functions), every `pub fn` that calls a
//!   binding links it in its doc comment (`docs/documentation.md`, "The C header").
//! - **One BLAS operation, one name**: no public function or `pub use`
//!   of `fb-recipes-blas` (`recipes/blas/src/`) shares a name with a
//!   free function of `fb_core_wwr::blas` (`core/wwr/src/blas.rs`), and
//!   no `pub use` there re-exports a glob (`docs/gpu_layer.md`,
//!   "Layering").
//! - **A test that trips a contract is debug-only exactly when the
//!   contract is**: a `#[should_panic(expected = …)]` whose text is a
//!   `#[debug_requires]`, `#[debug_ensures]` or `#[debug_invariant]`
//!   message is under `#[cfg(debug_assertions)]`, and one whose text is only
//!   an always-on `#[requires]`, `#[ensures]` or `#[invariant]` message is
//!   not (`docs/testing.md`, "Contracts are part of the spec").
//! - **A DAG builder is `add_*(dag, worker, …)`**: outside `core/` and test
//!   code, a public function that takes a `&mut Dag` is named `add_*`,
//!   takes `dag` first after any `self`, and `worker`, when it has one,
//!   right after it (`docs/programming_model.md`, "A builder adds tasks").
//!
//! Each is a text scan, so each sees only the spelling it looks for; the
//! register's row says what that misses. Tier 3: it reads files and
//! nothing else.

#![forbid(unsafe_code)]

use fb_doc_check::{
    Attr, attributes, code_only, is_ident, line_of, rust_root, string_literals, word_at, words,
    workspace_sources,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;

/// The one reason an `#[ignore]` may give.
const DEVICE_REASON: &str = "tier 4: needs a GPU";
/// The contract attributes compiled out of a release build.
const DEBUG_CONTRACTS: [&str; 3] = ["debug_requires", "debug_ensures", "debug_invariant"];
/// The contract attributes checked in every build.
const ALWAYS_CONTRACTS: [&str; 3] = ["requires", "ensures", "invariant"];
/// The types whose associated functions, called in a doc example
/// (`Stream::new(…)`), mean it reaches a device: the handle, stream, event
/// and arena types of `fb-core-wwr`, `fb-core-arena` and
/// `fb-core-thread-pool`. A binding call (`…_sys::name(`) counts as well.
const DEVICE_NAMES: [&str; 10] = [
    "Block",
    "Device",
    "Stream",
    "DeviceStream",
    "Event",
    "BlasHandle",
    "SolverHandle",
    "DeviceBytes",
    "MemPool",
    "ScopedDeviceIndex",
];
/// The rustdoc attributes a Rust code block's info string may carry,
/// besides an error code; any other word makes the block not Rust.
const RUST_FENCE_WORDS: [&str; 10] = [
    "rust",
    "ignore",
    "no_run",
    "compile_fail",
    "should_panic",
    "test_harness",
    "standalone_crate",
    "edition2018",
    "edition2021",
    "edition2024",
];

/// Whether an attribute, as written, makes what it is on debug-only:
/// a `cfg` naming `debug_assertions` and not negating it.
fn debug_only(source: &str, a: &Attr) -> bool {
    let text: String = source[a.start..a.end]
        .chars()
        .filter(|c| !c.is_whitespace())
        .collect();
    a.name == "cfg" && text.contains("debug_assertions") && !text.contains("not(debug_assertions")
}

// ---------------------------------------------------------------- ignore

/// Every `ignore` in an attribute of one file that is not
/// `ignore = "tier 4: needs a GPU"`, `cfg_attr` included.
fn bad_ignores(path: &str, source: &str) -> Vec<String> {
    let code = code_only(source);
    let mut failures = Vec::new();
    for a in attributes(&code) {
        for (at, w) in words(&code[a.start..a.end]) {
            if w != "ignore" {
                continue;
            }
            let after: String = source[a.start + at + w.len()..a.end]
                .chars()
                .filter(|c| !c.is_whitespace())
                .collect();
            let reason = format!("=\"{}\"", DEVICE_REASON.replace(' ', ""));
            if !after.starts_with(&reason) {
                failures.push(format!(
                    "{path}:{}: an ignore without the reason `{DEVICE_REASON}`; \
                     #[ignore] means a device case and nothing else",
                    line_of(source, a.start + at)
                ));
            }
        }
    }
    failures
}

#[test]
fn every_ignore_is_a_device_case() {
    let sources = workspace_sources();
    assert!(sources.len() > 50, "only {} sources found", sources.len());
    let failures: Vec<String> = sources
        .iter()
        .flat_map(|(p, s)| bad_ignores(p, s))
        .collect();
    assert!(
        failures.is_empty(),
        "{} ignore(s) that are not device cases:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn an_ignore_needs_the_device_reason() {
    let good = "#[test]\n#[ignore = \"tier 4: needs a GPU\"]\nfn f() {}\n\
                #[cfg_attr(not(x), ignore = \"tier 4:  needs a GPU\")]\nfn g() {}\n";
    assert_eq!(bad_ignores("a.rs", good), Vec::<String>::new());
    let bad = "#[ignore]\nfn f() {}\n#[ignore = \"slow\"]\nfn g() {}\n\
               #[cfg_attr(x, ignore)]\nfn h() {}\n";
    let f = bad_ignores("a.rs", bad);
    assert_eq!(f.len(), 3, "{f:#?}");
    assert!(
        f[0].starts_with("a.rs:1:") && f[2].starts_with("a.rs:5:"),
        "{f:#?}"
    );
    // In a comment or a string it is no attribute.
    let not_attrs = "// #[ignore]\nlet s = \"#[ignore]\";\n";
    assert_eq!(bad_ignores("a.rs", not_attrs), Vec::<String>::new());
}

// -------------------------------------------------------------- markdown

/// Whether a Markdown file, by its path from `rust/`, is one the docs
/// allow; `front_page` says whether its crate's `src/lib.rs` includes it.
fn markdown_allowed(rel: &str, front_page: bool) -> bool {
    let parts: Vec<&str> = rel.split('/').collect();
    match parts.as_slice() {
        ["README.md" | "CLAUDE.md"] | ["docs", ..] => true,
        [.., "tests", "data", "README.md"] => true,
        [.., "README.md"] => front_page,
        _ => false,
    }
}

/// Every `.md` under `dir`, by its path from `root`, skipping `target/`,
/// the CMake build trees and hidden directories.
fn markdown_files(root: &Path, dir: &Path, out: &mut BTreeSet<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries {
        let p = e.unwrap().path();
        let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if p.is_dir() {
            if name != "target" && !name.starts_with("build-") && !name.starts_with('.') {
                markdown_files(root, &p, out);
            }
        } else if p.extension().is_some_and(|x| x.eq_ignore_ascii_case("md")) {
            let rel = p.strip_prefix(root).unwrap_or(&p);
            out.insert(rel.display().to_string());
        }
    }
}

/// The Markdown files under `root` the docs do not allow.
fn stray_markdown(root: &Path) -> Vec<String> {
    let mut files = BTreeSet::new();
    markdown_files(root, root, &mut files);
    files
        .into_iter()
        .filter(|rel| {
            let dir = Path::new(rel).parent().unwrap_or(Path::new(""));
            let lib = std::fs::read_to_string(root.join(dir).join("src/lib.rs"));
            let front = lib.is_ok_and(|l| l.contains("include_str!(\"../README.md\")"));
            !markdown_allowed(rel, front)
        })
        .map(|rel| {
            format!(
                "{rel}:1: Markdown beside a module; the doc comment is the spec \
                 (a crate's README is allowed only as the front page its src/lib.rs includes)"
            )
        })
        .collect()
}

#[test]
fn no_markdown_sits_beside_a_module() {
    let failures = stray_markdown(&rust_root());
    assert!(
        failures.is_empty(),
        "{} Markdown file(s) beside a module:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn only_the_allowed_markdown_passes() {
    for ok in [
        "README.md",
        "CLAUDE.md",
        "docs/memory.md",
        "integrals/mmd/tests/data/README.md",
    ] {
        assert!(markdown_allowed(ok, false), "{ok}");
    }
    assert!(markdown_allowed("sad/README.md", true));
    for bad in [
        "sad/README.md",
        "sad/src/README.md",
        "core/arena/DESIGN.md",
        "core/arena/src/block.md",
        "integrals/mmd/tests/notes.md",
    ] {
        assert!(!markdown_allowed(bad, false), "{bad}");
    }
}

// ----------------------------------------------------------- doc examples

/// The departures of the doc examples of one file: an `ignore` block, a
/// `compile_fail` with no error code, and a block that runs but calls a
/// device type's associated function or a binding.
fn bad_examples(path: &str, source: &str) -> Vec<String> {
    let mut failures = Vec::new();
    // The open block: its fence line, its info words, and its code so far.
    let mut open: Option<(usize, Vec<String>, String)> = None;
    for (n, line) in source.lines().enumerate() {
        let t = line.trim_start();
        let doc = match t.strip_prefix("///").or_else(|| t.strip_prefix("//!")) {
            Some(d) if !t.starts_with("////") => d,
            _ => {
                open = None;
                continue;
            }
        };
        let body = doc.strip_prefix(' ').unwrap_or(doc);
        let Some(info) = body.trim_start().strip_prefix("```") else {
            if let Some((_, _, code)) = &mut open {
                code.push_str(body);
                code.push('\n');
            }
            continue;
        };
        if let Some((at, info, code)) = open.take() {
            failures.extend(judge(path, at, &info, &code));
            continue;
        }
        let info: Vec<String> = info
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|w| !w.is_empty())
            .map(str::to_string)
            .collect();
        open = Some((n + 1, info, String::new()));
    }
    failures
}

/// Whether `w` is a rustdoc error code, `E` and four digits.
fn error_code(w: &str) -> bool {
    w.len() == 5 && w.starts_with('E') && w[1..].bytes().all(|c| c.is_ascii_digit())
}

/// The departure of one closed doc block, if it is Rust and has one.
fn judge(path: &str, at: usize, info: &[String], code: &str) -> Option<String> {
    let rust = info.iter().all(|w| {
        RUST_FENCE_WORDS.contains(&w.as_str()) || error_code(w) || w.starts_with("ignore-")
    });
    if !rust {
        return None;
    }
    let has = |m: &str| info.iter().any(|w| w == m);
    if info.iter().any(|w| w.starts_with("ignore")) {
        return Some(format!(
            "{path}:{at}: an `ignore` doc example; an example compiles (`no_run` for a device), \
             and a block that is not a test is fenced `text`"
        ));
    }
    if has("compile_fail") {
        return (!info.iter().any(|w| error_code(w))).then(|| {
            format!(
                "{path}:{at}: a `compile_fail` example with no error code (`compile_fail,E0277`)"
            )
        });
    }
    if has("no_run") {
        return None;
    }
    let code = code_only(code);
    let ws = words(&code);
    // A device type made or called through (`Stream::new(`, at the end of
    // any path), not one named as a type or a variant (`Placement::Stream(0)`).
    let named = ws.iter().find_map(|(i, w)| {
        let head = DEVICE_NAMES.contains(w);
        let rest = code[i + w.len()..].trim_start();
        let call = rest.strip_prefix("::").is_some_and(|r| {
            let r = r.trim_start();
            let name = r.bytes().take_while(|&c| is_ident(c)).count();
            name > 0 && r[name..].trim_start().starts_with('(')
        });
        (head && call).then_some(*w)
    });
    let called = ws.iter().enumerate().find_map(|(k, (i, w))| {
        let sys = k > 0 && ws[k - 1].1.ends_with("_sys") && code[..*i].ends_with("::");
        (sys && code[i + w.len()..].trim_start().starts_with('(')).then_some(*w)
    });
    named.or(called).map(|what| {
        format!("{path}:{at}: a doc example that runs calls through `{what}`; a device example is `no_run`")
    })
}

#[test]
fn every_doc_example_carries_its_marker() {
    let failures: Vec<String> = workspace_sources()
        .iter()
        .flat_map(|(p, s)| bad_examples(p, s))
        .collect();
    assert!(
        failures.is_empty(),
        "{} doc example(s) with the wrong marker:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn a_doc_example_marker_is_judged_by_its_block() {
    // Each fixture line opens with a quote, so no line of this file is a
    // doc comment holding the fence it shows.
    let clean = concat!(
        "/// ```\n/// let x = 1;\n/// ```\n//! ```compile_fail,E0277\n//! x\n//! ```\n",
        "/// ```no_run\n/// let s = Stream::new();\n/// ```\n/// ```text\n/// ignore\n/// ```\n",
        "/// ```toml\n/// a = 1\n/// ```\nfn f() {}\n",
        "/// ```\n/// fn f(_: Lane<'_, DeviceStream>, p: Placement) {}\n",
        "/// f(lane, Placement::Stream(0));\n/// let b: Option<Block<Device>> = None;\n/// ```\n",
    );
    assert_eq!(bad_examples("a.rs", clean), Vec::<String>::new());
    let bad = concat!(
        "/// ```ignore\n/// x\n/// ```\n/// ```compile_fail\n/// x\n/// ```\n",
        "/// ```rust\n/// let b = Block::new(0, t);\n/// ```\n",
        "/// ```\n/// unsafe { fb_recipes_sys::fb_inverse_sqrt(s, 0, p, 0.0) };\n/// ```\n",
    );
    let f = bad_examples("a.rs", bad);
    assert_eq!(f.len(), 4, "{f:#?}");
    assert!(
        f[0].starts_with("a.rs:1:") && f[0].contains("`ignore`"),
        "{f:#?}"
    );
    assert!(
        f[1].starts_with("a.rs:4:") && f[1].contains("error code"),
        "{f:#?}"
    );
    assert!(
        f[2].starts_with("a.rs:7:") && f[2].contains("`Block`"),
        "{f:#?}"
    );
    assert!(
        f[3].starts_with("a.rs:10:") && f[3].contains("`fb_inverse_sqrt`"),
        "{f:#?}"
    );
    // A device name in a comment of the example, or a binding type, runs.
    let host = "/// ```\n/// // a Block would need a card\n/// let x: fb_recipes_sys::fb_backend = 1;\n/// ```\n";
    assert_eq!(bad_examples("a.rs", host), Vec::<String>::new());
}

// ------------------------------------------------------------ exports to C

/// The words that may stand between a visibility and `extern`.
const QUALIFIERS: [&str; 3] = ["unsafe", "const", "async"];

/// What exports a symbol to C in one file: a `no_mangle` or `export_name`
/// attribute, and a `pub` `extern` function.
fn exports(path: &str, source: &str) -> Vec<String> {
    let code = code_only(source);
    let mut failures = Vec::new();
    for a in attributes(&code) {
        for (at, w) in words(&code[a.start..a.end]) {
            if w == "no_mangle" || w == "export_name" {
                failures.push(format!(
                    "{path}:{}: #[{w}] exports a symbol to C; nothing crosses from Rust back into C++",
                    line_of(source, a.start + at)
                ));
            }
        }
    }
    let ws = words(&code);
    for (k, (i, w)) in ws.iter().enumerate() {
        if *w != "extern" || ws.get(k + 1).is_none_or(|(_, n)| *n != "fn") {
            continue;
        }
        // Back over the qualifiers, then a visibility's `(…)`, to `pub`.
        let mut before = code[..*i].trim_end();
        while let Some(q) = QUALIFIERS.iter().find(|q| {
            before.ends_with(**q)
                && !before[..before.len() - q.len()].ends_with(|c: char| is_ident(c as u8))
        }) {
            before = before[..before.len() - q.len()].trim_end();
        }
        if before.ends_with(')')
            && let Some(p) = before.rfind('(')
        {
            before = before[..p].trim_end();
        }
        let public = before.ends_with("pub")
            && !before[..before.len() - 3].ends_with(|c: char| is_ident(c as u8));
        if public {
            failures.push(format!(
                "{path}:{}: a pub extern fn is a C entry point; nothing crosses from Rust back \
                 into C++ (a callback a C API takes is a private extern fn)",
                line_of(source, *i)
            ));
        }
    }
    failures
}

/// A member manifest that builds a library C can link.
fn c_library(path: &str, manifest: &str) -> Option<String> {
    let line = manifest.lines().position(|l| {
        let l = l.split('#').next().unwrap_or("");
        l.contains("crate-type") && (l.contains("cdylib") || l.contains("staticlib"))
    })?;
    Some(format!(
        "{path}:{}: a cdylib or staticlib is a library C++ can link; nothing crosses from Rust \
         back into C++",
        line + 1
    ))
}

#[test]
fn nothing_is_exported_to_c() {
    let root = rust_root();
    let mut failures: Vec<String> = workspace_sources()
        .iter()
        .flat_map(|(p, s)| exports(p, s))
        .collect();
    for (path, _) in workspace_sources()
        .iter()
        .filter(|(p, _)| p.ends_with("/build.rs"))
    {
        let manifest = Path::new(path).with_file_name("Cargo.toml");
        let text = std::fs::read_to_string(root.join(&manifest)).unwrap_or_default();
        failures.extend(c_library(&manifest.display().to_string(), &text));
    }
    let members = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
    for m in members
        .split_once("members = [")
        .and_then(|(_, r)| r.split_once(']'))
        .map_or("", |(l, _)| l)
        .split(',')
        .map(|m| m.trim().trim_matches('"'))
        .filter(|m| !m.is_empty())
    {
        let path = format!("{m}/Cargo.toml");
        let text = std::fs::read_to_string(root.join(&path)).unwrap();
        failures.extend(c_library(&path, &text));
    }
    failures.sort();
    failures.dedup();
    assert!(
        failures.is_empty(),
        "{} export(s) to C:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn an_export_is_found_and_a_callback_is_not_one() {
    let bad = "#[no_mangle]\npub extern \"C\" fn a() {}\n#[unsafe(no_mangle)]\nfn b() {}\n\
               #[unsafe(export_name = \"c\")]\nfn c() {}\npub(crate) unsafe extern \"C-unwind\" fn d() {}\n\
               pub extern fn e() {}\n";
    let f = exports("a.rs", bad);
    assert_eq!(f.len(), 6, "{f:#?}");
    assert!(f.iter().any(|m| m.starts_with("a.rs:7:")), "{f:#?}");
    let ok = "extern \"C\" fn flush_at_exit() {}\nunsafe extern \"C\" {\n    pub fn fb_x();\n}\n\
              // #[no_mangle]\nlet s = \"pub extern fn\";\nfn republish() {}\n";
    assert_eq!(exports("a.rs", ok), Vec::<String>::new());
    assert!(c_library("a/Cargo.toml", "[lib]\ncrate-type = [\"cdylib\"]\n").is_some());
    assert!(c_library("a/Cargo.toml", "[lib]\n# crate-type = [\"cdylib\"]\n").is_none());
}

// ------------------------------------------------------ one name per BLAS call

/// The free functions `fb_core_wwr::blas` exports: each `pub fn` at the
/// start of a line of its code, so a method in an `impl` is not one.
fn single_calls(code: &str) -> BTreeSet<String> {
    let ws = words(code);
    let mut out = BTreeSet::new();
    for (k, (i, w)) in ws.iter().enumerate() {
        let at_line_start = *i == 0 || code.as_bytes()[i - 1] == b'\n';
        if *w == "pub"
            && at_line_start
            && let [(_, "fn"), (_, name), ..] = &ws[k + 1..]
        {
            out.insert((*name).to_owned());
        }
    }
    out
}

/// The names one source of `fb-recipes-blas` exports that are also single
/// calls: each `pub fn` (not `pub(crate) fn`, whose next word is `crate`)
/// and each word of a `pub use`, which also fails on a glob.
fn shared_blas_names(path: &str, source: &str, calls: &BTreeSet<String>) -> Vec<String> {
    let code = code_only(source);
    let ws = words(&code);
    let mut failures = Vec::new();
    for (k, (i, w)) in ws.iter().enumerate() {
        if *w != "pub" {
            continue;
        }
        let line = line_of(source, *i);
        match ws.get(k + 1) {
            Some((_, "fn")) => {
                if let Some((_, name)) = ws.get(k + 2)
                    && calls.contains(*name)
                {
                    failures.push(format!(
                        "{path}:{line}: pub fn {name} is a single BLAS call, which is \
                         fb_core_wwr::blas::{name}'s name alone; keep it private to the crate"
                    ));
                }
            }
            Some((j, "use")) => {
                let end = code[*j..].find(';').map_or(code.len(), |e| j + e);
                let stmt = &code[*j..end];
                if stmt.contains('*') {
                    failures.push(format!(
                        "{path}:{line}: a glob pub use may re-export a single BLAS call"
                    ));
                }
                for (_, name) in words(stmt) {
                    if calls.contains(name) {
                        failures.push(format!(
                            "{path}:{line}: pub use re-exports {name}, a single BLAS call, \
                             which is fb_core_wwr::blas::{name}'s name alone"
                        ));
                    }
                }
            }
            _ => {}
        }
    }
    failures
}

#[test]
fn recipes_blas_exports_no_single_call() {
    let root = rust_root();
    let core = std::fs::read_to_string(root.join("core/wwr/src/blas.rs")).unwrap();
    let calls = single_calls(&code_only(&core));
    for name in ["gemm", "dgmm", "gemv", "geam", "syrk", "trsm"] {
        assert!(
            calls.contains(name),
            "core/wwr/src/blas.rs: no pub fn {name} found: {calls:?}"
        );
    }
    let failures: Vec<String> = workspace_sources()
        .iter()
        .filter(|(p, _)| p.starts_with("recipes/blas/src/"))
        .flat_map(|(p, s)| shared_blas_names(p, s, &calls))
        .collect();
    assert!(
        failures.is_empty(),
        "{} single BLAS call(s) exported by fb-recipes-blas:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn a_shared_blas_name_is_found_and_a_private_one_is_not() {
    let calls = single_calls(&code_only(
        "pub fn gemm() {}\nimpl H {\n    pub fn new() {}\n}\n/// pub fn trsm\npub fn dgmm() {}\n",
    ));
    assert_eq!(
        calls,
        BTreeSet::from(["gemm".to_owned(), "dgmm".to_owned()])
    );
    let bad = "pub fn gemm() {}\npub use fb_core_wwr::blas::dgmm;\npub use x::*;\n\
               mod m {\n    pub fn gemm() {}\n}\n";
    let f = shared_blas_names("a.rs", bad, &calls);
    assert_eq!(f.len(), 4, "{f:#?}");
    assert!(f.iter().any(|m| m.starts_with("a.rs:5:")), "{f:#?}");
    let ok = "pub(crate) fn gemm() {}\nfn dgmm() {}\npub fn gemmwich() {}\n\
              // pub fn gemm\nlet s = \"pub fn dgmm\";\nuse fb_core_wwr::blas::{self, gemm};\n";
    assert_eq!(shared_blas_names("a.rs", ok, &calls), Vec::<String>::new());
}

// ------------------------------------------------------- safe wrapper links

/// One function of a binding crate's safe layer.
struct Func {
    path: String,
    line: usize,
    name: String,
    public: bool,
    doc: String,
    /// The names it calls, as `name(` not after a `.`.
    calls: BTreeSet<String>,
}

/// The functions of one source, with their doc comments and calls.
fn functions(path: &str, source: &str) -> Vec<Func> {
    let code = code_only(source);
    let lines: Vec<&str> = source.lines().collect();
    let attrs = attributes(&code);
    let b = code.as_bytes();
    let ws = words(&code);
    let mut out = Vec::new();
    for (k, (i, w)) in ws.iter().enumerate() {
        if *w != "fn" {
            continue;
        }
        let Some((name_at, name)) = ws.get(k + 1).copied() else {
            continue;
        };
        // A `fn` type or a declaration has no body before its `;`.
        let Some(open) = (name_at..b.len()).find(|&j| b[j] == b'{' || b[j] == b';') else {
            continue;
        };
        if b[open] == b';' {
            continue;
        }
        let mut depth = 0i32;
        let close = (open..b.len())
            .find(|&j| {
                depth += match b[j] {
                    b'{' => 1,
                    b'}' => -1,
                    _ => 0,
                };
                depth == 0
            })
            .unwrap_or(b.len());
        let body = &code[open..close];
        let bw = words(body);
        let calls = bw
            .iter()
            .filter(|(at, w)| {
                !body[..*at].ends_with('.') && body[at + w.len()..].trim_start().starts_with('(')
            })
            .map(|(_, w)| (*w).to_string())
            .collect();
        let head = code[..*i].rfind('\n').map_or(0, |p| p + 1);
        let vis = code[head..*i].trim();
        let public = vis.split_whitespace().next() == Some("pub");
        // The doc comment: up the lines over attributes, doc lines, and the
        // plain comments between attributes (a `// RELEASE:`).
        let mut doc = Vec::new();
        let mut n = line_of(&code, *i) - 1;
        while n > 0 {
            let above = lines[n - 1].trim_start();
            if let Some(d) = above.strip_prefix("///") {
                doc.push(d);
                n -= 1;
            } else if above.starts_with("//") && !above.starts_with("//!") {
                n -= 1;
            } else if let Some(a) = attrs.iter().find(|a| line_of(&code, a.end - 1) == n) {
                n = a.line - 1;
            } else {
                break;
            }
        }
        doc.reverse();
        out.push(Func {
            path: path.to_string(),
            line: line_of(&code, *i),
            name: name.to_string(),
            public,
            doc: doc.join("\n"),
            calls,
        });
    }
    out
}

/// Whether a doc comment links to `name` as an intra-doc link.
fn links(doc: &str, name: &str) -> bool {
    [
        format!("[`{name}`]"),
        format!("[{name}]"),
        format!("({name})"),
        format!("::{name})"),
    ]
    .iter()
    .any(|l| doc.contains(l.as_str()))
}

/// The bindings' function names, from a `bindings.rs`.
fn binding_names(bindings: &str) -> BTreeSet<String> {
    let code = code_only(bindings);
    let ws = words(&code);
    ws.windows(3)
        .filter(|w| w[0].1 == "pub" && w[1].1 == "fn")
        .map(|w| w[2].1.to_string())
        .collect()
}

/// Every public function of a safe layer that reaches a binding and does
/// not link down to it: each binding it calls itself, or, when it calls
/// none itself, one it reaches through the layer's private functions.
fn unlinked(bindings: &BTreeSet<String>, funcs: &[Func]) -> Vec<String> {
    let private: BTreeMap<&str, &Func> = funcs
        .iter()
        .filter(|f| !f.public)
        .map(|f| (f.name.as_str(), f))
        .collect();
    // What each private function reaches, to a fixed point.
    let mut reach: BTreeMap<&str, BTreeSet<String>> = private
        .iter()
        .map(|(n, f)| (*n, f.calls.intersection(bindings).cloned().collect()))
        .collect();
    loop {
        let mut grew = false;
        for (n, f) in &private {
            let more: BTreeSet<String> = f
                .calls
                .iter()
                .filter_map(|c| reach.get(c.as_str()))
                .flatten()
                .cloned()
                .collect();
            let mine = reach.get_mut(n).unwrap();
            let before = mine.len();
            mine.extend(more);
            grew |= mine.len() > before;
        }
        if !grew {
            break;
        }
    }
    let mut failures = Vec::new();
    for f in funcs.iter().filter(|f| f.public) {
        let direct: Vec<&String> = f.calls.intersection(bindings).collect();
        let missing: Vec<&String> = direct
            .iter()
            .copied()
            .filter(|b| !links(&f.doc, b))
            .collect();
        if !missing.is_empty() {
            failures.push(format!(
                "{}:{}: {} calls {} but its doc comment does not link it; the safe wrapper \
                 links down to the -sys item",
                f.path,
                f.line,
                f.name,
                missing
                    .iter()
                    .map(|m| format!("`{m}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
            continue;
        }
        if !direct.is_empty() {
            continue;
        }
        let via: BTreeSet<&String> = f
            .calls
            .iter()
            .filter_map(|c| reach.get(c.as_str()))
            .flatten()
            .collect();
        if !via.is_empty() && !via.iter().any(|b| links(&f.doc, b)) {
            failures.push(format!(
                "{}:{}: {} reaches {} but its doc comment links none of them; the safe wrapper \
                 links down to the -sys item",
                f.path,
                f.line,
                f.name,
                via.iter()
                    .map(|m| format!("`{m}`"))
                    .collect::<Vec<_>>()
                    .join(", ")
            ));
        }
    }
    failures
}

#[test]
fn every_safe_wrapper_links_down_to_its_binding() {
    let sources = workspace_sources();
    let crates: Vec<&str> = sources
        .iter()
        // A bindings file with no extern block is a shared header's
        // (`fb-kernel-abi`'s), which binds no function to wrap.
        .filter(|(_, s)| s.contains("extern \"C\""))
        .filter_map(|(p, _)| p.strip_suffix("/src/bindings.rs"))
        .collect();
    assert!(crates.len() >= 2, "binding crates found: {crates:?}");
    let mut failures = Vec::new();
    for c in crates {
        let prefix = format!("{c}/src/");
        let bindings = sources
            .iter()
            .find(|(p, _)| *p == format!("{prefix}bindings.rs"))
            .map(|(_, s)| binding_names(s))
            .unwrap_or_default();
        assert!(!bindings.is_empty(), "{c}: no binding functions read");
        let funcs: Vec<Func> = sources
            .iter()
            .filter(|(p, _)| p.starts_with(&prefix) && !p.ends_with("/bindings.rs"))
            .flat_map(|(p, s)| functions(p, s))
            .collect();
        failures.extend(unlinked(&bindings, &funcs));
    }
    assert!(
        failures.is_empty(),
        "{} safe wrapper(s) not linking down:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn a_wrapper_links_what_it_calls_or_what_its_helper_does() {
    let bindings =
        binding_names("unsafe extern \"C\" {\n    pub fn fb_a(n: u32);\n    pub fn fb_b();\n}\n");
    assert_eq!(bindings.len(), 2);
    let src = concat!(
        "/// [`fb_a`], checked.\n// RELEASE: x.\n#[track_caller]\npub fn a() { unsafe { fb_a(1) } }\n",
        "/// Calls it.\n#[debug_requires(\n    x,\n    \"m\"\n)]\npub fn b() { unsafe { fb_b() } }\n",
        "/// [`a`] by another name.\npub fn c() { helper() }\n",
        "/// [`fb_b`](crate::fb_b), through a helper.\npub fn d() { helper() }\n",
        "fn helper() { inner() }\nfn inner() { unsafe { fb_b() } }\n",
        "/// No binding.\npub fn e(x: &X) { x.fb_a() }\n",
    );
    let f = unlinked(&bindings, &functions("lib.rs", src));
    assert_eq!(f.len(), 2, "{f:#?}");
    assert!(f[0].starts_with("lib.rs:10: b calls `fb_b`"), "{f:#?}");
    assert!(f[1].starts_with("lib.rs:12: c reaches `fb_b`"), "{f:#?}");
}

// ------------------------------------------------ contract-tripping tests

/// The messages of every contract in the workspace whose attribute is one
/// of `kinds`.
fn contract_messages(sources: &[(String, String)], kinds: &[&str]) -> BTreeSet<String> {
    let mut out = BTreeSet::new();
    for (_, source) in sources {
        let code = code_only(source);
        for a in attributes(&code) {
            if kinds.contains(&a.name.as_str()) {
                out.extend(string_literals(&source[a.start..a.end]).pop());
            }
        }
    }
    out.retain(|m| !m.trim().is_empty());
    out
}

/// One `#[should_panic(expected = …)]` test: the line of its attribute,
/// the expected text, and whether it is under `#[cfg(debug_assertions)]`,
/// on itself or on a module, `impl` or file around it.
struct PanicTest {
    line: usize,
    expected: String,
    debug: bool,
}

/// Every `#[should_panic(expected = …)]` test of one file.
fn panic_tests(source: &str) -> Vec<PanicTest> {
    let code = code_only(source);
    let b = code.as_bytes();
    let attrs = attributes(&code);
    let mut out = Vec::new();
    // One flag per open brace: whether what it opens is debug-only.
    let mut frames = vec![false];
    let mut pending: Vec<&Attr> = Vec::new();
    let mut next_brace = false;
    let mut ai = 0;
    let mut i = 0;
    while i < b.len() {
        if let Some(a) = attrs.get(ai).filter(|a| a.start == i) {
            if a.inner {
                if debug_only(source, a) {
                    *frames.last_mut().unwrap() = true;
                }
            } else {
                pending.push(a);
            }
            ai += 1;
            i = a.end;
            continue;
        }
        let here_debug = *frames.last().unwrap();
        match b[i] {
            b'{' => {
                frames.push(here_debug || next_brace);
                next_brace = false;
                pending.clear();
            }
            b'}' => {
                if frames.len() > 1 {
                    frames.pop();
                }
                pending.clear();
            }
            b';' => {
                next_brace = false;
                pending.clear();
            }
            c if is_ident(c) && (i == 0 || !is_ident(b[i - 1])) => {
                let w = word_at(b, i);
                let debug = pending.iter().any(|a| debug_only(source, a));
                match w {
                    "pub" | "crate" | "super" | "self" | "in" | "async" | "unsafe" | "const"
                    | "extern" => {}
                    "mod" | "impl" | "trait" => {
                        next_brace = debug;
                        pending.clear();
                    }
                    "fn" => {
                        for a in pending.iter().filter(|a| a.name == "should_panic") {
                            let expected = string_literals(&source[a.start..a.end])
                                .pop()
                                .unwrap_or_default();
                            if !expected.is_empty() {
                                out.push(PanicTest {
                                    line: a.line,
                                    expected,
                                    debug: here_debug || debug,
                                });
                            }
                        }
                        next_brace = debug;
                        pending.clear();
                    }
                    _ => pending.clear(),
                }
                i += w.len();
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    out
}

/// The message of `messages` a test's expected text trips: one that holds
/// it, or that it holds.
fn tripped<'m>(messages: &'m BTreeSet<String>, expected: &str) -> Option<&'m String> {
    messages
        .iter()
        .find(|m| m.contains(expected) || expected.contains(m.as_str()))
}

/// Every `#[should_panic(expected = …)]` test in one file whose text is a
/// debug contract's message and that is not under `#[cfg(debug_assertions)]`,
/// on itself or on a module, `impl` or file around it.
fn undebugged(path: &str, source: &str, messages: &BTreeSet<String>) -> Vec<String> {
    panic_tests(source)
        .into_iter()
        .filter(|t| !t.debug)
        .filter_map(|t| {
            tripped(messages, &t.expected).map(|m| {
                format!(
                    "{path}:{}: a test that trips the contract \"{m}\" is not \
                     #[cfg(debug_assertions)]; a release build compiles the contract out",
                    t.line
                )
            })
        })
        .collect()
}

/// Every `#[should_panic(expected = …)]` test in one file whose text is an
/// always-on contract's message, and no debug contract's, and that is under
/// `#[cfg(debug_assertions)]`: the contract holds in release too, so its
/// test runs there.
fn needlessly_debug(
    path: &str,
    source: &str,
    always: &BTreeSet<String>,
    debug: &BTreeSet<String>,
) -> Vec<String> {
    panic_tests(source)
        .into_iter()
        .filter(|t| t.debug && tripped(debug, &t.expected).is_none())
        .filter_map(|t| {
            tripped(always, &t.expected).map(|m| {
                format!(
                    "{path}:{}: a test that trips the always-on contract \"{m}\" is \
                     #[cfg(debug_assertions)]; the contract holds in release, so its test \
                     runs there",
                    t.line
                )
            })
        })
        .collect()
}

#[test]
fn a_test_that_trips_a_contract_is_debug_only() {
    let sources = workspace_sources();
    let messages = contract_messages(&sources, &DEBUG_CONTRACTS);
    assert!(
        messages.len() > 10,
        "only {} contract messages",
        messages.len()
    );
    let failures: Vec<String> = sources
        .iter()
        .flat_map(|(p, s)| undebugged(p, s, &messages))
        .collect();
    assert!(
        failures.is_empty(),
        "{} contract-tripping test(s) built in release:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn a_contract_test_is_debug_by_its_own_cfg_or_its_modules() {
    let messages = contract_messages(
        &[(
            "c.rs".to_string(),
            "#[debug_requires(n > 0, \"a matrix of at least one row\")]\nfn f(n: usize) {}\n"
                .to_string(),
        )],
        &DEBUG_CONTRACTS,
    );
    assert_eq!(messages.len(), 1);
    let ok = "#[test]\n#[cfg(debug_assertions)] // compiled out\n\
              #[should_panic(expected = \"at least one row\")]\nfn a() {}\n\
              #[cfg(all(test, debug_assertions))]\nmod tests {\n    #[test]\n    \
              #[should_panic(expected = \"a matrix of at least one row\")]\n    fn b() { {} }\n}\n\
              #[test]\n#[should_panic(expected = \"count is 5, but data holds 4\")]\nfn c() {}\n";
    assert_eq!(undebugged("t.rs", ok, &messages), Vec::<String>::new());
    let bad = "#[cfg(test)]\nmod tests {\n    #[test]\n    #[should_panic(\n        expected = \
               \"a matrix of at least one row\"\n    )]\n    fn a() {}\n}\n\
               #[cfg(not(debug_assertions))]\n#[test]\n#[should_panic(expected = \"one row\")]\nfn b() {}\n\
               #[cfg(debug_assertions)]\nfn helper() {}\n#[test]\n#[should_panic(expected = \"one row\")]\nfn c() {}\n";
    let f = undebugged("t.rs", bad, &messages);
    assert_eq!(f.len(), 3, "{f:#?}");
    assert!(f[0].starts_with("t.rs:4:"), "{f:#?}");
    let whole =
        "#![cfg(debug_assertions)]\n#[test]\n#[should_panic(expected = \"one row\")]\nfn a() {}\n";
    assert_eq!(undebugged("t.rs", whole, &messages), Vec::<String>::new());
}

// ------------------------------------------------------------ DAG builders

/// The byte just past the bracket that closes the one opening at `open`,
/// counting `open_b` and `close_b`, and not the `>` of a `->`.
fn closing(b: &[u8], open: usize, open_b: u8, close_b: u8) -> usize {
    let mut depth = 0i32;
    for j in open..b.len() {
        if b[j] == open_b {
            depth += 1;
        } else if b[j] == close_b && !(close_b == b'>' && j > 0 && b[j - 1] == b'-') {
            depth -= 1;
            if depth == 0 {
                return j + 1;
            }
        }
    }
    b.len()
}

/// `params` split at its top-level commas, each trimmed, a trailing comma's
/// empty one dropped.
fn split_params(params: &str) -> Vec<&str> {
    let b = params.as_bytes();
    let mut out = Vec::new();
    let (mut depth, mut start) = (0i32, 0);
    for (j, &c) in b.iter().enumerate() {
        match c {
            b'(' | b'[' | b'<' => depth += 1,
            b'>' if j > 0 && b[j - 1] == b'-' => {}
            b')' | b']' | b'>' => depth -= 1,
            b',' if depth == 0 => {
                out.push(params[start..j].trim());
                start = j + 1;
            }
            _ => {}
        }
    }
    out.push(params[start..].trim());
    out.retain(|p| !p.is_empty());
    out
}

/// Whether a type is a `&mut Dag`: `&mut Dag<…>`, `&'a mut Dag<…>`, or a
/// path to it (`&mut fb_core_planner::Dag<'_>`).
fn is_dag_mut(ty: &str) -> bool {
    let Some(t) = ty.trim().strip_prefix('&') else {
        return false;
    };
    let t = t.trim_start();
    let t = match t.strip_prefix('\'') {
        Some(l) => l.trim_start_matches(|c: char| c.is_ascii_alphanumeric() || c == '_'),
        None => t,
    };
    let Some(t) = t.trim_start().strip_prefix("mut ") else {
        return false;
    };
    let t: String = t.split_whitespace().collect();
    t.split('<').next().unwrap_or("").rsplit("::").next() == Some("Dag")
}

/// The byte spans of every `#[cfg(test)]` module with a body in `code`.
fn test_modules(code: &str) -> Vec<(usize, usize)> {
    let b = code.as_bytes();
    attributes(code)
        .iter()
        .filter(|a| {
            !a.inner
                && code[a.start..a.end].split_whitespace().collect::<String>() == "#[cfg(test)]"
        })
        .filter_map(|a| {
            let rest = code[a.end..].trim_start();
            let at = code.len() - rest.len();
            if !rest
                .strip_prefix("pub ")
                .unwrap_or(rest)
                .starts_with("mod ")
            {
                return None;
            }
            let open = (at..b.len()).find(|&j| b[j] == b'{' || b[j] == b';')?;
            (b[open] == b'{').then(|| (at, closing(b, open, b'{', b'}')))
        })
        .collect()
}

/// Every public function of `source` that takes a `&mut Dag` and is not
/// shaped as a DAG builder: named `add_*`, `dag` its first parameter after
/// any `self`, and `worker`, when it has one, its second.
fn bad_builders(path: &str, source: &str) -> Vec<String> {
    let code = code_only(source);
    let b = code.as_bytes();
    let skip = test_modules(&code);
    let ws = words(&code);
    let mut failures = Vec::new();
    for (k, (i, w)) in ws.iter().enumerate() {
        if *w != "fn" || skip.iter().any(|(s, e)| (*s..*e).contains(i)) {
            continue;
        }
        let Some((name_at, name)) = ws.get(k + 1).copied() else {
            continue;
        };
        let head = code[..*i].rfind('\n').map_or(0, |p| p + 1);
        if code[head..*i].split_whitespace().next() != Some("pub") {
            continue;
        }
        let mut at = name_at + name.len();
        while b.get(at).is_some_and(u8::is_ascii_whitespace) {
            at += 1;
        }
        if b.get(at) == Some(&b'<') {
            at = closing(b, at, b'<', b'>');
        }
        let Some(open) = (at..b.len()).find(|&j| b[j] == b'(') else {
            continue;
        };
        let close = closing(b, open, b'(', b')');
        let mut params = split_params(&code[open + 1..close - 1]);
        if params
            .first()
            .is_some_and(|p| p.contains("self") && !p.contains(':'))
        {
            params.remove(0);
        }
        let named: Vec<(&str, &str)> = params
            .iter()
            .filter_map(|p| p.split_once(':'))
            .map(|(n, t)| (n.trim().trim_start_matches("mut ").trim(), t.trim()))
            .collect();
        let Some(d) = named.iter().position(|(_, t)| is_dag_mut(t)) else {
            continue;
        };
        let mut why = Vec::new();
        if !name.starts_with("add_") {
            why.push("is not named `add_*`".to_string());
        }
        if d != 0 || named[d].0 != "dag" {
            why.push(format!(
                "takes `{}: &mut Dag` as parameter {}, not `dag` first",
                named[d].0,
                d + 1
            ));
        }
        if let Some(p) = named.iter().position(|(n, _)| *n == "worker")
            && p != d + 1
        {
            why.push(format!(
                "takes `worker` as parameter {}, not right after `dag`",
                p + 1
            ));
        }
        if !why.is_empty() {
            failures.push(format!(
                "{path}:{}: {name} {}; a public function that takes a `&mut Dag` is \
                 `add_*(dag, worker, …)`",
                line_of(&code, *i),
                why.join(", ")
            ));
        }
    }
    failures
}

/// Whether a source is test code: a file under a `tests/` directory, or a
/// `tests.rs`.
fn test_source(path: &str) -> bool {
    path.split('/').any(|s| s == "tests") || path.ends_with("/tests.rs")
}

#[test]
fn every_dag_builder_is_an_add_function() {
    let sources = workspace_sources();
    let checked: Vec<&(String, String)> = sources
        .iter()
        .filter(|(p, _)| !p.starts_with("core/") && !test_source(p))
        .collect();
    let builders = checked
        .iter()
        .filter(|(_, s)| s.contains("dag: &mut Dag"))
        .count();
    assert!(
        builders >= 10,
        "sources with a DAG builder found: {builders}"
    );
    let failures: Vec<String> = checked
        .iter()
        .flat_map(|(p, s)| bad_builders(p, s))
        .collect();
    assert!(
        failures.is_empty(),
        "{} DAG builder(s) not shaped add_*(dag, worker, …):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn a_dag_builder_is_judged_by_its_name_and_its_parameters() {
    let ok = concat!(
        "pub fn add_a(dag: &mut Dag<'_>, worker: usize, x: RowId<Device, f64>) -> RowId {}\n",
        "pub fn add_b<'a, H: StagingKind, S: 'a>(\n    dag: &mut Dag<'a, H, S>,\n    \
         worker: usize,\n) {}\n",
        "pub fn add_c(dag: &mut fb_core_planner::Dag<'_>, rows: &[ClassRows]) {}\n",
        "impl State {\n    pub fn add_push(\n        &self,\n        dag: &mut Dag<'_>,\n        \
         r: RowId,\n    ) {}\n}\n",
        "fn upload(&self, x: usize, dag: &mut Dag<'_>) {}\n",
        "pub(crate) fn helper(x: usize, dag: &mut Dag<'_>) {}\n",
        "pub fn not_a_builder(dag: &Dag<'_>, f: impl Fn(usize) -> usize) {}\n",
        "#[cfg(test)]\nmod tests {\n    pub fn upload(x: u8, dag: &mut Dag<'_>) {}\n}\n",
        "// pub fn commented(x: u8, dag: &mut Dag<'_>) {}\n",
    );
    assert_eq!(bad_builders("ok.rs", ok), Vec::<String>::new());
    let bad = concat!(
        "pub fn upload(dag: &mut Dag<'_>, worker: usize) {}\n",
        "pub fn add_x(\n    worker: usize,\n    dag: &mut Dag<'_>,\n) {}\n",
        "pub fn add_y(dag: &mut Dag<'_>, n: usize, worker: usize) {}\n",
        "pub fn add_z<F: Fn(u8) -> u8>(d: &'a mut Dag<'a>, f: F) {}\n",
    );
    let f = bad_builders("bad.rs", bad);
    assert_eq!(f.len(), 4, "{f:#?}");
    assert!(
        f[0].starts_with("bad.rs:1: upload is not named `add_*`;"),
        "{f:#?}"
    );
    assert!(
        f[1].starts_with(
            "bad.rs:2: add_x takes `dag: &mut Dag` as parameter 2, not `dag` first, \
             takes `worker` as parameter 1"
        ),
        "{f:#?}"
    );
    assert!(
        f[2].starts_with("bad.rs:6: add_y takes `worker` as parameter 3"),
        "{f:#?}"
    );
    assert!(
        f[3].starts_with("bad.rs:7: add_z takes `d: &mut Dag` as parameter 1"),
        "{f:#?}"
    );
}

#[test]
fn a_test_that_trips_an_always_on_contract_runs_in_release() {
    let sources = workspace_sources();
    let always = contract_messages(&sources, &ALWAYS_CONTRACTS);
    let debug = contract_messages(&sources, &DEBUG_CONTRACTS);
    let failures: Vec<String> = sources
        .iter()
        .flat_map(|(p, s)| needlessly_debug(p, s, &always, &debug))
        .collect();
    assert!(
        failures.is_empty(),
        "{} always-on contract test(s) built in debug only:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn an_always_on_contract_test_is_found_under_a_debug_cfg() {
    let src = "#[requires(n > 0, \"a matrix of at least one row\")]\nfn f(n: usize) {}\n\
               #[debug_requires(m > 0, \"m is positive\")]\nfn g(m: usize) {}\n";
    let sources = [("c.rs".to_string(), src.to_string())];
    let always = contract_messages(&sources, &ALWAYS_CONTRACTS);
    let debug = contract_messages(&sources, &DEBUG_CONTRACTS);
    assert_eq!(always.len(), 1);
    assert_eq!(debug.len(), 1);
    let ok = "#[test]\n#[should_panic(expected = \"at least one row\")]\nfn a() {}\n\
              #[test]\n#[cfg(debug_assertions)]\n#[should_panic(expected = \"m is positive\")]\nfn b() {}\n";
    assert_eq!(
        needlessly_debug("t.rs", ok, &always, &debug),
        Vec::<String>::new()
    );
    let bad = "#[test]\n#[cfg(debug_assertions)]\n#[should_panic(expected = \"one row\")]\nfn a() {}\n\
               #[cfg(all(test, debug_assertions))]\nmod tests {\n    #[test]\n    \
               #[should_panic(expected = \"a matrix of at least one row\")]\n    fn b() {}\n}\n";
    let f = needlessly_debug("t.rs", bad, &always, &debug);
    assert_eq!(f.len(), 2, "{f:#?}");
    assert!(f[0].starts_with("t.rs:3:"), "{f:#?}");
    // And the debug-only check does not take an always-on message for one.
    assert_eq!(undebugged("t.rs", ok, &debug), Vec::<String>::new());
}
