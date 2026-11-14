//! Contracts are written once, and only as contracts (`docs/testing.md`,
//! "Contracts are part of the spec").
//!
//! - **No doc comment restates a contract.** rustdoc renders every
//!   `#[debug_requires]`, `#[debug_ensures]` and `#[debug_invariant]` as the
//!   item's `§ Contracts` section, its message and its expression
//!   (`docs/documentation.md`, "Code: rustdoc"), so a sentence in the doc
//!   comment saying the same is a second copy that drifts. Two shapes are
//!   refused: the old restating phrase, anywhere in a doc comment, and a doc
//!   comment that holds the message of a contract on its own item.
//! - **A `debug_assert!` is an internal invariant, and says so.** A caller's
//!   precondition is a `#[debug_requires]`, where rustdoc shows it; a check
//!   soundness rests on is an `assert!`, named in `# Panics`. What is left
//!   for `debug_assert!` (and its `_eq` and `_ne` forms) is a fact of the
//!   function's own making that no caller can break, and each one carries a
//!   `// INVARIANT:` comment directly above it saying why.
//!
//! clippy's `disallowed_macros` cannot hold the second rule: a
//! `#[debug_requires]` expands to a `debug_assert!` spanned at the
//! contract, so the lint flags every contract in the tree.
//!
//! Every `.rs` under every workspace member (the `members` of
//! `rust/Cargo.toml`) is read, `target/` excluded.

#![forbid(unsafe_code)]

use fb_doc_check::{code_only, is_ident, string_literals, workspace_sources};

/// The phrase every hand-written restatement used, matched
/// case-insensitively.
const RESTATED: &str = "checked in debug build";
/// The comment that marks a `debug_assert!` as an internal invariant.
const INVARIANT: &str = "// INVARIANT:";
/// The attributes rustdoc renders as `§ Contracts`.
const CONTRACTS: [&str; 6] = [
    "debug_requires",
    "debug_ensures",
    "debug_invariant",
    "requires",
    "ensures",
    "invariant",
];
/// The macros only an internal invariant may use.
const DEBUG_ASSERTS: [&str; 3] = ["debug_assert", "debug_assert_eq", "debug_assert_ne"];

/// Whether a line is a doc comment, outer or inner.
fn doc_line(line: &str) -> Option<&str> {
    let t = line.trim_start();
    t.strip_prefix("///").or_else(|| t.strip_prefix("//!"))
}

/// Lowercase, with every run of whitespace one space.
fn normal(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// The restatements in one file: a doc line with the old phrase, and an
/// item whose doc comment holds one of its own contracts' messages.
fn restated(path: &str, source: &str) -> Vec<String> {
    let lines: Vec<&str> = source.lines().collect();
    let code = code_only(source);
    let code: Vec<&str> = code.lines().collect();
    let mut failures = Vec::new();
    // The current item's doc text, and its contracts' messages.
    let mut doc = String::new();
    let mut messages: Vec<String> = Vec::new();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        if let Some(text) = doc_line(line) {
            if text.to_lowercase().contains(RESTATED) {
                failures.push(format!(
                    "{path}:{}: restates a contract (\"{RESTATED}\"); rustdoc renders it",
                    i + 1
                ));
            }
            doc.push(' ');
            doc.push_str(text);
            i += 1;
            continue;
        }
        let t = code.get(i).copied().unwrap_or("").trim_start();
        if let Some(after) = t.strip_prefix("#[") {
            // One attribute, to its closing bracket, which may be lines on.
            let start = i;
            let mut depth = 0i32;
            loop {
                for c in code.get(i).copied().unwrap_or("").chars() {
                    match c {
                        '[' | '(' | '{' => depth += 1,
                        ']' | ')' | '}' => depth -= 1,
                        _ => {}
                    }
                }
                i += 1;
                if depth <= 0 || i >= lines.len() {
                    break;
                }
            }
            let attr = lines[start..i].join("\n");
            let name: String = after
                .bytes()
                .take_while(|&c| is_ident(c))
                .map(char::from)
                .collect();
            if CONTRACTS.contains(&name.as_str()) {
                messages.extend(string_literals(&attr).last().cloned());
            }
            continue;
        }
        // The item, or anything else: the run ends here.
        let text = normal(&doc);
        for m in &messages {
            let m = normal(m);
            if !m.is_empty() && text.contains(&m) {
                failures.push(format!(
                    "{path}:{}: the doc comment restates its contract \"{m}\"; rustdoc renders it",
                    i + 1
                ));
            }
        }
        doc.clear();
        messages.clear();
        i += 1;
    }
    failures
}

/// The `debug_assert!`s in one file with no `// INVARIANT:` comment
/// directly above them.
fn unmarked(path: &str, source: &str) -> Vec<String> {
    let lines: Vec<&str> = source.lines().collect();
    let code = code_only(source);
    let mut failures = Vec::new();
    for (n, line) in code.lines().enumerate() {
        let b = line.as_bytes();
        let mut j = 0;
        while j < b.len() {
            if !is_ident(b[j]) {
                j += 1;
                continue;
            }
            let start = j;
            while j < b.len() && is_ident(b[j]) {
                j += 1;
            }
            let word = &line[start..j];
            let bang = line[j..].trim_start().starts_with('!');
            if !(bang && DEBUG_ASSERTS.contains(&word)) {
                continue;
            }
            let marked = lines[..n]
                .iter()
                .rev()
                .map(|l| l.trim_start())
                .take_while(|l| l.starts_with("//") && doc_line(l).is_none())
                .any(|l| l.starts_with(INVARIANT));
            if !marked {
                failures.push(format!(
                    "{path}:{}: {word}! with no {INVARIANT} comment above it; a caller's \
                     precondition is a #[debug_requires], a check soundness rests on an assert!",
                    n + 1
                ));
            }
        }
    }
    failures
}

#[test]
fn no_doc_comment_restates_a_contract() {
    let sources = workspace_sources();
    assert!(sources.len() > 50, "only {} sources found", sources.len());
    let failures: Vec<String> = sources.iter().flat_map(|(p, s)| restated(p, s)).collect();
    assert!(
        failures.is_empty(),
        "{} restated contract(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn every_debug_assert_is_a_marked_invariant() {
    let failures: Vec<String> = workspace_sources()
        .iter()
        .flat_map(|(p, s)| unmarked(p, s))
        .collect();
    assert!(
        failures.is_empty(),
        "{} debug_assert(s) not marked as invariants:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn a_restatement_is_found_by_its_phrase_or_its_message() {
    let phrase = format!("/// Requires, {RESTATED}s: n is at least 1.\nfn f() {{}}\n");
    assert_eq!(restated("a.rs", &phrase).len(), 1);

    let message = "/// Lay out a matrix of\n/// at least one row.\n#[must_use]\n\
                   #[debug_requires(\n    n > 0,\n    \"a matrix of at least one row\"\n)]\n\
                   pub fn f(n: usize) {}\n";
    let f = restated("a.rs", message);
    assert_eq!(f.len(), 1, "{f:#?}");
    assert!(f[0].starts_with("a.rs:8:"), "{f:#?}");

    // Another item's message, and a contract with no doc, are not restated.
    let clean = "/// A matrix of at least one row.\nfn g() {}\n\n/// Other.\n\
                 #[debug_requires(n > 0, \"a matrix of at least one row\")]\nfn f(n: usize) {}\n\
                 #[debug_requires(n > 0, \"other\")]\nfn h(n: usize) {}\n";
    assert_eq!(restated("a.rs", clean), Vec::<String>::new());
}

#[test]
fn a_debug_assert_needs_the_marker_directly_above() {
    let marked = "fn f() {\n    // INVARIANT: ranks only grow.\n    // More.\n    debug_assert!(a < b);\n}\n";
    assert_eq!(unmarked("a.rs", marked), Vec::<String>::new());
    let bare = "fn f() {\n    debug_assert_eq!(a, b);\n    debug_assert_ne !(a, b);\n}\n";
    assert_eq!(unmarked("a.rs", bare).len(), 2);
    let gap = "// INVARIANT: too far.\nlet x = 1;\ndebug_assert!(x == 1);\n";
    assert_eq!(unmarked("a.rs", gap).len(), 1);
    // In a comment, a string, or as a cfg name, it is no call.
    let not_calls =
        "// debug_assert!(x)\nlet s = \"debug_assert!(x)\";\n#[cfg(debug_assertions)]\nfn f() {}\n";
    assert_eq!(unmarked("a.rs", not_calls), Vec::<String>::new());
}
