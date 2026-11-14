//! The documentation checks that need nothing of the tree but its text:
//! `tests/section_citations.rs`, which checks that every section the
//! workspace's documents and comments cite by name is a heading of the file
//! cited; `tests/unsafe_code.rs`, which checks that every crate root
//! forbids `unsafe_code` unless `docs/programming_model.md` lists its crate
//! as allowed it; `tests/contracts.rs`, which checks that no doc comment
//! restates a contract rustdoc already renders, and that every
//! `debug_assert!` is an internal invariant that says so;
//! `tests/rules.rs`, which checks that every lint, type, test file and test
//! the register `docs/rules.md` names exists; and `tests/workspace_graph.rs`,
//! which reads the package graph (`cargo metadata`, over the manifests'
//! text) and checks the reverse-dependents bar, that no crate is an
//! umbrella, that a crate which only logs reaches no handle crate, that
//! every direct dependency is named in its crate's sources, and the path
//! rules of `devtools/test.py` against the layout; and
//! `tests/source_rules.rs`, which scans the sources for six rules of the
//! register: every `#[ignore]` is a device case's, no Markdown sits beside
//! a module, doc examples carry their markers, nothing is exported to C,
//! the safe wrappers link down to their bindings, and a test that trips a
//! contract is debug-only. A crate of its own
//! so that it builds and runs without the wwr library. The checks are its tests; the
//! library holds only what more than one of them reads.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// `source` with every comment and every string and character literal
/// blanked to spaces, newlines kept, so a byte offset or a line number in
/// the result is the same in `source`. What is left is the code a keyword
/// search should see.
///
/// A raw string's hashes are honoured; a lifetime is code, since it has no
/// closing quote.
#[must_use]
pub fn code_only(source: &str) -> String {
    let b = source.as_bytes();
    let mut out = b.to_vec();
    let mut blank = |from: usize, to: usize| {
        for c in &mut out[from..to.min(b.len())] {
            if *c != b'\n' {
                *c = b' ';
            }
        }
    };
    let mut i = 0;
    while i < b.len() {
        let end = match b[i] {
            b'/' if b.get(i + 1) == Some(&b'/') => {
                source[i..].find('\n').map_or(b.len(), |e| i + e)
            }
            b'/' if b.get(i + 1) == Some(&b'*') => source[i + 2..]
                .find("*/")
                .map_or(b.len(), |e| i + 2 + e + 2),
            b'r' if matches!(b.get(i + 1), Some(b'"' | b'#'))
                && (i == 0 || !is_ident(b[i - 1])) =>
            {
                let hashes = b[i + 1..].iter().take_while(|&&c| c == b'#').count();
                if b.get(i + 1 + hashes) != Some(&b'"') {
                    i += 1;
                    continue;
                }
                let close = format!("\"{}", "#".repeat(hashes));
                let start = i + 2 + hashes;
                source[start..]
                    .find(&close)
                    .map_or(b.len(), |e| start + e + close.len())
            }
            b'"' => {
                let mut j = i + 1;
                while j < b.len() && b[j] != b'"' {
                    j += if b[j] == b'\\' { 2 } else { 1 };
                }
                j + 1
            }
            // A character literal; a lifetime has no closing quote.
            b'\'' if b.get(i + 1) == Some(&b'\\') => source[i + 2..]
                .find('\'')
                .map_or(b.len(), |e| i + 2 + e + 1),
            b'\'' if b.get(i + 2) == Some(&b'\'') => i + 3,
            c if is_ident(c) => {
                while i < b.len() && is_ident(b[i]) {
                    i += 1;
                }
                continue;
            }
            _ => {
                i += 1;
                continue;
            }
        };
        blank(i, end);
        i = end;
    }
    // Only ASCII delimiters bound a blanked span, and every byte inside one
    // became a space, so the bytes are still UTF-8.
    String::from_utf8(out).unwrap_or_default()
}

/// Whether `c` can be part of an identifier.
#[must_use]
pub fn is_ident(c: u8) -> bool {
    c.is_ascii_alphanumeric() || c == b'_'
}

/// The string literals in `text`, in order, each backslash taken as
/// escaping the character after it and dropped (so `\"` is a quote, and
/// `\n` an `n`).
#[must_use]
pub fn string_literals(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '"' {
            continue;
        }
        let mut s = String::new();
        while let Some(c) = chars.next() {
            match c {
                '\\' => s.extend(chars.next()),
                '"' => break,
                c => s.push(c),
            }
        }
        out.push(s);
    }
    out
}

/// The workspace root, `rust/`.
#[must_use]
pub fn rust_root() -> PathBuf {
    let root = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/.."));
    root.canonicalize().unwrap_or(root)
}

/// Every `.rs` of every workspace member (the `members` of
/// `rust/Cargo.toml`), `target/` excluded, as its path from the root and its
/// text, in path order.
///
/// # Panics
///
/// If the manifest or a source cannot be read, or the manifest has no
/// `members` list.
#[must_use]
pub fn workspace_sources() -> Vec<(String, String)> {
    let root = rust_root();
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
    let list = manifest
        .split_once("members = [")
        .and_then(|(_, rest)| rest.split_once(']'))
        .expect("the workspace manifest's members list")
        .0;
    let mut files = BTreeSet::new();
    for m in list.split(',').map(|m| m.trim().trim_matches('"')) {
        if !m.is_empty() {
            rust_files(&root.join(m), &mut files);
        }
    }
    files
        .into_iter()
        .map(|f| {
            let text = std::fs::read_to_string(&f).unwrap();
            let name = f.strip_prefix(&root).unwrap_or(&f).display().to_string();
            (name, text)
        })
        .collect()
}

/// Every `.rs` under `dir`, `target/` excluded, into `out`.
fn rust_files(dir: &Path, out: &mut BTreeSet<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries {
        let p = e.unwrap().path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n != "target") {
                rust_files(&p, out);
            }
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.insert(p);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{code_only, string_literals};

    #[test]
    fn comments_and_literals_are_blanked_and_lines_kept() {
        let src = "a // x\n/* y\n z */ b \"s\\\"t\" r#\"q\"# 'c' '\\n' &'a x\n";
        let got = code_only(src);
        assert_eq!(got.len(), src.len());
        assert_eq!(got.lines().count(), src.lines().count());
        assert_eq!(
            got.split_whitespace().collect::<Vec<_>>(),
            ["a", "b", "&'a", "x"]
        );
    }

    #[test]
    fn string_literals_are_read_in_order_with_escaped_quotes() {
        assert_eq!(
            string_literals(r#"debug_requires(n > 0, "a \"b\" c")"#),
            ["a \"b\" c"]
        );
    }
}
