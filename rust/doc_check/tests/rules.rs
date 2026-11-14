//! Every check the rules register names exists (`docs/rules.md`, "Reading
//! the register"). The register is the tables of `docs/rules.md` whose
//! header is `| Rule | Stated | Checked by |`; each row's last cell is a
//! list joined by `;` of: ``lint `name` ``, configured in
//! `[workspace.lints]` of `Cargo.toml` or in `clippy.toml`; ``type `Name` ``,
//! defined by a workspace source; a path in backticks, a file under `rust/`
//! or else the repository root, with `::name` for a test or function the
//! file must define; or `**unenforced**`, alone. An item may end in a
//! parenthesised note. The `Stated` cell must cite a document beside the
//! register (or the register itself), which `section_citations.rs` then
//! checks heading by heading, and every document beside it must be cited by
//! some row.
//!
//! It checks that a named check exists, not that it checks its rule. Tier 3:
//! it reads files and nothing else.

#![forbid(unsafe_code)]

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The register, relative to the workspace root.
const REGISTER: &str = "docs/rules.md";
/// The header of a register table, cells trimmed.
const HEADER: [&str; 3] = ["Rule", "Stated", "Checked by"];
/// The one item that stands for no check.
const UNENFORCED: &str = "**unenforced**";

/// One parsed item of a `Checked by` cell.
#[derive(Debug, PartialEq, Eq)]
enum Check {
    Lint(String),
    Type(String),
    File { path: String, item: Option<String> },
    Unenforced,
}

/// What the register's checks are looked up in.
struct Tree<'a> {
    /// The lints `Cargo.toml` and `clippy.toml` configure, as lint names.
    lints: BTreeSet<String>,
    /// Whether a struct, enum, trait or type alias of this name is defined.
    has_type: &'a dyn Fn(&str) -> bool,
    /// A file's text, by its path as written in the register.
    read: &'a dyn Fn(&str) -> Option<String>,
    /// The documents beside the register, by file name.
    docs: BTreeSet<String>,
}

/// How many rows named each kind of check, so a parser that stopped
/// matching fails rather than passes.
#[derive(Default, Debug)]
struct Counts {
    rows: usize,
    lint: usize,
    ty: usize,
    file: usize,
    unenforced: usize,
}

/// The rows of every register table in `text`: (line number, cells).
fn rows(text: &str) -> Vec<(usize, Vec<String>)> {
    let mut out = Vec::new();
    let mut fenced = false;
    let mut in_register = false;
    for (i, line) in text.lines().enumerate() {
        let t = line.trim();
        if t.starts_with("```") {
            fenced = !fenced;
            continue;
        }
        if fenced || !t.starts_with('|') {
            in_register = false;
            continue;
        }
        let cells: Vec<String> = t
            .trim_matches('|')
            .split('|')
            .map(|c| c.trim().to_owned())
            .collect();
        if cells.iter().map(String::as_str).eq(HEADER) {
            in_register = true;
        } else if in_register && !cells.iter().all(|c| c.chars().all(|ch| ch == '-')) {
            out.push((i + 1, cells));
        }
    }
    out
}

/// `s` split at each `;` outside parentheses.
fn items(s: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let (mut depth, mut start) = (0i32, 0);
    for (i, c) in s.char_indices() {
        match c {
            '(' => depth += 1,
            ')' => depth -= 1,
            ';' if depth == 0 => {
                out.push(s[start..i].trim());
                start = i + 1;
            }
            _ => {}
        }
    }
    out.push(s[start..].trim());
    out
}

/// One item, its trailing note dropped.
fn parse(item: &str) -> Result<Check, String> {
    let head = match item.find(" (") {
        Some(i) if item.ends_with(')') => item[..i].trim(),
        _ => item,
    };
    let quoted = |s: &str| {
        s.strip_prefix('`')
            .and_then(|s| s.strip_suffix('`'))
            .filter(|s| !s.is_empty() && !s.contains('`'))
            .map(str::to_owned)
    };
    if head == UNENFORCED {
        return Ok(Check::Unenforced);
    }
    if let Some(name) = head.strip_prefix("lint ").and_then(quoted) {
        return Ok(Check::Lint(name));
    }
    if let Some(name) = head.strip_prefix("type ").and_then(quoted) {
        return Ok(Check::Type(name));
    }
    if let Some(p) = quoted(head) {
        let (path, item) = match p.split_once("::") {
            Some((path, item)) => (path.to_owned(), Some(item.to_owned())),
            None => (p, None),
        };
        if path.contains('.') {
            return Ok(Check::File { path, item });
        }
    }
    Err(format!(
        "\"{item}\" is not lint `x`, type `X`, a `path` or {UNENFORCED}"
    ))
}

/// Whether `text` defines a function, Rust's or Python's, named `name`.
fn defines_fn(text: &str, name: &str) -> bool {
    ["fn ", "def "].iter().any(|kw| {
        let pat = format!("{kw}{name}");
        text.match_indices(&pat).any(|(i, _)| {
            let before = text[..i].chars().next_back();
            let after = text[i + pat.len()..].chars().next();
            !before.is_some_and(is_ident) && matches!(after, Some('(' | '<'))
        })
    })
}

fn is_ident(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// The failures of the register `text` against `tree`, and the counts.
fn check(text: &str, tree: &Tree<'_>) -> (Vec<String>, Counts) {
    let mut failures = Vec::new();
    let mut counts = Counts::default();
    let mut cited = BTreeSet::new();
    for (line, cells) in rows(text) {
        let mut fail = |why: String| failures.push(format!("{REGISTER}:{line}: {why}"));
        let [rule, stated, checked] = cells.as_slice() else {
            fail(format!("{} cells, not 3", cells.len()));
            continue;
        };
        counts.rows += 1;
        if rule.is_empty() {
            fail("no rule".to_owned());
        }
        let docs: Vec<&str> = stated
            .match_indices("](")
            .filter_map(|(i, _)| {
                let rest = &stated[i + 2..];
                Some(&rest[..rest.find(')')?]).filter(|t| t.ends_with(".md"))
            })
            .collect();
        if docs.is_empty() || !stated.contains('"') {
            fail(format!(
                "Stated \"{stated}\" cites no [`x.md`](x.md), \"Section\""
            ));
        }
        for d in docs {
            if tree.docs.contains(d) {
                cited.insert(d.to_owned());
            } else if format!("docs/{d}") != REGISTER {
                fail(format!("Stated cites {d}, which is not beside {REGISTER}"));
            }
        }
        let parsed: Vec<Result<Check, String>> = items(checked).into_iter().map(parse).collect();
        if parsed.len() > 1 && parsed.iter().any(|p| p == &Ok(Check::Unenforced)) {
            fail(format!("{UNENFORCED} beside a check: \"{checked}\""));
        }
        for p in parsed {
            match p {
                Err(why) => fail(why),
                Ok(Check::Unenforced) => counts.unenforced += 1,
                Ok(Check::Lint(name)) => {
                    counts.lint += 1;
                    if !tree.lints.contains(&name) {
                        fail(format!(
                            "lint `{name}` is configured in neither Cargo.toml's [workspace.lints] nor clippy.toml"
                        ));
                    }
                }
                Ok(Check::Type(name)) => {
                    counts.ty += 1;
                    if !(tree.has_type)(&name) {
                        fail(format!("type `{name}` is defined by no workspace source"));
                    }
                }
                Ok(Check::File { path, item }) => {
                    counts.file += 1;
                    match ((tree.read)(&path), item) {
                        (None, _) => fail(format!("`{path}`: no such file")),
                        (Some(text), Some(item)) if !defines_fn(&text, &item) => {
                            fail(format!("`{path}` defines no `{item}`"));
                        }
                        _ => {}
                    }
                }
            }
        }
    }
    for d in tree.docs.difference(&cited) {
        failures.push(format!("{REGISTER}: no row cites {d}"));
    }
    (failures, counts)
}

/// The lint names configured: every key of a `[workspace.lints.*]` table in
/// `manifest`, and every key of `clippy_toml`, with `-` read as `_`.
fn configured_lints(manifest: &str, clippy_toml: &str) -> BTreeSet<String> {
    let key = |line: &str| {
        let t = line.trim();
        (!t.starts_with('#') && !t.starts_with('[') && t.contains('='))
            .then(|| t.split('=').next().unwrap().trim().replace('-', "_"))
            .filter(|k| !k.is_empty() && k.chars().all(|c| is_ident(c) || c == ':'))
    };
    let mut lints = BTreeSet::new();
    let mut in_lints = false;
    for line in manifest.lines() {
        let t = line.trim();
        if t.starts_with('[') {
            in_lints = t.starts_with("[workspace.lints");
        } else if in_lints {
            lints.extend(key(t));
        }
    }
    lints.extend(clippy_toml.lines().filter_map(key));
    lints
}

/// Whether some line of `text` defines a struct, enum, trait or type alias
/// named `name`.
fn defines_type(text: &str, name: &str) -> bool {
    text.lines().any(|line| {
        let mut t = line.trim_start();
        for prefix in ["pub(crate) ", "pub(super) ", "pub ", "unsafe "] {
            t = t.strip_prefix(prefix).unwrap_or(t);
        }
        ["struct ", "enum ", "trait ", "type "].iter().any(|kw| {
            t.strip_prefix(kw)
                .and_then(|rest| rest.strip_prefix(name))
                .is_some_and(|rest| !rest.starts_with(is_ident))
        })
    })
}

/// Every `.rs` under the workspace's members' `src/` and `tests/`, read
/// from the `members` list of its `Cargo.toml`.
fn workspace_rust(root: &Path) -> Vec<String> {
    fn walk(p: &Path, out: &mut Vec<String>) {
        if p.is_dir() {
            for e in std::fs::read_dir(p).unwrap() {
                walk(&e.unwrap().path(), out);
            }
        } else if p.extension().is_some_and(|e| e == "rs") {
            out.push(std::fs::read_to_string(p).unwrap());
        }
    }
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
    let list = manifest
        .split_once("members = [")
        .and_then(|(_, rest)| rest.split_once(']'))
        .expect("the workspace manifest's members list")
        .0;
    let mut out = Vec::new();
    for m in list.split(',').map(|m| m.trim().trim_matches('"')) {
        if !m.is_empty() {
            for top in ["src", "tests"] {
                walk(&root.join(m).join(top), &mut out);
            }
        }
    }
    out
}

#[test]
fn every_check_the_register_names_exists() {
    let root = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/.."));
    let root = root.canonicalize().unwrap_or(root);
    let repo = root.join("..");
    let text = std::fs::read_to_string(root.join(REGISTER)).unwrap();
    let read_root = |p: &str| std::fs::read_to_string(root.join(p)).unwrap_or_default();
    let sources = workspace_rust(&root);
    let docs = std::fs::read_dir(root.join("docs"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .filter(|n| n.ends_with(".md") && format!("docs/{n}") != REGISTER)
        .collect();
    let tree = Tree {
        lints: configured_lints(&read_root("Cargo.toml"), &read_root("clippy.toml")),
        has_type: &|name| sources.iter().any(|s| defines_type(s, name)),
        read: &|p| {
            std::fs::read_to_string(root.join(p))
                .or_else(|_| std::fs::read_to_string(repo.join(p)))
                .ok()
        },
        docs,
    };
    let (failures, counts) = check(&text, &tree);
    assert!(
        failures.is_empty(),
        "{} broken register row(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert!(
        counts.rows > 0
            && counts.lint > 0
            && counts.ty > 0
            && counts.file > 0
            && counts.unenforced > 0,
        "a kind of check matched no row: the parser has stopped matching ({counts:?})"
    );
}

/// A tree of in-memory files: lints `disallowed_methods`, type `Block`,
/// files `t/a.rs` (defining `fn real`) and `t/b.py` (`def real`), and the
/// documents `a.md` and `b.md`.
fn check_fake(rows: &str) -> Vec<String> {
    let files = [
        ("t/a.rs", "#[test]\nfn real() {}\n"),
        ("t/b.py", "def real() -> None:\n    pass\n"),
    ];
    let tree = Tree {
        lints: configured_lints(
            "[workspace.lints.clippy]\ndisallowed_methods = \"deny\"\n[profile.dev]\nopt-level = 3\n",
            "disallowed-methods = []\n",
        ),
        has_type: &|name| defines_type("pub struct Block<K: Kind> {", name),
        read: &|p| {
            files
                .iter()
                .find(|(f, _)| *f == p)
                .map(|(_, t)| (*t).to_owned())
        },
        docs: ["a.md", "b.md"].map(str::to_owned).into(),
    };
    let text = format!(
        "| Rule | Stated | Checked by |\n|---|---|---|\n\
         | Cover b | [`b.md`](b.md), \"B\" | **unenforced** |\n{rows}"
    );
    check(&text, &tree).0
}

fn row(checked: &str) -> String {
    format!("| A rule | [`a.md`](a.md), \"A\" | {checked} |\n")
}

#[test]
fn every_form_of_check_resolves() {
    let f = check_fake(&row(
        "lint `disallowed_methods`; type `Block`; `t/a.rs`; `t/a.rs::real` (a note; with a semicolon); `t/b.py::real`",
    ));
    assert!(f.is_empty(), "{f:?}");
}

#[test]
fn a_missing_file_or_test_is_reported() {
    let f = check_fake(&row("`t/gone.rs`; `t/a.rs::gone`; `t/a.rs::rea`"));
    assert_eq!(
        f,
        [
            "docs/rules.md:4: `t/gone.rs`: no such file",
            "docs/rules.md:4: `t/a.rs` defines no `gone`",
            "docs/rules.md:4: `t/a.rs` defines no `rea`",
        ]
    );
}

#[test]
fn an_unconfigured_lint_or_unknown_type_is_reported() {
    let f = check_fake(&row(
        "lint `undocumented_unsafe_blocks`; lint `opt_level`; type `Blo`",
    ));
    assert_eq!(f.len(), 3, "{f:?}");
    assert!(f[0].contains("lint `undocumented_unsafe_blocks` is configured in neither"));
    assert!(
        f[1].contains("lint `opt_level`"),
        "a [profile] key is no lint"
    );
    assert!(f[2].contains("type `Blo` is defined by no workspace source"));
}

#[test]
fn a_cell_in_no_form_is_reported() {
    for bad in [
        "",
        "unenforced",
        "a test somewhere",
        "`nodot`",
        "lint disallowed_methods",
    ] {
        let f = check_fake(&row(bad));
        assert_eq!(f.len(), 1, "{bad:?}: {f:?}");
        assert!(f[0].contains("is not lint"), "{bad:?}: {f:?}");
    }
    let f = check_fake(&row("**unenforced**; `t/a.rs`"));
    assert_eq!(f.len(), 1, "{f:?}");
    assert!(f[0].contains("beside a check"), "{f:?}");
}

#[test]
fn a_row_must_cite_a_document_beside_the_register() {
    let f = check_fake(
        "| R | nowhere | **unenforced** |\n| R | [`c.md`](c.md), \"C\" | **unenforced** |\n| R | [`a.md`](a.md), \"A\" | **unenforced** | extra |\n",
    );
    assert_eq!(f.len(), 4, "{f:?}");
    assert!(f[0].contains("cites no"), "{f:?}");
    assert!(f[1].contains("cites c.md, which is not beside"), "{f:?}");
    assert!(f[2].contains("4 cells, not 3"), "{f:?}");
    assert!(f[3].contains("no row cites a.md"), "{f:?}");
}

#[test]
fn only_register_tables_are_read() {
    let f = check_fake(&format!(
        "{}\n| Other | table |\n|---|---|\n| `t/gone.rs` | x |\n\n```\n{}```\n",
        row("**unenforced**"),
        row("`t/gone.rs`")
    ));
    assert!(f.is_empty(), "{f:?}");
}
