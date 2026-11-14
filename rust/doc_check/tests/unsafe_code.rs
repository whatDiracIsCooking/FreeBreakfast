//! `unsafe` is where `docs/programming_model.md` says it is
//! (`docs/programming_model.md`, "Capabilities are enforced by lints"). That
//! section lists the crates allowed `unsafe` as a bulleted list after its
//! `**`unsafe`.**` paragraph, every crate named in backticks; this test reads
//! that list, so the document and the enforcement are one list.
//!
//! Every workspace member (the `members` of `rust/Cargo.toml`) is checked:
//! each of its crate roots (`src/lib.rs`, `src/main.rs`, `src/bin/*.rs` and
//! every `[[bin]]` path) carries `#![forbid(unsafe_code)]`, or the crate is
//! listed. A listed crate must still hold `unsafe` somewhere under `src/`,
//! outside comments and string literals, so the list only shrinks; and every
//! name on the list must be a member's package name.

#![forbid(unsafe_code)]

use fb_doc_check::{code_only, is_ident};
use std::path::{Path, PathBuf};

/// The document holding the list, relative to the workspace root.
const DOC: &str = "docs/programming_model.md";
/// The line the list follows.
const ANCHOR: &str = "**`unsafe`.**";
/// The attribute every other crate root carries, on a line of its own.
const FORBID: &str = "#![forbid(unsafe_code)]";

/// One member, as the check sees it.
struct Crate {
    name: String,
    /// Each crate root, with whether it forbids `unsafe_code`.
    roots: Vec<(PathBuf, bool)>,
    /// Whether any source under `src/` holds `unsafe`.
    has_unsafe: bool,
}

/// The crates the list in `doc` names, in order.
fn allowlist(doc: &str) -> Vec<String> {
    let mut lines = doc.lines().skip_while(|l| !l.starts_with(ANCHOR));
    assert!(
        lines.next().is_some(),
        "{DOC} has no line starting {ANCHOR}"
    );
    let list: Vec<&str> = lines
        .skip_while(|l| !l.starts_with("- "))
        .take_while(|l| l.starts_with("- ") || l.starts_with("  "))
        .collect();
    let mut names = Vec::new();
    for line in list {
        let mut parts = line.split('`');
        parts.next();
        while let Some(name) = parts.next() {
            names.push(name.to_owned());
            parts.next();
        }
    }
    names
}

/// The failures: an unlisted crate with a root that does not forbid
/// `unsafe_code`, a listed crate with no `unsafe`, a listed name that is no
/// member.
fn check(crates: &[Crate], allowed: &[String]) -> Vec<String> {
    let mut failures = Vec::new();
    if allowed.is_empty() {
        failures.push(format!("{DOC}: no list of crates after {ANCHOR}"));
    }
    for name in allowed {
        if !crates.iter().any(|c| &c.name == name) {
            failures.push(format!(
                "{name}: listed in {DOC}, but no member is named so"
            ));
        }
    }
    for c in crates {
        let listed = allowed.contains(&c.name);
        if listed && !c.has_unsafe {
            failures.push(format!(
                "{}: listed in {DOC} but holds no unsafe; take it off the list and add {FORBID} to its root",
                c.name
            ));
        }
        if !listed {
            for (root, forbids) in &c.roots {
                if !forbids {
                    failures.push(format!(
                        "{}: {} lacks {FORBID}, and the crate is not listed in {DOC}",
                        c.name,
                        root.display()
                    ));
                }
            }
            if c.roots.is_empty() {
                failures.push(format!("{}: no crate root found", c.name));
            }
        }
    }
    failures
}

/// Whether `source` holds the keyword `unsafe` in code: comments, string and
/// character literals are skipped ([`code_only`]).
fn holds_unsafe(source: &str) -> bool {
    code_only(source)
        .split(|c: char| !is_ident(c as u8) || !c.is_ascii())
        .any(|word| word == "unsafe")
}

/// Whether a root carries the attribute on a line of its own.
fn forbids(source: &str) -> bool {
    source.lines().any(|l| l.trim() == FORBID)
}

/// Every `.rs` under `dir`, sorted.
fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut entries: Vec<_> = entries.map(|e| e.unwrap().path()).collect();
        entries.sort();
        for e in entries {
            if e.is_dir() {
                out.extend(rust_files(&e));
            } else if e.extension().is_some_and(|x| x == "rs") {
                out.push(e);
            }
        }
    }
    out
}

/// A member's package name and its crate roots, read from its manifest.
fn manifest(member: &Path) -> (String, Vec<PathBuf>) {
    let text = std::fs::read_to_string(member.join("Cargo.toml")).unwrap();
    let mut section = "";
    let mut name = None;
    let mut roots: Vec<PathBuf> = ["src/lib.rs", "src/main.rs"]
        .iter()
        .map(|r| member.join(r))
        .filter(|r| r.is_file())
        .collect();
    roots.extend(
        rust_files(&member.join("src/bin"))
            .into_iter()
            .filter(|p| p.parent() == Some(&member.join("src/bin")) || p.ends_with("main.rs")),
    );
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            section = line;
        } else if let Some((key, value)) = line.split_once('=') {
            let value = value.trim().trim_matches('"');
            match (section, key.trim()) {
                ("[package]", "name") => name = Some(value.to_owned()),
                ("[[bin]]" | "[lib]", "path") => roots.push(member.join(value)),
                _ => {}
            }
        }
    }
    roots.sort();
    roots.dedup();
    let name = name.unwrap_or_else(|| panic!("{}: no package name", member.display()));
    (name, roots)
}

/// The workspace's members, read from the `members` list of its
/// `Cargo.toml`, so a new crate is checked without being named here.
fn members(root: &Path) -> Vec<PathBuf> {
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
    let list = manifest
        .split_once("members = [")
        .and_then(|(_, rest)| rest.split_once(']'))
        .expect("the workspace manifest's members list")
        .0;
    list.split(',')
        .map(|m| m.trim().trim_matches('"'))
        .filter(|m| !m.is_empty())
        .map(|m| root.join(m))
        .collect()
}

#[test]
fn unsafe_is_only_in_the_crates_the_doc_lists() {
    let root = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/.."));
    let root = root.canonicalize().unwrap_or(root);
    let crates: Vec<Crate> = members(&root)
        .iter()
        .map(|m| {
            let (name, roots) = manifest(m);
            let roots = roots
                .into_iter()
                .map(|r| {
                    let text = std::fs::read_to_string(&r)
                        .unwrap_or_else(|e| panic!("{}: {e}", r.display()));
                    (
                        r.strip_prefix(&root).unwrap_or(&r).to_owned(),
                        forbids(&text),
                    )
                })
                .collect();
            let has_unsafe = rust_files(&m.join("src"))
                .iter()
                .any(|f| holds_unsafe(&std::fs::read_to_string(f).unwrap()));
            Crate {
                name,
                roots,
                has_unsafe,
            }
        })
        .collect();
    assert!(crates.len() > 1, "no members found");
    let allowed = allowlist(&std::fs::read_to_string(root.join(DOC)).unwrap());
    let failures = check(&crates, &allowed);
    assert!(
        failures.is_empty(),
        "{} unsafe_code failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

fn fake(name: &str, forbids: bool, has_unsafe: bool) -> Crate {
    Crate {
        name: name.to_owned(),
        roots: vec![(PathBuf::from(format!("{name}/src/lib.rs")), forbids)],
        has_unsafe,
    }
}

#[test]
fn the_list_is_read_from_the_bullets_after_the_anchor() {
    let doc = "# T\n\nSee `x`.\n\n**`unsafe`.** Prose with `y`.\n\n\
               - **Core:** `a`, `b`,\n  `c`.\n- **More:** `d`.\n\nAfter `e`.\n";
    assert_eq!(allowlist(doc), ["a", "b", "c", "d"]);
}

#[test]
fn each_drift_is_reported() {
    let crates = [
        fake("ok-forbids", true, false),
        fake("ok-listed", false, true),
        fake("unlisted", false, false),
        fake("listed-clean", false, false),
    ];
    let allowed = ["ok-listed", "listed-clean", "gone"].map(String::from);
    let f = check(&crates, &allowed);
    assert_eq!(f.len(), 3, "{f:#?}");
    assert!(f[0].starts_with("gone: listed"));
    assert!(f[1].starts_with("unlisted: unlisted/src/lib.rs lacks"));
    assert!(f[2].starts_with("listed-clean: listed"));
}

#[test]
fn unsafe_in_comments_and_strings_does_not_count() {
    assert!(!holds_unsafe(
        "// unsafe\n/* unsafe */ let s = \"unsafe\"; let r = r#\"unsafe\"#;\n\
         let c = '\"'; fn f<'a>(x: &'a u8) {} #![forbid(unsafe_code)]\n"
    ));
    assert!(holds_unsafe("let c = '\"'; let x = unsafe { f() };"));
    assert!(holds_unsafe("unsafe impl Sync for D {}"));
}
