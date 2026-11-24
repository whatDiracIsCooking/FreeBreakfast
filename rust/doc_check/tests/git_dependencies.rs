//! A git dependency is pinned once, by its commit (`docs/testing.md`, "What
//! runs when"). It fails when:
//!
//! - a git dependency of `rust/Cargo.toml`'s `[workspace.dependencies]`
//!   has no `rev`, a `rev` that is not a full 40-hex commit, or a `branch`
//!   or `tag`, any of which `cargo update` could move;
//! - a member's manifest names `git` itself, in any dependency table,
//!   rather than inheriting the pin with `workspace = true`;
//! - a manifest under `experimental/rust/` (a prototype, a workspace of
//!   its own) names a repository the workspace pins at another `rev`.
//!
//! Tier 3: it reads manifests and links nothing.

#![forbid(unsafe_code)]

use fb_doc_check::{rust_root, workspace_members};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use toml::{Table, Value};

/// A manifest, parsed.
fn read(path: &Path) -> Table {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    toml::from_str(&text).unwrap_or_else(|e| panic!("{}: {e}", path.display()))
}

/// Every dependency table of a manifest, as its key path and its entries:
/// `[dependencies]`, `[dev-dependencies]`, `[build-dependencies]`, the same
/// under each `[target.*]`, and `[workspace.dependencies]`.
fn dependency_tables(manifest: &Table) -> Vec<(String, &Table)> {
    const KINDS: [&str; 3] = ["dependencies", "dev-dependencies", "build-dependencies"];
    let mut out = Vec::new();
    for kind in KINDS {
        if let Some(t) = manifest.get(kind).and_then(Value::as_table) {
            out.push((kind.to_string(), t));
        }
    }
    if let Some(targets) = manifest.get("target").and_then(Value::as_table) {
        for (cfg, target) in targets {
            for kind in KINDS {
                if let Some(t) = target.get(kind).and_then(Value::as_table) {
                    out.push((format!("target.{cfg}.{kind}"), t));
                }
            }
        }
    }
    if let Some(t) = manifest
        .get("workspace")
        .and_then(|w| w.get("dependencies"))
        .and_then(Value::as_table)
    {
        out.push(("workspace.dependencies".to_string(), t));
    }
    out
}

/// Each git dependency of `tables`: its key path, its name, and its entry.
fn git_dependencies<'a>(tables: &[(String, &'a Table)]) -> Vec<(String, String, &'a Table)> {
    let mut out = Vec::new();
    for (path, table) in tables {
        for (name, dep) in *table {
            if let Some(dep) = dep.as_table().filter(|d| d.contains_key("git")) {
                out.push((path.clone(), name.clone(), dep));
            }
        }
    }
    out
}

/// A string field of a dependency entry.
fn field<'a>(dep: &'a Table, key: &str) -> Option<&'a str> {
    dep.get(key).and_then(Value::as_str)
}

/// Whether `rev` is a full commit: 40 lowercase hex digits.
fn is_full_commit(rev: &str) -> bool {
    rev.len() == 40
        && rev
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

/// The workspace's git pins, repository to `rev`, and a failure for each
/// git dependency of `workspace` that is not pinned by a full commit, or
/// that is not in `[workspace.dependencies]`.
fn check_workspace(workspace: &Table, name: &str) -> (BTreeMap<String, String>, Vec<String>) {
    let mut pins = BTreeMap::new();
    let mut failures = Vec::new();
    for (path, dep_name, dep) in git_dependencies(&dependency_tables(workspace)) {
        if path != "workspace.dependencies" {
            failures.push(format!(
                "{name}: [{path}] {dep_name} names git; write it in [workspace.dependencies]"
            ));
            continue;
        }
        let url = field(dep, "git").unwrap_or_default().to_string();
        for moving in ["branch", "tag"] {
            if dep.contains_key(moving) {
                failures.push(format!(
                    "{name}: {dep_name} names a {moving}, which cargo update moves; pin a rev"
                ));
            }
        }
        match field(dep, "rev") {
            Some(rev) if is_full_commit(rev) => {
                pins.insert(url, rev.to_string());
            }
            rev => failures.push(format!(
                "{name}: {dep_name}'s rev is {rev:?}, not a full 40-hex commit"
            )),
        }
    }
    (pins, failures)
}

/// A failure for each git dependency a member names itself.
fn check_member(manifest: &Table, name: &str) -> Vec<String> {
    git_dependencies(&dependency_tables(manifest))
        .into_iter()
        .map(|(path, dep_name, _)| {
            format!(
                "{name}: [{path}] {dep_name} names git; inherit the pin with \
                 `{dep_name}.workspace = true`"
            )
        })
        .collect()
}

/// A failure for each git dependency of a prototype's manifest that names
/// a repository in `pins` at another `rev`.
fn check_prototype(manifest: &Table, name: &str, pins: &BTreeMap<String, String>) -> Vec<String> {
    let mut failures = Vec::new();
    for (path, dep_name, dep) in git_dependencies(&dependency_tables(manifest)) {
        let Some(pin) = field(dep, "git").and_then(|url| pins.get(url)) else {
            continue;
        };
        let rev = field(dep, "rev");
        if rev != Some(pin.as_str()) {
            failures.push(format!(
                "{name}: [{path}] {dep_name} is at rev {rev:?}, rust/Cargo.toml pins {pin}; \
                 move it or retire the prototype"
            ));
        }
    }
    failures
}

/// Every `Cargo.toml` under `dir`, `target/` excluded, into `out`.
fn manifests(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries {
        let p = e.unwrap().path();
        if p.is_dir() {
            if p.file_name().is_some_and(|n| n != "target") {
                manifests(&p, out);
            }
        } else if p.file_name().is_some_and(|n| n == "Cargo.toml") {
            out.push(p);
        }
    }
}

#[test]
fn every_git_dependency_is_pinned_once_by_its_commit() {
    let root = rust_root();
    let (pins, mut failures) = check_workspace(&read(&root.join("Cargo.toml")), "rust/Cargo.toml");
    for member in workspace_members() {
        let path = member.join("Cargo.toml");
        let name = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .display()
            .to_string();
        failures.extend(check_member(&read(&path), &name));
    }
    let repo = root.parent().expect("rust/ has a parent");
    let mut prototypes = Vec::new();
    manifests(&repo.join("experimental/rust"), &mut prototypes);
    prototypes.sort();
    for path in &prototypes {
        let name = path
            .strip_prefix(repo)
            .unwrap_or(path)
            .display()
            .to_string();
        failures.extend(check_prototype(&read(path), &name, &pins));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The checkers on manifests that break each clause, so a checker that
/// passes everything is caught.
#[test]
fn a_moved_or_unpinned_git_dependency_fails() {
    let url = "https://example.invalid/flow";
    let commit = "f184bcabac5ab01dec0def8c3f369d1bbdb3bbaf";
    let parse = |s: &str| -> Table { toml::from_str(s).unwrap() };

    let pinned = parse(&format!(
        "[workspace.dependencies]\nflow = {{ git = \"{url}\", rev = \"{commit}\" }}\n"
    ));
    let (pins, failures) = check_workspace(&pinned, "ws");
    assert!(failures.is_empty(), "{failures:?}");
    assert_eq!(pins.get(url).map(String::as_str), Some(commit));

    for (bad, why) in [
        (format!("flow = {{ git = \"{url}\" }}"), "no rev"),
        (
            format!("flow = {{ git = \"{url}\", rev = \"f184bca\" }}"),
            "a short rev",
        ),
        (
            format!("flow = {{ git = \"{url}\", branch = \"main\" }}"),
            "a branch",
        ),
        (
            format!("flow = {{ git = \"{url}\", rev = \"{commit}\", tag = \"v1\" }}"),
            "a tag beside the rev",
        ),
    ] {
        let ws = parse(&format!("[workspace.dependencies]\n{bad}\n"));
        assert!(!check_workspace(&ws, "ws").1.is_empty(), "{why} passed");
    }

    for table in [
        "dependencies",
        "dev-dependencies",
        "target.'cfg(unix)'.dependencies",
    ] {
        let member = parse(&format!(
            "[{table}]\nflow = {{ git = \"{url}\", rev = \"{commit}\" }}\n"
        ));
        assert_eq!(
            check_member(&member, "m").len(),
            1,
            "a member's [{table}] git passed"
        );
    }
    let inherits = parse("[dependencies]\nflow.workspace = true\n");
    assert!(check_member(&inherits, "m").is_empty());

    let at = |rev: &str| {
        parse(&format!(
            "[dependencies]\nflow = {{ git = \"{url}\", rev = \"{rev}\" }}\n"
        ))
    };
    assert!(check_prototype(&at(commit), "p", &pins).is_empty());
    let old = "b77e5474091d84bc49808245b9ac3f9004070a3f";
    assert_eq!(
        check_prototype(&at(old), "p", &pins).len(),
        1,
        "a moved pin passed"
    );
    let elsewhere = parse(&format!(
        "[dependencies]\nother = {{ git = \"https://example.invalid/other\", rev = \"{old}\" }}\n"
    ));
    assert!(check_prototype(&elsewhere, "p", &pins).is_empty());
}
