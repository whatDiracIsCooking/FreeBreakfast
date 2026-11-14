//! The workspace graph is the selection graph, and is tested
//! (`docs/testing.md`, "Selection: the workspace graph is the selection
//! graph"; `docs/design.md`, "What it depends on, and what depends on it").
//! It reads the package graph with `guppy` (through `cargo metadata`, so
//! the manifests' own text) and fails when:
//!
//! - a member's reverse dependents pass the bar ([`bar`]), unless [`GLOBAL`]
//!   names it as allowed to be global;
//! - a member is an umbrella: it re-exports a whole workspace crate, or
//!   items of two or more of them, or its crate root holds nothing but
//!   re-exports of them;
//! - a crate that only logs ([`LOG_ONLY`]), or one that offers the logging
//!   macros by re-exporting them ([`LOG_MACROS`]), reaches a handle crate
//!   ([`HANDLES`]) through any edge;
//! - a path rule in `devtools/test.py` (the files that lie in no crate but
//!   belong to one) names a source that does not exist or a crate that is
//!   not a member, or a `kernels/` directory under `rust/` is one the rule
//!   does not cover or has no `-sys` crate wrapping it;
//! - a member declares a direct dependency its sources never name
//!   ([`check_unused`]), unless [`UNNAMED`] gives the reason.
//!
//! Tier 3: it reads text and runs `cargo metadata`, and links nothing.

#![forbid(unsafe_code)]

use fb_doc_check::{code_only, is_ident};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// The members allowed any number of reverse dependents. Each is depended
/// on by nearly every crate because nearly every crate logs or returns an
/// error, which is the C++ tree's `core` split down to its two leaves, and
/// neither reaches the handle types ([`LOG_ONLY`]).
const GLOBAL: &[&str] = &["fb-core-log", "fb-core-error-handling"];

/// The crates that only log, which must not depend on the handle types
/// (`docs/testing.md`, "Selection": "`core` is split so that a crate which
/// only logs does not depend on the handle types"). `fb-core-log` is the
/// logger, and `fb-core-error-handling` is the error value over it, which
/// its own docs say does not name WarpWraps.
const LOG_ONLY: &[&str] = &["fb-core-log", "fb-core-error-handling"];

/// The logging macros' paths, as [`reexports`] flattens them. A crate that
/// re-exports any of them offers itself as a way to log, so a crate that
/// only logs may come to depend on it, and it is held to [`LOG_ONLY`]'s
/// rule. `fb-core-diagnostics` did this while depending on `fb-core-wwr`;
/// the macros are now imported from where they are defined
/// (`core/diagnostics/src/lib.rs`).
const LOG_MACROS: &[&str] = &[
    "log::*",
    "log::trace",
    "log::debug",
    "log::info",
    "log::warn",
    "log::error",
    "log::log",
    "fb_core_log::*",
    "fb_core_log::critical",
];

/// The handle types: the safe wrappers over wwr's handles, and the C ABI
/// they wrap.
const HANDLES: &[&str] = &["fb-core-wwr", "wwr-sys"];

/// The most reverse dependents a member outside [`GLOBAL`] may have, for a
/// workspace of `members` members: two thirds of the others, rounded down.
///
/// A fraction, so the bar grows with the tree rather than being outgrown by
/// it. Two thirds is chosen from the distribution when the test landed (38
/// members, so 37 others and a bar of 24): the two global crates sat at 30
/// (`fb-core-log`) and 29 (`fb-core-error-handling`), 81% and 78%, and every
/// other member at 20 or fewer (`fb-core-wwr` 20, `fb-core-types` 18,
/// `fb-atom` 17, `fb-molecule` 16, `fb-test-support` and
/// `fb-core-thread-pool` 15), 54% at most. Two thirds sits in that gap: the
/// highest ordinary crate has a sixth of the workspace in headroom, and a
/// crate everything above the core leaves depends on (the C++ tree's
/// `free_breakfast.core`, imported by 154 of 157 GPU test files) fails.
fn bar(members: usize) -> usize {
    members.saturating_sub(1) * 2 / 3
}

/// The graph as the checks see it.
struct Graph {
    /// Each member's package name, with its directory relative to `rust/`
    /// and whether it declares `links`.
    members: BTreeMap<String, (String, bool)>,
    /// Every package's direct dependencies through any edge (normal, build
    /// or dev), by name, members and external crates alike. A dev edge is
    /// followed past its first hop too, as `devtools/test.py`'s selection
    /// follows `cargo metadata`'s resolve graph, so the bar measures the
    /// selection the gate runs.
    deps: BTreeMap<String, BTreeSet<String>>,
}

impl Graph {
    /// The members that depend on `name` through member edges alone,
    /// transitively.
    fn reverse_dependents(&self, name: &str) -> BTreeSet<String> {
        let mut seen = BTreeSet::new();
        let mut todo = vec![name.to_owned()];
        while let Some(n) = todo.pop() {
            for (user, deps) in &self.deps {
                if self.members.contains_key(user) && deps.contains(&n) && seen.insert(user.clone())
                {
                    todo.push(user.clone());
                }
            }
        }
        seen.remove(name);
        seen
    }

    /// A chain of edges from `from` to `to`, `from` first, if there is one.
    fn path(&self, from: &str, to: &str) -> Option<Vec<String>> {
        let mut parent: BTreeMap<String, String> = BTreeMap::new();
        let mut todo = std::collections::VecDeque::from([from.to_owned()]);
        while let Some(n) = todo.pop_front() {
            if n == to {
                let mut chain = vec![n];
                while let Some(p) = parent.get(chain.last().unwrap()) {
                    chain.push(p.clone());
                }
                chain.reverse();
                return Some(chain);
            }
            for d in self.deps.get(&n).into_iter().flatten() {
                if d != from && !parent.contains_key(d) {
                    parent.insert(d.clone(), n.clone());
                    todo.push_back(d.clone());
                }
            }
        }
        None
    }
}

/// The bar's failures, each naming the crate and its direct reverse edges.
fn check_bar(g: &Graph) -> Vec<String> {
    let mut failures = Vec::new();
    for name in GLOBAL {
        if !g.members.contains_key(*name) {
            failures.push(format!("{name}: in GLOBAL but no member is named so"));
        }
    }
    let limit = bar(g.members.len());
    for name in g.members.keys() {
        if GLOBAL.contains(&name.as_str()) {
            continue;
        }
        let users = g.reverse_dependents(name);
        if users.len() > limit {
            let direct: Vec<String> = g
                .deps
                .iter()
                .filter(|(u, d)| g.members.contains_key(*u) && d.contains(name))
                .map(|(u, _)| format!("{u} -> {name}"))
                .collect();
            failures.push(format!(
                "{name}: {} of the {} other members depend on it, over the bar of {limit} \
                 (two thirds); a crate everything depends on is an umbrella by another name. \
                 Its direct edges: {}",
                users.len(),
                g.members.len() - 1,
                direct.join(", ")
            ));
        }
    }
    failures
}

/// The members that offer the logging macros, each with the first
/// [`LOG_MACROS`] path it re-exports.
fn logging_facades(crates: &[Sources]) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for c in crates {
        let offered = c
            .files
            .iter()
            .flat_map(|(_, text)| reexports(text))
            .find(|path| LOG_MACROS.contains(&path.as_str()));
        if let Some(path) = offered {
            out.insert(c.name.clone(), path);
        }
    }
    out
}

/// The handle check's failures, each naming the chain of edges: for each
/// [`LOG_ONLY`] crate, and for each crate in `facades` (from
/// [`logging_facades`]).
fn check_log_only(g: &Graph, facades: &BTreeMap<String, String>) -> Vec<String> {
    let mut failures = Vec::new();
    for (name, path) in facades {
        for handle in HANDLES {
            if let Some(chain) = g.path(name, handle) {
                failures.push(format!(
                    "{name}: offers the logging macros (`pub use {path}`), so a crate that only \
                     logs may depend on it, but it depends on the handle crate {handle}: {}; \
                     drop the re-export and log through `log` and `fb-core-log` directly",
                    chain.join(" -> ")
                ));
            }
        }
    }
    for name in LOG_ONLY {
        if !g.members.contains_key(*name) {
            failures.push(format!("{name}: in LOG_ONLY but no member is named so"));
            continue;
        }
        for handle in HANDLES {
            if let Some(chain) = g.path(name, handle) {
                failures.push(format!(
                    "{name}: only logs, but depends on the handle crate {handle}: {}",
                    chain.join(" -> ")
                ));
            }
        }
    }
    failures
}

/// `use` trees flattened to paths, whitespace removed but around `as`:
/// `a::{self, b::*}` is `a::self` and `a::b::*`. A rename's alias is
/// dropped.
fn flatten(prefix: &str, tree: &str, out: &mut Vec<String>) {
    let tree = tree.trim().trim_start_matches("::");
    if let Some(open) = tree.find('{') {
        let head = &tree[..open];
        let close = tree.rfind('}').unwrap_or(tree.len());
        let inner = &tree[open + 1..close.max(open + 1)];
        let mut depth = 0;
        let mut start = 0;
        for (i, c) in inner.char_indices() {
            match c {
                '{' => depth += 1,
                '}' => depth -= 1,
                ',' if depth == 0 => {
                    flatten(&format!("{prefix}{head}"), &inner[start..i], out);
                    start = i + 1;
                }
                _ => {}
            }
        }
        flatten(&format!("{prefix}{head}"), &inner[start..], out);
    } else if !tree.is_empty() {
        let path = tree.split(" as ").next().unwrap_or(tree).trim();
        out.push(format!("{prefix}{path}").replace(' ', ""));
    }
}

/// The paths `source` re-exports with `pub use`, flattened ([`flatten`]);
/// comments and literals are skipped ([`code_only`]), and `pub(crate) use`
/// is no re-export.
fn reexports(source: &str) -> Vec<String> {
    let code = code_only(source);
    let words: Vec<&str> = code.split_whitespace().collect();
    let text = words.join(" ");
    let mut out = Vec::new();
    let mut rest = text.as_str();
    while let Some(at) = rest.find("pub use ") {
        let before = rest[..at].chars().last();
        let tail = &rest[at + "pub use ".len()..];
        let end = tail.find(';').unwrap_or(tail.len());
        if before.is_none_or(|c| !c.is_ascii_alphanumeric() && c != '_') {
            flatten("", tail[..end].trim(), &mut out);
        }
        rest = &tail[end..];
    }
    out
}

/// `source` with comments, literals, attributes and `pub use` statements
/// removed, trimmed: empty when a crate root holds nothing else.
fn beyond_reexports(source: &str) -> String {
    let code = code_only(source);
    let b = code.as_bytes();
    let mut kept = String::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'#' {
            let open = code[i..].find('[').map(|o| i + o);
            if let Some(open) = open
                && code[i + 1..open].trim_matches('!').trim().is_empty()
            {
                let mut depth = 0;
                let mut j = open;
                while j < b.len() {
                    match b[j] {
                        b'[' => depth += 1,
                        b']' => {
                            depth -= 1;
                            if depth == 0 {
                                break;
                            }
                        }
                        _ => {}
                    }
                    j += 1;
                }
                i = j + 1;
                continue;
            }
        }
        kept.push(b[i] as char);
        i += 1;
    }
    let words: Vec<&str> = kept.split_whitespace().collect();
    let mut text = words.join(" ");
    while let Some(at) = text.find("pub use ") {
        let end = text[at..].find(';').map_or(text.len(), |e| at + e + 1);
        text.replace_range(at..end, "");
    }
    text.trim().to_owned()
}

/// One member's sources, for the umbrella check.
struct Sources {
    /// The member's package name.
    name: String,
    /// Each `.rs` under `src/`, by path, with its text.
    files: Vec<(String, String)>,
    /// Its library root's text, if it has one.
    lib: Option<String>,
}

/// The umbrella check's failures. `libs` maps a workspace crate's library
/// name (`fb_core_log`) to its package name.
fn check_umbrella(crates: &[Sources], libs: &BTreeMap<String, String>) -> Vec<String> {
    let mut failures = Vec::new();
    for c in crates {
        let mut from: BTreeSet<&str> = BTreeSet::new();
        for (file, text) in &c.files {
            for path in reexports(text) {
                let root = path.split("::").next().unwrap_or(&path);
                let Some(target) = libs.get(root) else {
                    continue;
                };
                if target == &c.name {
                    continue;
                }
                from.insert(target.as_str());
                let rest = &path[root.len()..];
                if rest.is_empty() || rest == "::self" || rest == "::*" {
                    failures.push(format!(
                        "{}: {file} re-exports the whole of {target} (`pub use {path}`); \
                         depend on {target} where it is used instead",
                        c.name
                    ));
                }
            }
        }
        if from.len() >= 2 {
            failures.push(format!(
                "{}: re-exports items of {} workspace crates ({}), a prelude; \
                 each user depends on what it uses",
                c.name,
                from.len(),
                from.iter().copied().collect::<Vec<_>>().join(", ")
            ));
        }
        if let Some(lib) = &c.lib
            && !from.is_empty()
            && beyond_reexports(lib).is_empty()
        {
            failures.push(format!(
                "{}: src/lib.rs holds nothing but re-exports of {}; a crate of re-exports is \
                 an umbrella",
                c.name,
                from.iter().copied().collect::<Vec<_>>().join(", ")
            ));
        }
    }
    failures
}

/// A path rule of `devtools/test.py`'s `affected_crates`: the files it
/// covers, and the crate it gives them to.
#[derive(Debug, PartialEq)]
struct Rule {
    /// A repo-relative file (`f == "..."`) or a regex constant's name
    /// (`NAME.match(f)`).
    source: String,
    /// The pattern, for a regex constant; empty for a file.
    pattern: String,
    /// A member's directory relative to the repo root (`rust/doc_check/`),
    /// or `None` for every member that declares `links`.
    target: Option<String>,
}

/// The function that holds the path rules.
const RULES_FN: &str = "def affected_crates(";

/// The lines of `affected_crates` that give a file to a crate
/// (`f = "..."` or `hit |= kernel_crates`), so a rule [`parse_rules`] does
/// not understand is a failure rather than a rule silently unchecked.
fn rule_actions(test_py: &str) -> usize {
    let body = test_py.split_once(RULES_FN).map_or("", |(_, b)| b);
    let body = body.split("\ndef ").next().unwrap_or(body);
    body.lines()
        .map(str::trim)
        .filter(|l| l.starts_with("f = \"") || *l == "hit |= kernel_crates")
        .count()
}

/// The path rules `test_py` holds: in `affected_crates`, each
/// `if f == "<file>":` or `if NAME.match(f):` (or `elif`) followed by
/// `f = "<member dir>"` or `hit |= kernel_crates`, with each `NAME`'s
/// `re.compile(r"...")` from the module's top level.
fn parse_rules(test_py: &str) -> Vec<Rule> {
    let Some(body) = test_py.split_once(RULES_FN).map(|(_, b)| b) else {
        return Vec::new();
    };
    let body = body.split("\ndef ").next().unwrap_or(body);
    let lines: Vec<&str> = body
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let mut rules = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        let cond = line
            .strip_prefix("elif ")
            .or_else(|| line.strip_prefix("if "))
            .and_then(|c| c.strip_suffix(':'));
        let Some(cond) = cond else { continue };
        let (source, pattern) = if let Some(file) = cond.strip_prefix("f == \"") {
            (file.trim_end_matches('"').to_owned(), String::new())
        } else if let Some(name) = cond.strip_suffix(".match(f)") {
            let decl = format!("\n{name} = re.compile(r\"");
            let Some(pattern) = format!("\n{test_py}")
                .split_once(&decl)
                .and_then(|(_, rest)| rest.split_once("\")"))
                .map(|(p, _)| p.to_owned())
            else {
                continue;
            };
            (name.to_owned(), pattern)
        } else {
            continue;
        };
        let next = lines.get(i + 1).copied().unwrap_or("");
        let target = if let Some(dir) = next.strip_prefix("f = \"") {
            Some(dir.trim_end_matches('"').to_owned())
        } else if next == "hit |= kernel_crates" {
            None
        } else {
            continue;
        };
        rules.push(Rule {
            source,
            pattern,
            target,
        });
    }
    rules
}

/// One alternative of a rule's pattern as a path: `^` and `$` dropped and
/// `\.` unescaped, with `[^/]+` (one path component) split out: the part
/// before it, and the part after it if there is one.
fn literal(alternative: &str) -> (String, Option<String>) {
    let a = alternative
        .trim_start_matches('^')
        .trim_end_matches('$')
        .replace("\\.", ".");
    match a.split_once("[^/]+") {
        Some((head, tail)) => (head.to_owned(), Some(tail.to_owned())),
        None => (a, None),
    }
}

/// Every directory named `kernels` under `dir`, relative to `base`, with
/// `target` and hidden directories skipped.
fn kernels_dirs(dir: &Path, base: &Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = entries.map(|e| e.unwrap().path()).collect();
    entries.sort();
    for e in entries.into_iter().filter(|e| e.is_dir()) {
        let name = e.file_name().unwrap().to_string_lossy().into_owned();
        if name == "target" || name.starts_with('.') || name.starts_with("build") {
            continue;
        }
        if name == "kernels" {
            let rel = e.strip_prefix(base).unwrap_or(&e);
            out.push(format!("{}/", rel.to_string_lossy()));
        } else {
            kernels_dirs(&e, base, out);
        }
    }
}

/// The path rules' failures. `exists` says whether a repo-relative path
/// exists; `kernels` is every `kernels/` directory under `rust/`,
/// repo-relative.
fn check_rules(
    rules: &[Rule],
    g: &Graph,
    exists: &dyn Fn(&str) -> bool,
    kernels: &[String],
) -> Vec<String> {
    let mut failures = Vec::new();
    if !rules.iter().any(|r| r.target.is_none()) {
        failures.push(format!(
            "devtools/test.py: no kernels rule (`hit |= kernel_crates`) found in {RULES_FN}"
        ));
    }
    let member_dirs: BTreeMap<String, &str> = g
        .members
        .iter()
        .map(|(name, (dir, _))| (format!("rust/{dir}/"), name.as_str()))
        .collect();
    for r in rules {
        let edge = format!(
            "{} -> {}",
            r.source,
            r.target
                .as_deref()
                .unwrap_or("every crate that declares links")
        );
        if let Some(target) = &r.target
            && !member_dirs.contains_key(target)
        {
            failures.push(format!(
                "devtools/test.py: path rule {edge}: {target} is no member's directory"
            ));
        }
        if r.pattern.is_empty() {
            if !exists(&r.source) {
                failures.push(format!(
                    "devtools/test.py: path rule {edge}: {} does not exist",
                    r.source
                ));
            }
            continue;
        }
        for alternative in r.pattern.split('|') {
            let (head, tail) = literal(alternative);
            if !exists(&head) {
                failures.push(format!(
                    "devtools/test.py: path rule {edge}: {head} does not exist"
                ));
            }
            if tail.is_some() && r.target.is_some() {
                failures.push(format!(
                    "devtools/test.py: path rule {edge}: a pattern over a path component \
                     this test reads only for the kernels rule"
                ));
            }
        }
        if r.target.is_none() {
            for dir in kernels {
                let covered = r.pattern.split('|').any(|alt| match literal(alt) {
                    (head, Some(tail)) => dir.strip_prefix(&head).is_some_and(|rest| {
                        rest.strip_suffix(&tail)
                            .is_some_and(|m| !m.is_empty() && !m.contains('/'))
                    }),
                    (head, None) => dir.starts_with(&head),
                });
                if !covered {
                    failures.push(format!(
                        "devtools/test.py: path rule {edge}: {dir} is a kernels directory \
                         the rule does not cover"
                    ));
                }
                let sys = format!("{}sys/", dir.trim_end_matches("kernels/"));
                match member_dirs.get(&sys) {
                    Some(name) if g.members[*name].1 => {}
                    Some(name) => failures.push(format!(
                        "devtools/test.py: path rule {edge}: {dir} is wrapped by {name} \
                         ({sys}), which declares no links, so the rule does not reach it"
                    )),
                    None => failures.push(format!(
                        "devtools/test.py: path rule {edge}: {dir} has no -sys crate at {sys} \
                         wrapping it"
                    )),
                }
            }
        }
    }
    failures
}

/// The workspace root, `rust/`.
fn rust_root() -> PathBuf {
    let root = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/.."));
    root.canonicalize().unwrap_or(root)
}

/// The workspace's package graph, read with `guppy`, under `--locked` as
/// every cargo call of the gates is, so the test never rewrites
/// `Cargo.lock`.
fn read_graph(root: &Path) -> Graph {
    let graph = guppy::MetadataCommand::new()
        .manifest_path(root.join("Cargo.toml"))
        .other_options(["--locked"])
        .build_graph()
        .expect("cargo metadata over rust/Cargo.toml");
    let mut members = BTreeMap::new();
    for p in graph.workspace().iter() {
        let dir = p.manifest_path().parent().expect("a manifest's directory");
        let dir = dir
            .as_std_path()
            .strip_prefix(root)
            .unwrap_or(dir.as_std_path());
        members.insert(
            p.name().to_owned(),
            (dir.to_string_lossy().into_owned(), p.links().is_some()),
        );
    }
    let mut deps: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
    for p in graph.packages() {
        let entry = deps.entry(p.name().to_owned()).or_default();
        for link in p.direct_links() {
            entry.insert(link.to().name().to_owned());
        }
    }
    Graph { members, deps }
}

#[test]
fn no_member_passes_the_reverse_dependents_bar() {
    let failures = check_bar(&read_graph(&rust_root()));
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Every member's sources under `src/`.
fn read_sources(root: &Path, g: &Graph) -> Vec<Sources> {
    g.members
        .iter()
        .map(|(name, (dir, _))| {
            let mut paths = Vec::new();
            rust_files(&root.join(dir).join("src"), &mut paths);
            let files = paths
                .iter()
                .map(|p| {
                    let rel = p.strip_prefix(root).unwrap_or(p);
                    (
                        rel.to_string_lossy().into_owned(),
                        std::fs::read_to_string(p).unwrap(),
                    )
                })
                .collect();
            let lib = std::fs::read_to_string(root.join(dir).join("src/lib.rs")).ok();
            Sources {
                name: name.clone(),
                files,
                lib,
            }
        })
        .collect()
}

#[test]
fn a_crate_that_only_logs_reaches_no_handle_crate() {
    let root = rust_root();
    let g = read_graph(&root);
    let facades = logging_facades(&read_sources(&root, &g));
    let failures = check_log_only(&g, &facades);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn no_member_is_an_umbrella() {
    let root = rust_root();
    let g = read_graph(&root);
    let libs: BTreeMap<String, String> = g
        .members
        .keys()
        .map(|n| (n.replace('-', "_"), n.clone()))
        .collect();
    let failures = check_umbrella(&read_sources(&root, &g), &libs);
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn the_path_rules_match_the_layout() {
    let root = rust_root();
    let repo = root.parent().expect("the repository root").to_owned();
    let test_py = std::fs::read_to_string(repo.join("devtools/test.py")).expect("devtools/test.py");
    let rules = parse_rules(&test_py);
    assert_eq!(
        rules.len(),
        rule_actions(&test_py),
        "devtools/test.py: {RULES_FN} gives files to crates in a form this test does not read: {rules:?}"
    );
    let mut kernels = Vec::new();
    kernels_dirs(&root, &repo, &mut kernels);
    let failures = check_rules(
        &rules,
        &read_graph(&root),
        &|p| repo.join(p).exists(),
        &kernels,
    );
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Every `.rs` under `dir`.
fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<PathBuf> = entries.map(|e| e.unwrap().path()).collect();
    entries.sort();
    for e in entries {
        if e.is_dir() {
            rust_files(&e, out);
        } else if e.extension().is_some_and(|x| x == "rs") {
            out.push(e);
        }
    }
}

/// A graph from `(package, its dependencies)` pairs, every package a member
/// in `rust/<package>/` unless it is in `external`.
fn graph(edges: &[(&str, &[&str])], external: &[&str]) -> Graph {
    let mut g = Graph {
        members: BTreeMap::new(),
        deps: BTreeMap::new(),
    };
    for (p, ds) in edges {
        if !external.contains(p) {
            g.members
                .insert((*p).to_owned(), ((*p).to_owned(), p.ends_with("-sys")));
        }
        g.deps.insert(
            (*p).to_owned(),
            ds.iter().map(|d| (*d).to_owned()).collect(),
        );
    }
    g
}

/// Catches a flattening: a new crate that every other member depends on
/// passes the bar, while the global crates do not count against it.
#[test]
fn a_crate_everything_depends_on_fails_the_bar() {
    let mut edges: Vec<(String, Vec<&str>)> = vec![
        ("fb-core-log".into(), vec![]),
        ("fb-core-error-handling".into(), vec!["fb-core-log"]),
        ("fb-prelude".into(), vec!["fb-core-error-handling"]),
    ];
    for i in 0..6 {
        edges.push((format!("fb-m{i}"), vec!["fb-prelude", "fb-core-log"]));
    }
    let edges: Vec<(&str, &[&str])> = edges
        .iter()
        .map(|(p, d)| (p.as_str(), d.as_slice()))
        .collect();
    let failures = check_bar(&graph(&edges, &[]));
    assert_eq!(failures.len(), 1, "{failures:?}");
    assert!(
        failures[0].starts_with("fb-prelude: 6 of the 8"),
        "{}",
        failures[0]
    );
    assert!(
        failures[0].contains("fb-m0 -> fb-prelude"),
        "{}",
        failures[0]
    );
}

/// Catches the split collapsing: the logger reaching the handle types
/// through another crate is named with the whole chain.
#[test]
fn a_logger_reaching_the_handles_fails_with_the_chain() {
    let g = graph(
        &[
            ("fb-core-log", &["fb-mid"]),
            ("fb-core-error-handling", &["fb-core-log"]),
            ("fb-mid", &["fb-core-wwr"]),
            ("fb-core-wwr", &["wwr-sys"]),
            ("wwr-sys", &[]),
        ],
        &["wwr-sys"],
    );
    let failures = check_log_only(&g, &BTreeMap::new());
    assert_eq!(failures.len(), 4, "{failures:?}");
    assert!(
        failures.contains(
            &"fb-core-log: only logs, but depends on the handle crate fb-core-wwr: \
          fb-core-log -> fb-mid -> fb-core-wwr"
                .to_owned()
        )
    );
}

/// Catches the old `fb-core-diagnostics`: a crate that re-exports the
/// logging macros while depending on `fb-core-wwr`. A crate that depends on
/// it without offering them, or whose re-export is commented out, passes,
/// and so does one that offers them and reaches no handle crate.
#[test]
fn a_logging_facade_reaching_the_handles_fails() {
    let src = |name: &str, lib: &str| Sources {
        name: name.to_owned(),
        files: vec![("src/lib.rs".to_owned(), lib.to_owned())],
        lib: Some(lib.to_owned()),
    };
    let crates = [
        src(
            "fb-core-diagnostics",
            "pub use ::log::{debug, error, info, trace, warn};\n\
             pub use fb_core_log::critical;\npub fn gpu_must() {}\n",
        ),
        src("fb-gpu", "pub fn gpu_must() {}\n"),
        src("fb-quiet", "// pub use log::info;\npub fn f() {}\n"),
        src("fb-logs", "pub use fb_core_log::critical;\n"),
    ];
    let facades = logging_facades(&crates);
    assert_eq!(
        facades.keys().map(String::as_str).collect::<Vec<_>>(),
        ["fb-core-diagnostics", "fb-logs"]
    );
    assert_eq!(facades["fb-core-diagnostics"], "log::debug");
    let g = graph(
        &[
            ("fb-core-log", &["log"]),
            ("fb-core-error-handling", &["fb-core-log"]),
            ("fb-core-wwr", &["fb-core-error-handling", "wwr-sys"]),
            ("wwr-sys", &[]),
            ("log", &[]),
            (
                "fb-core-diagnostics",
                &["fb-core-log", "fb-core-wwr", "log"],
            ),
            ("fb-gpu", &["fb-core-wwr"]),
            ("fb-quiet", &["fb-core-wwr"]),
            ("fb-logs", &["fb-core-log"]),
        ],
        &["wwr-sys", "log"],
    );
    let failures = check_log_only(&g, &facades);
    assert_eq!(failures.len(), 2, "{failures:#?}");
    assert!(
        failures[0].starts_with(
            "fb-core-diagnostics: offers the logging macros (`pub use log::debug`), so a \
             crate that only logs may depend on it, but it depends on the handle crate \
             fb-core-wwr: fb-core-diagnostics -> fb-core-wwr;"
        ),
        "{}",
        failures[0]
    );
    assert!(
        failures[1].contains("wwr-sys: fb-core-diagnostics -> fb-core-wwr -> wwr-sys"),
        "{}",
        failures[1]
    );
}

/// Catches each umbrella shape, and leaves a crate that re-exports its own
/// modules and one foreign item alone.
#[test]
fn umbrellas_fail_and_ordinary_reexports_pass() {
    let libs: BTreeMap<String, String> = ["fb_a", "fb_b", "fb_c", "fb_d", "fb_e", "fb_f"]
        .iter()
        .map(|l| ((*l).to_owned(), l.replace('_', "-")))
        .collect();
    let src = |name: &str, lib: &str| Sources {
        name: name.to_owned(),
        files: vec![("src/lib.rs".to_owned(), lib.to_owned())],
        lib: Some(lib.to_owned()),
    };
    let crates = [
        src("fb-a", "pub use fb_b::*;\n"),
        src(
            "fb-b",
            "//! one item\n#![forbid(unsafe_code)]\npub use ::fb_c::Thing;\n",
        ),
        src("fb-c", "pub use {fb_d::X, fb_e::{self as e}};\nfn f() {}\n"),
        src(
            "fb-d",
            "mod m;\npub use m::{Y, Z};\npub use fb_e::critical;\n/* pub use fb_f::*; */\npub fn g() {}\n",
        ),
        src("fb-e", "pub(crate) use fb_f::*;\npub fn h() {}\n"),
    ];
    let failures = check_umbrella(&crates, &libs);
    let names: Vec<&str> = failures
        .iter()
        .map(|f| f.split(':').next().unwrap())
        .collect();
    assert_eq!(
        names,
        ["fb-a", "fb-a", "fb-b", "fb-c", "fb-c"],
        "{failures:#?}"
    );
}

/// Catches a path rule that drifted from the layout: a rule naming a crate
/// that is not a member, a source that is gone, and a kernels directory
/// with no `-sys` crate.
#[test]
fn path_rules_parse_and_fail_on_drift() {
    let test_py = "\
KERNELS_PATH = re.compile(r\"^rust/[^/]+/kernels/\")
DOC_PATHS = re.compile(r\"^rust/docs/|^rust/README\\.md$\")


def affected_crates(files):
    for f in files:
        if KERNELS_PATH.match(f):
            hit |= kernel_crates
            continue
        if f == \"devtools/gone.py\":
            f = \"rust/doc_check/\"
        elif DOC_PATHS.match(f):
            f = \"rust/doc_chek/\"


def other():
    if f == \"x\":
        f = \"y\"
";
    let rules = parse_rules(test_py);
    assert_eq!(rules.len(), 3, "{rules:?}");
    assert_eq!(rule_actions(test_py), 3);
    assert_eq!(rules[2].pattern, "^rust/docs/|^rust/README\\.md$");
    let g = Graph {
        members: [
            ("fb-doc-check".to_owned(), ("doc_check".to_owned(), false)),
            (
                "fb-recipes-sys".to_owned(),
                ("recipes/sys".to_owned(), true),
            ),
        ]
        .into(),
        deps: BTreeMap::new(),
    };
    let exists = |p: &str| ["rust/", "rust/docs/", "rust/README.md"].contains(&p);
    let kernels = [
        "rust/recipes/kernels/".to_owned(),
        "rust/fft/kernels/".to_owned(),
    ];
    let failures = check_rules(&rules, &g, &exists, &kernels);
    assert_eq!(failures.len(), 3, "{failures:#?}");
    assert!(failures[0].contains("rust/fft/kernels/ has no -sys crate at rust/fft/sys/"));
    assert!(failures[1].contains("devtools/gone.py does not exist"));
    assert!(failures[2].contains("DOC_PATHS -> rust/doc_chek/: rust/doc_chek/ is no member"));
}

/// The direct dependencies a member may declare without naming them in its
/// sources, as `(member, dependency, reason)`. Each is a package name; the
/// reason is what the edge does that no `use` shows (a `links` crate kept so
/// its link line reaches the binary, a feature switched on in another
/// crate). An entry whose edge is gone, or whose dependency is now named,
/// fails, so the list only holds what it must.
const UNNAMED: &[(&str, &str, &str)] = &[];

/// Where a member's sources name things, each `code_only`: comments and
/// literals blanked, so a crate named in a doc comment or a string is not
/// used.
struct Named {
    /// Every `.rs` under `src/`: the library, the binaries and their unit
    /// tests.
    src: String,
    /// `build.rs`.
    build: String,
    /// Every `.rs` under `tests/`, `benches/` and `examples/`.
    tests: String,
}

/// One direct dependency edge of a member.
struct Dep {
    /// The dependency's package name (`fb-core-wwr`).
    package: String,
    /// The name the member's code calls it by (`fb_core_wwr`): the library
    /// name with any rename applied.
    name: String,
    /// Declared under `[dependencies]`.
    normal: bool,
    /// Declared under `[build-dependencies]`.
    build: bool,
    /// Declared under `[dev-dependencies]`.
    dev: bool,
}

/// Whether `code` holds `name` as a whole identifier.
fn names(code: &str, name: &str) -> bool {
    let b = code.as_bytes();
    code.match_indices(name).any(|(at, _)| {
        let before = at.checked_sub(1).map(|i| b[i]);
        let after = b.get(at + name.len()).copied();
        !before.is_some_and(is_ident) && !after.is_some_and(is_ident)
    })
}

/// The unused-dependency check's failures for one member. A dependency is
/// used where its kind can reach: a normal one in `src/` (named only in the
/// tests, it is a dev-dependency that every reverse dependent inherits), a
/// build one in `build.rs`, and a dev-only one in `src/`, `tests/`,
/// `benches/` or `examples/`. A dev entry beside a normal one adds features
/// to it and needs no use of its own.
fn check_unused(member: &str, deps: &[Dep], code: &Named) -> Vec<String> {
    let mut failures = Vec::new();
    for d in deps {
        if UNNAMED
            .iter()
            .any(|(m, p, _)| *m == member && *p == d.package)
        {
            continue;
        }
        let in_src = names(&code.src, &d.name);
        let in_tests = names(&code.tests, &d.name);
        if d.normal && !in_src {
            failures.push(if in_tests {
                format!(
                    "{member}: depends on {} under [dependencies], but only its tests name \
                     `{}`; move it to [dev-dependencies], so its reverse dependents do not \
                     inherit the edge",
                    d.package, d.name
                )
            } else {
                format!(
                    "{member}: depends on {} under [dependencies], but no source in src/ \
                     names `{}`; remove the edge, or add it to UNNAMED with the reason",
                    d.package, d.name
                )
            });
        }
        if d.build && !names(&code.build, &d.name) {
            failures.push(format!(
                "{member}: depends on {} under [build-dependencies], but build.rs does not \
                 name `{}`; remove the edge, or add it to UNNAMED with the reason",
                d.package, d.name
            ));
        }
        if d.dev && !d.normal && !in_src && !in_tests {
            failures.push(format!(
                "{member}: depends on {} under [dev-dependencies], but no source in src/, \
                 tests/, benches/ or examples/ names `{}`; remove the edge, or add it to \
                 UNNAMED with the reason",
                d.package, d.name
            ));
        }
    }
    failures
}

/// Every member's directory relative to `rust/` and its direct
/// dependencies, by member, read with `guppy` as [`read_graph`] reads them.
fn read_deps(root: &Path) -> BTreeMap<String, (String, Vec<Dep>)> {
    let graph = guppy::MetadataCommand::new()
        .manifest_path(root.join("Cargo.toml"))
        .other_options(["--locked"])
        .build_graph()
        .expect("cargo metadata over rust/Cargo.toml");
    let mut out = BTreeMap::new();
    for p in graph.workspace().iter() {
        let dir = p.manifest_path().parent().expect("a manifest's directory");
        let dir = dir
            .as_std_path()
            .strip_prefix(root)
            .unwrap_or(dir.as_std_path());
        let deps = p
            .direct_links()
            .map(|l| Dep {
                package: l.to().name().to_owned(),
                name: l.resolved_name().to_owned(),
                normal: l.normal().is_present(),
                build: l.build().is_present(),
                dev: l.dev().is_present(),
            })
            .collect();
        out.insert(
            p.name().to_owned(),
            (dir.to_string_lossy().into_owned(), deps),
        );
    }
    out
}

/// The intra-doc link targets in `source`'s doc comments, one per line:
/// the path in `` [`a::b`] ``, `[a::b]`, `[text](a::b)` or
/// ``[text](`a::b`)``. rustdoc resolves each against the crate's
/// dependencies, so a crate a doc comment links to is used, and as a normal
/// dependency, since `cargo doc` builds a library without its
/// dev-dependencies.
fn doc_links(source: &str) -> String {
    let mut out = Vec::new();
    for line in source.lines() {
        let t = line.trim_start();
        if !(t.starts_with("///") || t.starts_with("//!")) {
            continue;
        }
        for (at, open) in t.match_indices(['[', '(']) {
            if open == "(" && !t[..at].ends_with(']') {
                continue;
            }
            let rest = t[at + 1..].trim_start_matches('`');
            let end = rest
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_' || c == ':'))
                .unwrap_or(rest.len());
            if rest[..end].contains("::") {
                out.push(&rest[..end]);
            }
        }
    }
    out.join("\n")
}

/// What the check reads of one source: its code ([`code_only`]), then its
/// [`doc_links`].
fn named_in(source: &str) -> String {
    format!("{}\n{}", code_only(source), doc_links(source))
}

/// [`named_in`] of every `.rs` under each of `dirs`, or of `dirs`
/// themselves when they are files.
fn code_under(dirs: &[PathBuf]) -> String {
    let mut paths = Vec::new();
    for d in dirs {
        if d.is_file() {
            paths.push(d.clone());
        } else {
            rust_files(d, &mut paths);
        }
    }
    paths
        .iter()
        .map(|p| named_in(&std::fs::read_to_string(p).unwrap()))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn every_direct_dependency_is_named_in_its_crates_sources() {
    let root = rust_root();
    let members = read_deps(&root);
    let mut failures = Vec::new();
    for (member, dep, _) in UNNAMED {
        let Some((dir, deps)) = members.get(*member) else {
            failures.push(format!("UNNAMED: {member} is no member"));
            continue;
        };
        let Some(d) = deps.iter().find(|d| d.package == *dep) else {
            failures.push(format!("UNNAMED: {member} does not depend on {dep}"));
            continue;
        };
        let dir = root.join(dir);
        let all =
            code_under(&["src", "tests", "benches", "examples", "build.rs"].map(|s| dir.join(s)));
        if names(&all, &d.name) {
            failures.push(format!(
                "UNNAMED: {member} names `{}` now; drop its entry",
                d.name
            ));
        }
    }
    for (member, (dir, deps)) in &members {
        let dir = root.join(dir);
        let code = Named {
            src: code_under(&[dir.join("src")]),
            build: code_under(&[dir.join("build.rs")]),
            tests: code_under(&["tests", "benches", "examples"].map(|s| dir.join(s))),
        };
        failures.extend(check_unused(member, deps, &code));
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// Catches each kind of unused edge, a crate named only in a comment, a
/// string, a longer identifier or a doc comment's code span among them,
/// and leaves alone a use in a nested path, an intra-doc link, a dev entry
/// that only adds features, and a dev-dependency a unit test names.
#[test]
fn unused_dependencies_fail_and_used_ones_pass() {
    let dep = |name: &str, normal, build, dev| Dep {
        package: name.replace('_', "-"),
        name: name.to_owned(),
        normal,
        build,
        dev,
    };
    let deps = [
        dep("fb_used", true, false, true),
        dep("fb_only_tested", true, false, false),
        dep("fb_unused", true, false, false),
        dep("fb_prefix", true, false, false),
        dep("fb_build", false, true, false),
        dep("fb_unit", false, false, true),
        dep("fb_dev_unused", false, false, true),
        dep("fb_linked", true, false, false),
        dep("fb_linked_too", true, false, false),
        dep("fb_in_parens", true, false, false),
    ];
    let code = Named {
        src: named_in(
            "//! Runs after [`fb_linked::add`], or [it](fb_linked_too::Thing).\n\
             //! Not a link: (`fb_in_parens::x`).\n\
             use std::{io, fb_used::x};\n// fb_unused\nconst S: &str = \"fb_unused\";\n\
             fn f() { fb_prefix_more::g(); }\n#[cfg(test)] mod t { use fb_unit::*; }\n",
        ),
        build: named_in("fn main() { fb_used::h(); }\n"),
        tests: named_in("use fb_only_tested as t;\n"),
    };
    let failures = check_unused("fb-m", &deps, &code);
    let named: Vec<&str> = failures
        .iter()
        .map(|f| f.split_whitespace().nth(3).unwrap())
        .collect();
    assert_eq!(
        named,
        [
            "fb-only-tested",
            "fb-unused",
            "fb-prefix",
            "fb-build",
            "fb-dev-unused",
            "fb-in-parens"
        ],
        "{failures:#?}"
    );
    assert!(
        failures[0].contains("move it to [dev-dependencies]"),
        "{}",
        failures[0]
    );
}
