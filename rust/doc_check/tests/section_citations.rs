//! Every section the workspace's documents and comments cite by name is a
//! heading of the file cited (`docs/documentation.md`, "Narrative: mdBook
//! beside the rustdoc"). The workspace's first tier-3 check; it touches no
//! device, and its crate does not link the wwr library.
//!
//! A citation is a run of files and quoted section names joined by commas,
//! starting at a file: a Markdown link to a `.md` (resolved against the
//! citing file's directory) or a `.md` path in single backticks (resolved
//! against the workspace root, `rust/`). Each name resolves against the nearest file
//! before it in the run. In a Markdown document, a name followed by
//! `, below` or `, above` cites that document itself.
//!
//! A name matches a heading when it is the whole heading, or exactly the
//! part before the heading's first `": "`, and it must match exactly one
//! heading in the file: a short name two headings share is a failure, not a
//! guess. Headings are ATX (`#` to `######`) lines outside code fences.
//! Citations inside code fences or inline code are examples, not citations.

#![forbid(unsafe_code)]

use std::collections::HashMap;
use std::path::{Component, Path, PathBuf};

/// Which shape a name was cited in; the real tree must use all three, so a
/// scanner that matched nothing fails rather than passes.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
enum Form {
    Link,
    Code,
    SelfRef,
}

struct Report {
    failures: Vec<String>,
    counts: HashMap<Form, usize>,
}

/// Checks every citation in `sources` (paths under `root`, with their text);
/// `read` fetches a cited file's text.
fn check(
    sources: &[(PathBuf, String)],
    root: &Path,
    read: &dyn Fn(&Path) -> Option<String>,
) -> Report {
    let mut headings: HashMap<PathBuf, Option<Vec<String>>> = HashMap::new();
    let mut report = Report {
        failures: Vec::new(),
        counts: HashMap::new(),
    };
    for (path, source) in sources {
        let text = logical_text(path, source);
        let line_starts: Vec<usize> = std::iter::once(0)
            .chain(text.match_indices('\n').map(|(i, _)| i + 1))
            .collect();
        for c in scan(path, &text, root) {
            *report.counts.entry(c.form).or_default() += 1;
            let heads = headings
                .entry(c.target.clone())
                .or_insert_with(|| read(&c.target).map(|t| heading_list(&t)));
            let reason = match heads {
                None => Some("no such file".to_owned()),
                Some(heads) => resolve(&c.name, heads).err(),
            };
            if let Some(reason) = reason {
                let line = line_starts.partition_point(|&s| s <= c.pos);
                report.failures.push(format!(
                    "{}:{line}: {}, \"{}\": {reason}",
                    path.strip_prefix(root).unwrap_or(path).display(),
                    c.written,
                    c.name
                ));
            }
        }
    }
    report
}

/// The name matches exactly one heading, whole or by its short name.
fn resolve(name: &str, heads: &[String]) -> Result<(), String> {
    let hits: Vec<&String> = heads
        .iter()
        .filter(|h| *h == name || h.split_once(": ").is_some_and(|(s, _)| s == name))
        .collect();
    match hits.len() {
        0 => Err("no such heading".to_owned()),
        1 => Ok(()),
        _ => Err(format!(
            "ambiguous, it matches {}",
            hits.iter()
                .map(|h| format!("\"{h}\""))
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// The text citations are read from, one line per source line so line
/// numbers carry over: a Rust file's comments with their `//`, `///` or
/// `//!` dropped and its code blanked, and in either kind, fenced blocks
/// blanked.
fn logical_text(path: &Path, source: &str) -> String {
    let rust = path.extension().is_some_and(|e| e == "rs");
    let mut fenced = false;
    let mut out = String::with_capacity(source.len());
    for line in source.lines() {
        let line = if rust {
            let t = line.trim_start();
            ["//!", "///", "//"]
                .iter()
                .find_map(|p| t.strip_prefix(p))
                .unwrap_or("")
        } else {
            line
        };
        let t = line.trim_start();
        if t.starts_with("```") || t.starts_with("~~~") {
            fenced = !fenced;
        } else if !fenced {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
}

/// A Markdown file's headings, whitespace collapsed.
fn heading_list(source: &str) -> Vec<String> {
    logical_text(Path::new("x.md"), source)
        .lines()
        .filter_map(|line| {
            let t = line.trim_start();
            let hashes = t.len() - t.trim_start_matches('#').len();
            let rest = t[hashes..].strip_prefix(' ')?;
            ((1..=6).contains(&hashes)).then(|| squeeze(rest.trim_end_matches('#')))
        })
        .collect()
}

fn squeeze(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

struct Citation {
    pos: usize,
    form: Form,
    target: PathBuf,
    written: String,
    name: String,
}

/// Inline code spans as (start, end, backtick count), CommonMark-style: a
/// run of n backticks closes at the next run of exactly n, within a
/// paragraph.
fn code_spans(b: &[u8]) -> Vec<(usize, usize, usize)> {
    let run = |i: usize| b[i..].iter().take_while(|&&c| c == b'`').count();
    let mut spans = Vec::new();
    let mut i = 0;
    while i < b.len() {
        if b[i] != b'`' {
            i += 1;
            continue;
        }
        let n = run(i);
        let mut j = i + n;
        let mut close = None;
        while j < b.len() {
            if b[j..].starts_with(b"\n\n") {
                break;
            }
            if b[j] == b'`' {
                let m = run(j);
                if m == n {
                    close = Some(j + m);
                    break;
                }
                j += m;
            } else {
                j += 1;
            }
        }
        match close {
            Some(end) => {
                spans.push((i, end, n));
                i = end;
            }
            None => i += n,
        }
    }
    spans
}

fn scan(path: &Path, text: &str, root: &Path) -> Vec<Citation> {
    let b = text.as_bytes();
    let spans = code_spans(b);
    let inside = |p: usize| {
        let k = spans.partition_point(|s| s.0 < p);
        k > 0 && p < spans[k - 1].1
    };
    let dir = path.parent().unwrap_or(root);
    // A file reference at p: (resolved path, as written, end, form).
    let file_at = |p: usize| -> Option<(PathBuf, String, usize, Form)> {
        if b[p] == b'[' && !inside(p) {
            let close = p + text[p..].find(']')?;
            let rest = text[close + 1..].strip_prefix('(')?;
            let target = &rest[..rest.find(')')?];
            let end = close + 2 + target.len() + 1;
            let written = format!("{}({target})", &text[p..=close]);
            let ok = target.ends_with(".md") && !target.contains(char::is_whitespace);
            return ok.then(|| (normalize(&dir.join(target)), written, end, Form::Link));
        }
        let &(start, end, n) = spans.get(spans.partition_point(|s| s.0 < p))?;
        let target = &text[start + 1..end - 1];
        let ok = start == p
            && n == 1
            && target.ends_with(".md")
            && !target.contains(char::is_whitespace);
        ok.then(|| {
            (
                normalize(&root.join(target)),
                format!("`{target}`"),
                end,
                Form::Code,
            )
        })
    };
    // A quoted name at p: (name, end).
    let quote_at = |p: usize| -> Option<(String, usize)> {
        let len = text[p + 1..].find('"')?;
        let raw = &text[p + 1..p + 1 + len];
        (b[p] == b'"' && len < 200 && !raw.contains("\n\n")).then(|| (squeeze(raw), p + len + 2))
    };
    // After a list element ending at e, the start of the next one.
    let next = |e: usize| -> Option<usize> {
        let rest = text[e..].strip_prefix(',')?;
        let skip = rest.len() - rest.trim_start().len();
        Some(e + 1 + skip).filter(|&q| q < b.len())
    };
    let markdown = path.extension().is_some_and(|e| e == "md");
    let mut out = Vec::new();
    let mut p = 0;
    while p < b.len() {
        if let Some((mut target, mut written, mut end, mut form)) = file_at(p) {
            while let Some(q) = next(end) {
                if let Some((name, e)) = quote_at(q) {
                    out.push(Citation {
                        pos: q,
                        form,
                        target: target.clone(),
                        written: written.clone(),
                        name,
                    });
                    end = e;
                } else if let Some(f) = file_at(q) {
                    (target, written, end, form) = f;
                } else {
                    break;
                }
            }
            p = end;
        } else if markdown && b[p] == b'"' && !inside(p) {
            let Some((name, end)) = quote_at(p) else {
                p += 1;
                continue;
            };
            let after = next(end).map(|q| &text[q..]).unwrap_or("");
            let word = after.split(|c: char| !c.is_alphanumeric()).next();
            if matches!(word, Some("below" | "above")) {
                out.push(Citation {
                    pos: p,
                    form: Form::SelfRef,
                    target: normalize(path),
                    written: "(this file)".to_owned(),
                    name,
                });
            }
            p = end;
        } else {
            p += 1;
        }
    }
    out
}

/// `a/b/../c` as `a/c`, without touching the file system.
fn normalize(p: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in p.components() {
        match c {
            Component::ParentDir => {
                out.pop();
            }
            Component::CurDir => {}
            c => out.push(c),
        }
    }
    out
}

/// The workspace's documents and Rust sources: its own files, and every
/// member's `build.rs`, `src/` and `tests/`.
fn workspace_sources(root: &Path) -> Vec<(PathBuf, String)> {
    fn walk(p: &Path, out: &mut Vec<PathBuf>) {
        if p.is_dir() {
            let mut entries: Vec<_> = std::fs::read_dir(p)
                .unwrap()
                .map(|e| e.unwrap().path())
                .collect();
            entries.sort();
            for e in entries {
                walk(&e, out);
            }
        } else if p.is_file() && p.extension().is_some_and(|e| e == "md" || e == "rs") {
            out.push(p.to_owned());
        }
    }
    let mut paths = Vec::new();
    for top in ["README.md", "capi_dir.rs", "docs"] {
        walk(&root.join(top), &mut paths);
    }
    for member in members(root) {
        for top in ["build.rs", "src", "tests"] {
            walk(&member.join(top), &mut paths);
        }
    }
    paths
        .into_iter()
        .map(|p| {
            let text = std::fs::read_to_string(&p).unwrap();
            (p, text)
        })
        .collect()
}

/// The workspace's members, read from the `members` list of its
/// `Cargo.toml`, so a new crate is scanned without being named here.
fn members(root: &Path) -> Vec<PathBuf> {
    let manifest = std::fs::read_to_string(root.join("Cargo.toml")).unwrap();
    let list = manifest
        .split_once("members = [")
        .and_then(|(_, rest)| rest.split_once(']'))
        .expect("the workspace manifest's members list")
        .0;
    let members: Vec<PathBuf> = list
        .split(',')
        .map(|m| m.trim().trim_matches('"'))
        .filter(|m| !m.is_empty())
        .map(|m| root.join(m))
        .collect();
    assert!(
        members.iter().all(|m| m.join("Cargo.toml").is_file()),
        "a workspace member without a manifest: {members:?}"
    );
    members
}

#[test]
fn every_section_citation_names_a_heading() {
    let root = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/.."));
    let root = root.canonicalize().unwrap_or(root);
    let sources = workspace_sources(&root);
    let report = check(&sources, &root, &|p| std::fs::read_to_string(p).ok());
    assert!(
        report.failures.is_empty(),
        "{} broken section citation(s):\n{}",
        report.failures.len(),
        report.failures.join("\n")
    );
    for form in [Form::Link, Form::Code, Form::SelfRef] {
        assert!(
            report.counts.get(&form).copied().unwrap_or(0) > 0,
            "no {form:?} citation found: the scanner has stopped matching ({:?})",
            report.counts
        );
    }
}

/// Runs `check` over in-memory files under `/r`; the first file is the
/// citing one.
fn check_fake(files: &[(&str, &str)]) -> Vec<String> {
    let root = PathBuf::from("/r");
    let map: HashMap<PathBuf, String> = files
        .iter()
        .map(|(p, t)| (root.join(p), (*t).to_owned()))
        .collect();
    let (citing, text) = files[0];
    let sources = [(root.join(citing), text.to_owned())];
    check(&sources, &root, &|p| map.get(p).cloned()).failures
}

#[test]
fn a_broken_citation_is_reported() {
    let f = check_fake(&[
        ("docs/a.md", "# A\n\nSee\n[`b.md`](b.md), \"Nope\".\n"),
        ("docs/b.md", "# B\n\n## Yes\n"),
    ]);
    assert_eq!(
        f,
        ["docs/a.md:4: [`b.md`](b.md), \"Nope\": no such heading"]
    );
}

#[test]
fn a_missing_file_is_reported() {
    let f = check_fake(&[("docs/a.md", "([`gone.md`](gone.md), \"X\")\n")]);
    assert_eq!(
        f,
        ["docs/a.md:1: [`gone.md`](gone.md), \"X\": no such file"]
    );
}

#[test]
fn a_short_name_is_the_exact_text_before_the_colon() {
    let b = "## Selection: the graph\n\n## Memory: one\n\n## Memory: two\n";
    let cite = |name: &str| {
        check_fake(&[
            ("docs/a.md", &format!("([`b.md`](b.md), \"{name}\")\n")),
            ("docs/b.md", b),
        ])
    };
    assert!(cite("Selection").is_empty());
    assert!(cite("Selection: the graph").is_empty());
    assert_eq!(cite("Select").len(), 1, "a prefix is not a short name");
    let f = cite("Memory");
    assert_eq!(f.len(), 1);
    assert!(f[0].contains("ambiguous"), "{f:?}");
    assert!(cite("Memory: two").is_empty());
}

#[test]
fn each_name_resolves_against_the_nearest_file_before_it() {
    let files = |text: &str| {
        check_fake(&[
            ("docs/a.md", text),
            ("docs/b.md", "## In b\n"),
            ("docs/c.md", "## In c\n"),
        ])
    };
    let ok = "([`b.md`](b.md), \"In b\", [`c.md`](c.md), \"In c\")\n";
    assert!(files(ok).is_empty());
    let both = "([`b.md`](b.md), [`c.md`](c.md), \"In c\")\n";
    assert!(files(both).is_empty());
    let swapped = "([`b.md`](b.md), \"In c\", [`c.md`](c.md), \"In b\")\n";
    let f = files(swapped);
    assert_eq!(f.len(), 2, "{f:?}");
    assert!(f[0].contains("b.md") && f[0].contains("In c"), "{f:?}");
    assert!(f[1].contains("c.md") && f[1].contains("In b"), "{f:?}");
}

#[test]
fn rust_comments_cite_from_the_crate_root_across_lines() {
    let rs = "// (`docs/b.md`, \"Two\n    /// words\", \"Gone\")\n\
              let s = \"(`docs/b.md`, \\\"Not a comment\\\")\";\n";
    let f = check_fake(&[("src/x.rs", rs), ("docs/b.md", "## Two words\n")]);
    assert_eq!(f, ["src/x.rs:2: `docs/b.md`, \"Gone\": no such heading"]);
}

#[test]
fn fences_and_inline_code_are_examples() {
    let a = "``[`x.md`](x.md), \"Section\"`` is the shape.\n\n\
             ```\n([`b.md`](b.md), \"Nope\")\n```\n\n\
             ([`b.md`](b.md), \"Real\")\n";
    let b = "```\n# not a heading\n```\n\n## Real\n";
    assert!(check_fake(&[("docs/a.md", a), ("docs/b.md", b)]).is_empty());
    let c = "([`b.md`](b.md), \"not a heading\")\n";
    assert_eq!(check_fake(&[("docs/a.md", c), ("docs/b.md", b)]).len(), 1);
}

#[test]
fn below_and_above_cite_the_same_document() {
    let a = "# A\n\nSee \"Later\", below, and \"Gone\", above.\n\
             A \"quote\", then prose.\n\n## Later: more\n";
    let f = check_fake(&[("docs/a.md", a)]);
    assert_eq!(f, ["docs/a.md:3: (this file), \"Gone\": no such heading"]);
}
