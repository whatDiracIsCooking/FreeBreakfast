//! A doc comment's figure is a hand-written SVG beside its module, pulled
//! onto the rustdoc page inline with `#[doc = include_str!("….svg")]`
//! (`docs/documentation.md`, "Diagrams and animations"). rustdoc passes the
//! file through Markdown as one raw HTML block, and renders it under three
//! themes, so of every `.svg` under a member's `src/` this checks:
//!
//! - **It is on a page**: a `#[doc = include_str!(…)]` of the workspace
//!   names it. A figure nothing includes is a picture no reader sees.
//! - **It is one HTML block**: it opens with `<figure` at its first byte,
//!   closes with `</figure>`, and has no blank line, which would end the
//!   block and hand the rest to Markdown as text.
//! - **Its tags balance**, so it cannot swallow the page after it.
//! - **It is read without the picture**: its `<svg>` carries `role="img"`
//!   and an `aria-label`, and it has a `<figcaption>`.
//! - **Its ink is the theme's**: it names `currentColor`, and a `#rrggbb`
//!   colour appears only as the fallback of a `var(--…, #…)`.
//! - **Its ids are its own**: no `id` is shared with another figure, since
//!   several land on one page and a `url(#…)` takes the first.
//! - **The figures share one `<style>`**, byte for byte: each file carries
//!   it, so a figure renders alone, and the page applies whichever came
//!   first to all of them.
//!
//! What it misses: whether a label is true to the code, and whether two
//! labels overlap; a figure is looked at when it is drawn. Tier 3: it reads
//! text and links nothing.

#![forbid(unsafe_code)]

use fb_doc_check::{rust_root, workspace_members, workspace_sources};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Component, Path, PathBuf};

/// Every `.svg` under `dir`, into `out`.
fn svg_files(dir: &Path, out: &mut BTreeSet<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for e in entries {
        let p = e.unwrap().path();
        if p.is_dir() {
            svg_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "svg") {
            out.insert(p);
        }
    }
}

/// `path` with its `.` and `..` components folded, without touching the
/// disk.
fn folded(path: &Path) -> PathBuf {
    let mut out = PathBuf::new();
    for c in path.components() {
        match c {
            Component::CurDir => {}
            Component::ParentDir => {
                out.pop();
            }
            c => out.push(c),
        }
    }
    out
}

/// The `.svg` files a source's `#[doc = include_str!("…")]` attributes
/// name, each as its path from the root, given the source's own.
fn included(source_path: &str, source: &str) -> Vec<String> {
    const OPEN: &str = "#[doc = include_str!(\"";
    let dir = Path::new(source_path).parent().unwrap_or(Path::new(""));
    source
        .match_indices(OPEN)
        .filter_map(|(at, _)| {
            let rest = &source[at + OPEN.len()..];
            let name = &rest[..rest.find('"')?];
            name.ends_with(".svg")
                .then(|| folded(&dir.join(name)).display().to_string())
        })
        .collect()
}

/// The text between each `open` and the next `close` in `text`.
fn between<'a>(text: &'a str, open: &str, close: &str) -> Vec<&'a str> {
    text.match_indices(open)
        .filter_map(|(at, _)| {
            let rest = &text[at + open.len()..];
            Some(&rest[..rest.find(close)?])
        })
        .collect()
}

/// Why `text`'s tags do not balance, if they do not: a `>` inside a quoted
/// attribute is skipped, and a `<style>`'s text is not read for tags.
fn unbalanced(text: &str) -> Option<String> {
    let b = text.as_bytes();
    let mut open: Vec<&str> = Vec::new();
    let mut i = 0;
    while let Some(at) = text[i..].find('<').map(|k| i + k) {
        let mut end = at + 1;
        let mut quote = None;
        while end < b.len() && (quote.is_some() || b[end] != b'>') {
            match (quote, b[end]) {
                (None, q @ (b'"' | b'\'')) => quote = Some(q),
                (Some(q), c) if c == q => quote = None,
                _ => {}
            }
            end += 1;
        }
        if end >= b.len() {
            return Some("a tag is never closed with `>`".to_string());
        }
        let tag = &text[at + 1..end];
        let name = tag
            .trim_start_matches('/')
            .split(|c: char| c.is_whitespace() || c == '/')
            .next()
            .unwrap_or("");
        if tag.starts_with('/') {
            match open.pop() {
                Some(o) if o == name => {}
                o => return Some(format!("`</{name}>` closes {o:?}")),
            }
        } else if !tag.ends_with('/') {
            open.push(name);
        }
        i = end + 1;
        if name == "style" && !tag.starts_with('/') {
            i += text[i..].find("</style>").unwrap_or(0);
        }
    }
    open.last().map(|o| format!("`<{o}>` is never closed"))
}

/// The hex colours of `text` that are not a `var()`'s fallback, nor a
/// `url(#…)` that happens to read as one.
fn bare_colours(text: &str) -> Vec<&str> {
    text.match_indices('#')
        .filter_map(|(at, _)| {
            let digits = text[at + 1..]
                .bytes()
                .take_while(u8::is_ascii_hexdigit)
                .count();
            let after = text.as_bytes().get(at + 1 + digits);
            let whole = after.is_none_or(|c| !c.is_ascii_alphanumeric() && *c != b'_');
            let before = text[..at].trim_end();
            let fallback = before.ends_with(',') || before.ends_with("url(");
            (matches!(digits, 3 | 4 | 6 | 8) && whole && !fallback)
                .then(|| &text[at..at + 1 + digits])
        })
        .collect()
}

/// What is wrong with one figure's text, each a line naming `path`.
fn figure_failures(path: &str, text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut fail = |why: String| out.push(format!("{path}: {why}"));
    if !text.starts_with("<figure") || !text.trim_end().ends_with("</figure>") {
        fail("is not one `<figure>…</figure>` from its first byte".to_string());
    }
    if let Some(n) = text.lines().position(|l| l.trim().is_empty()) {
        fail(format!(
            "line {} is blank: Markdown ends the HTML block there",
            n + 1
        ));
    }
    if let Some(why) = unbalanced(text) {
        fail(format!("its tags do not balance: {why}"));
    }
    let svgs = between(text, "<svg", ">");
    if svgs.len() != 1 || !svgs[0].contains("role=\"img\"") || !svgs[0].contains("aria-label=\"") {
        fail("wants one `<svg>` with `role=\"img\"` and an `aria-label`".to_string());
    }
    if !text.contains("<figcaption>") {
        fail("has no `<figcaption>`".to_string());
    }
    if !text.contains("currentColor") {
        fail("names no `currentColor`: its ink is not the theme's".to_string());
    }
    for c in bare_colours(text) {
        fail(format!(
            "hard-codes the colour `{c}` outside a `var()` fallback"
        ));
    }
    out
}

/// What is wrong across the figures, `(path, text)` each: a shared `id`,
/// or a `<style>` that differs from the first figure's.
fn shared_failures(figures: &[(String, String)]) -> Vec<String> {
    let mut out = Vec::new();
    let mut ids: BTreeMap<&str, &str> = BTreeMap::new();
    for (path, text) in figures {
        for id in between(text, " id=\"", "\"") {
            if let Some(first) = ids.insert(id, path) {
                out.push(format!("{path}: id `{id}` is also in {first}"));
            }
        }
    }
    let style = |text: &'_ str| between(text, "<style>", "</style>").join("\n");
    if let Some((first, text)) = figures.first() {
        let shared = style(text);
        for (path, text) in &figures[1..] {
            if style(text) != shared {
                out.push(format!("{path}: its `<style>` is not {first}'s"));
            }
        }
    }
    out
}

#[test]
fn every_doc_figure_is_an_inline_svg_a_theme_can_ink() {
    let root = rust_root();
    let mut files = BTreeSet::new();
    for m in workspace_members() {
        svg_files(&m.join("src"), &mut files);
    }
    let on_a_page: BTreeSet<String> = workspace_sources()
        .iter()
        .flat_map(|(p, s)| included(p, s))
        .collect();
    let figures: Vec<(String, String)> = files
        .iter()
        .map(|f| {
            let rel = f.strip_prefix(&root).unwrap_or(f).display().to_string();
            (rel, std::fs::read_to_string(f).unwrap())
        })
        .collect();
    let mut failures = Vec::new();
    for (path, text) in &figures {
        if !on_a_page.contains(path) {
            failures.push(format!(
                "{path}: no `#[doc = include_str!(…)]` puts it on a page"
            ));
        }
        failures.extend(figure_failures(path, text));
    }
    failures.extend(shared_failures(&figures));
    assert!(
        failures.is_empty(),
        "{} doc figure failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
    // The scan found the figures it is for: a moved directory fails here,
    // not silently.
    assert!(
        figures
            .iter()
            .any(|(p, _)| p == "steps/src/steps/figures/eri_holds.svg"),
        "the EriManager's figures were not found under a member's src/"
    );
}

#[test]
fn each_way_a_figure_goes_wrong_is_found() {
    let good = "<figure class=\"f\">\n<svg viewBox=\"0 0 1 1\" role=\"img\" aria-label=\"a > b\">\n\
                <style>.f .a{stroke:currentColor;fill:var(--x, #fff)}</style>\n\
                <defs><marker id=\"m1\"/></defs>\n<rect class=\"a\"/>\n</svg>\n\
                <figcaption>It.</figcaption>\n</figure>\n";
    assert_eq!(figure_failures("a.svg", good), Vec::<String>::new());
    let one = |text: &str, why: &str| {
        let f = figure_failures("a.svg", text);
        assert!(
            f.len() == 1 && f[0].contains(why),
            "wanted one failure with {why:?}, got {f:?}"
        );
    };
    one(&good.replace("</svg>\n", "</svg>\n\n"), "line 7 is blank");
    one(
        &good.replace("<rect class=\"a\"/>", "<g>"),
        "do not balance",
    );
    one(&good.replace(" role=\"img\"", ""), "role=\"img\"");
    one(
        &good.replace("<figcaption>It.</figcaption>\n", ""),
        "no `<figcaption>`",
    );
    one(&good.replace("currentColor", "black"), "currentColor");
    one(&good.replace("var(--x, #fff)", "#fff"), "the colour `#fff`");
    one(&format!("<p>c</p>\n{good}"), "from its first byte");

    let two = vec![
        ("a.svg".to_string(), good.to_string()),
        ("b.svg".to_string(), good.replace(".f .a", ".f .b")),
    ];
    let f = shared_failures(&two);
    assert_eq!(f.len(), 2, "{f:?}");
    assert!(f[0].contains("id `m1` is also in a.svg"));
    assert!(f[1].contains("its `<style>` is not a.svg's"));

    let source = "/// A.\n#[doc = include_str!(\"figures/../figures/a.svg\")]\n\
                  #![doc = include_str!(\"../README.md\")]\nstruct A;\n";
    assert_eq!(
        included("steps/src/x.rs", source),
        ["steps/src/figures/a.svg"]
    );
}
