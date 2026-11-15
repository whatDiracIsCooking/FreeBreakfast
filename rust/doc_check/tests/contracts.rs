//! Contracts are written once, and only as contracts (`docs/testing.md`,
//! "Contracts are part of the spec").
//!
//! - **No doc comment restates a contract.** rustdoc renders every
//!   contract, debug (`#[debug_requires]`, `#[debug_ensures]`,
//!   `#[debug_invariant]`) or always-on (`#[requires]`, `#[ensures]`,
//!   `#[invariant]`), as the item's `§ Contracts` section, its message and
//!   its expression
//!   (`docs/documentation.md`, "Code: rustdoc"), so a sentence in the doc
//!   comment saying the same is a second copy that drifts. Two shapes are
//!   refused: the old restating phrase, anywhere in a doc comment, and a doc
//!   comment that holds the message of a contract on its own item.
//! - **A `debug_assert!` is an internal invariant, and says so.** A caller's
//!   precondition is a contract, where rustdoc shows it: a
//!   `#[debug_requires]`, or a `#[requires]` when soundness rests on it; a
//!   safety check on a computed value is an `assert!`, named in `# Panics`.
//!   What is left
//!   for `debug_assert!` (and its `_eq` and `_ne` forms) is a fact of the
//!   function's own making that no caller can break, and each one carries a
//!   `// INVARIANT:` comment directly above it saying why.
//!
//! - **A leading precondition is a `#[requires]`.** In a `pub fn`, an
//!   `assert!`, `assert_eq!`, `assert_ne!` or `if … { panic!(…) }` before
//!   any other statement, whose condition names only the parameters, is a
//!   plain condition on the arguments, so it is written where rustdoc
//!   shows it, whatever its message prints: the value checks carry the
//!   values. A site kept on purpose is listed in `KEPT`, with why.
//!
//! - **A value check is a whole `&&` term of a contract's condition.**
//!   `fb_core_log`'s `le!`, `lt!`, `eq!` and `fits_i32!` store their
//!   operands when false, for the panic hook to print, so a false one must
//!   end in its contract's panic. Each appears only inside a contract
//!   attribute (any of the six), and there only as the whole condition or a
//!   whole term of a top-level `&&` chain of it, parentheses aside: never
//!   negated (`!le!(…)` stores exactly when the check passes), never under
//!   `||`, in a closure or as an operand, any of which can leave a false
//!   check's values for the thread's next panic.
//!
//! clippy's `disallowed_macros` cannot hold the second rule: a
//! `#[debug_requires]` expands to a `debug_assert!` spanned at the
//! contract, so the lint flags every contract in the tree.
//!
//! Every `.rs` under every workspace member (the `members` of
//! `rust/Cargo.toml`) is read, `target/` excluded.

#![forbid(unsafe_code)]

use fb_doc_check::{
    attributes, code_only, is_ident, line_of, string_literals, words, workspace_sources,
};

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
/// `fb_core_log`'s value checks, which store their operands when false.
const VALUE_CHECKS: [&str; 4] = ["le", "lt", "eq", "fits_i32"];
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
        // A plain comment between attributes (a `// RELEASE:`) is inside
        // the run.
        if line.trim_start().starts_with("//") {
            i += 1;
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
                     precondition is a #[debug_requires] or, when soundness rests on it, a \
                     #[requires]",
                    n + 1
                ));
            }
        }
    }
    failures
}

/// Whether `path` is a test target's or a test module's file, by the
/// tree's layout: under a `tests/`, `benches/` or `examples/` directory,
/// or a `tests.rs` or `*_tests.rs` module.
fn test_file(path: &str) -> bool {
    let name = path.rsplit('/').next().unwrap_or(path);
    path.split('/')
        .any(|d| matches!(d, "tests" | "benches" | "examples"))
        || name == "tests.rs"
        || name.ends_with("_tests.rs")
}

/// The byte index of the bracket closing the one opening at `open` in
/// `b`, counting `(`, `[` and `{` alike, or `b.len()`.
fn closing(b: &[u8], open: usize) -> usize {
    let mut depth = 0i32;
    for (j, &c) in b.iter().enumerate().skip(open) {
        match c {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => {
                depth -= 1;
                if depth == 0 {
                    return j;
                }
            }
            _ => {}
        }
    }
    b.len()
}

/// The byte spans of `code`'s top-level comma-separated parts, each
/// trimmed as `text` reads, an empty part (a trailing comma) dropped. The
/// commas are read from `code` (`code_only`), so one in a string splits
/// nothing, and the trimming from `text`, the same span with its literals
/// kept (`without_comments`), so a part that is all string is not empty.
fn top_level_args(code: &str, text: &str) -> Vec<(usize, usize)> {
    let b = code.as_bytes();
    let mut out = Vec::new();
    let (mut depth, mut from) = (0i32, 0);
    let mut push = |from: usize, to: usize| {
        let part = &text[from..to];
        let lead = part.len() - part.trim_start().len();
        let len = part.trim().len();
        if len > 0 {
            out.push((from + lead, from + lead + len));
        }
    };
    for (j, &c) in b.iter().enumerate() {
        match c {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            // A closure's `|a, b|` and a generic's `<A, B>` are rare in a
            // condition; a split inside one only makes it unseen.
            b',' if depth == 0 => {
                push(from, j);
                from = j + 1;
            }
            _ => {}
        }
    }
    push(from, b.len());
    out
}

/// The identifiers of `code` with their byte offsets, numbers left out.
fn idents(code: &str) -> Vec<(usize, &str)> {
    words(code)
        .into_iter()
        .filter(|(_, w)| !w.as_bytes()[0].is_ascii_digit())
        .collect()
}

/// The byte spans of the `#[cfg(test)]` modules of one file (any `cfg`
/// naming `test` and not negating it, on a `mod` with a body), or the whole
/// file under an inner `#![cfg(test)]`.
fn test_modules(code: &str) -> Vec<(usize, usize)> {
    let b = code.as_bytes();
    let attrs = attributes(code);
    let mut out = Vec::new();
    for (k, a) in attrs.iter().enumerate() {
        let names = idents(&code[a.start..a.end]);
        let names_test = names.iter().any(|(_, w)| *w == "test");
        if a.name != "cfg" || !names_test || names.iter().any(|(_, w)| *w == "not") {
            continue;
        }
        if a.inner {
            return vec![(0, b.len())];
        }
        // The item after this attribute and any that follow it.
        let mut rest = a.end;
        for next in &attrs[k + 1..] {
            if code[rest..next.start].trim().is_empty() {
                rest = next.end;
            } else {
                break;
            }
        }
        let is_mod = idents(&code[rest..])
            .iter()
            .take(3)
            .any(|(_, w)| *w == "mod");
        let Some(brace) = code[rest..].find(['{', ';']).map(|j| rest + j) else {
            continue;
        };
        if is_mod && b[brace] == b'{' {
            out.push((rest, closing(b, brace)));
        }
    }
    out
}

/// The words a condition may hold that are no parameter: keywords and
/// literals.
const CONDITION_WORDS: [&str; 6] = ["as", "true", "false", "self", "Self", "crate"];

/// The leading preconditions the tree keeps as body checks, each
/// `path:function` with why. An entry whose site is gone fails
/// [`a_leading_precondition_is_a_requires`], so the list only shrinks.
const KEPT: &[(&str, &str)] = &[(
    "localization/sys/src/lib.rs:add_pm_start",
    "should move: follow-up (n_ao fits an i32, n_occ in range; no wave of #1636-#1638 \
     covered fb-localization-sys)",
)];

/// One leading check at the head of a public function's body that could
/// be a `#[requires]`: `path:function`, and the line it is on.
struct Leading {
    key: String,
    line: usize,
}

/// Whether `cond` names nothing lower-case but `params`: a name after `.`
/// or `::`, a call, a macro, a type, a keyword and an upper-case constant
/// aside.
fn only_params(cond: &str, params: &[&str]) -> bool {
    let cb = cond.as_bytes();
    idents(cond).iter().all(|&(j, w)| {
        let before = cond[..j].trim_end();
        let after = cond[j + w.len()..].trim_start();
        params.contains(&w)
            || CONDITION_WORDS.contains(&w)
            || cb[j].is_ascii_uppercase()
            || before.ends_with('.')
            || before.ends_with("::")
            || after.starts_with('(')
            || after.starts_with("::")
            || after.starts_with('!')
            || is_primitive(w)
    })
}

/// Whether `body`, a block's inside (`code_only`), is one `panic!` call and
/// nothing else, a trailing `;` aside.
fn only_a_panic(body: &str) -> bool {
    let t = body.trim();
    let Some(after) = t.strip_prefix("panic") else {
        return false;
    };
    let Some(after) = after.trim_start().strip_prefix('!') else {
        return false;
    };
    let after = after.trim_start();
    if !after.starts_with(['(', '[', '{']) {
        return false;
    }
    let open = t.len() - after.len();
    let close = closing(t.as_bytes(), open);
    matches!(t[close + 1..].trim(), "" | ";")
}

/// The leading preconditions of one file's public functions: each check in
/// the run that opens a `pub fn`'s body, before any other statement, whose
/// condition names nothing lower-case but the function's parameters
/// ([`only_params`]). A check is an `assert!`, `assert_eq!` or
/// `assert_ne!`, whatever its message, or an `if <condition> { panic!(…) }`
/// with no `else`; the run ends at the first statement that is neither.
/// Test files and `#[cfg(test)]` modules are skipped, and so is a
/// `const fn`, which `#[requires]` cannot carry.
fn leading_preconditions(path: &str, source: &str) -> Vec<Leading> {
    if test_file(path) {
        return Vec::new();
    }
    let code = code_only(source);
    let b = code.as_bytes();
    let skipped = test_modules(&code);
    let words = idents(&code);
    let mut out = Vec::new();
    for (k, &(at, w)) in words.iter().enumerate() {
        if w != "fn" || skipped.iter().any(|&(s, e)| s <= at && at < e) {
            continue;
        }
        // `pub fn`, `pub const fn`, `pub unsafe fn`, …, not `pub(crate)`.
        let head = code[..at]
            .rfind(['\n', ';', '{', '}', ']'])
            .map_or(0, |p| p + 1);
        let vis: Vec<&str> = code[head..at].split_whitespace().collect();
        if vis.first() != Some(&"pub") || vis.contains(&"const") {
            continue;
        }
        let Some(&(name_at, name)) = words.get(k + 1) else {
            continue;
        };
        // The parameter list, past any generics.
        let after_name = name_at + name.len();
        let Some(paren) = code[after_name..].find('(').map(|j| after_name + j) else {
            continue;
        };
        let close = closing(b, paren);
        let list = &code[paren + 1..close];
        let params: Vec<&str> = top_level_args(list, list)
            .iter()
            .filter_map(|&(s, e)| {
                let pat = list[s..e].split(':').next().unwrap_or("");
                idents(pat)
                    .into_iter()
                    .map(|(_, w)| w)
                    .rfind(|w| *w != "mut")
            })
            .collect();
        let Some(open) = code[close..].find(['{', ';']).map(|j| close + j) else {
            continue;
        };
        if b[open] == b';' {
            continue;
        }
        let end = closing(b, open);
        // The run of checks that opens the body.
        let mut i = open + 1;
        loop {
            let t = &code[i..end];
            i += t.len() - t.trim_start().len();
            let rest = &code[i..end];
            let line = code[..i].matches('\n').count() + 1;
            let starts_word = |w: &str| {
                rest.starts_with(w)
                    && !rest[w.len()..].starts_with(|c: char| c.is_alphanumeric() || c == '_')
            };
            if starts_word("if") {
                // `if <cond> { panic!(…) }`, the condition up to the first
                // brace outside brackets.
                let mut depth = 0i32;
                let mut brace = None;
                for (j, &c) in b.iter().enumerate().take(end).skip(i + 2) {
                    match c {
                        b'(' | b'[' => depth += 1,
                        b')' | b']' => depth -= 1,
                        b'{' if depth == 0 => {
                            brace = Some(j);
                            break;
                        }
                        _ => {}
                    }
                }
                let Some(brace) = brace else {
                    break;
                };
                let cond = &code[i + 2..brace];
                let shut = closing(b, brace);
                let tail = code[shut + 1..end].trim_start();
                let guard = !cond.trim_start().starts_with("let ")
                    && only_a_panic(&code[brace + 1..shut])
                    && !(tail.starts_with("else")
                        && !tail[4..].starts_with(|c: char| c.is_alphanumeric() || c == '_'));
                if !guard {
                    break;
                }
                if only_params(cond, &params) {
                    out.push(Leading {
                        key: format!("{path}:{name}"),
                        line,
                    });
                }
                i = shut + 1;
                continue;
            }
            let mac = ["assert_eq", "assert_ne", "assert"]
                .into_iter()
                .find(|m| rest.starts_with(m) && rest[m.len()..].trim_start().starts_with('!'));
            let Some(mac) = mac else {
                break;
            };
            let bang = i + rest.find('!').unwrap_or(0);
            let Some(p) = code[bang..end].find(['(', '[', '{']).map(|j| bang + j) else {
                break;
            };
            let q = closing(b, p);
            let n_cond = if mac == "assert" { 1 } else { 2 };
            let inside = &code[p + 1..q];
            let cond = top_level_args(inside, inside)
                .iter()
                .take(n_cond)
                .map(|&(s, e)| &inside[s..e])
                .collect::<Vec<_>>()
                .join(" ");
            if only_params(&cond, &params) {
                out.push(Leading {
                    key: format!("{path}:{name}"),
                    line,
                });
            }
            i = q + 1;
            if code[i..end].trim_start().starts_with(';') {
                i += code[i..end].find(';').unwrap_or(0) + 1;
            } else {
                break;
            }
        }
    }
    out
}

/// Whether `w` is a primitive type's name, as in `u32::MAX` or `as usize`.
fn is_primitive(w: &str) -> bool {
    matches!(
        w,
        "u8" | "u16"
            | "u32"
            | "u64"
            | "u128"
            | "usize"
            | "i8"
            | "i16"
            | "i32"
            | "i64"
            | "i128"
            | "isize"
            | "f32"
            | "f64"
            | "bool"
            | "char"
    )
}

/// The failures of the leading-precondition rule over `found`: every
/// leading precondition not `kept`, and every `kept` entry with no site.
fn leading_failures(found: &[Leading], kept: &[(&str, &str)]) -> Vec<String> {
    let mut failures: Vec<String> = found
        .iter()
        .filter(|l| !kept.iter().any(|&(k, _)| k == l.key))
        .map(|l| {
            let (path, name) = l.key.rsplit_once(':').unwrap_or((&l.key, ""));
            format!(
                "{path}:{}: {name} opens with a check on its arguments; write it as #[requires]",
                l.line
            )
        })
        .collect();
    failures.extend(
        kept.iter()
            .filter(|&&(k, _)| !found.iter().any(|l| l.key == k))
            .map(|(k, _)| format!("{k}: kept, but no leading precondition is there; drop it")),
    );
    failures
}

/// The byte spans of the value-check calls in `code` (`code_only`), each
/// from its name to its closing bracket: a [`VALUE_CHECKS`] name, not after
/// `.`, then `!` (not `!=`), then a bracket.
fn value_checks(code: &str) -> Vec<(usize, usize)> {
    let b = code.as_bytes();
    let mut out = Vec::new();
    for (at, w) in idents(code) {
        if !VALUE_CHECKS.contains(&w) || code[..at].trim_end().ends_with('.') {
            continue;
        }
        let rest = &code[at + w.len()..];
        let Some(after) = rest.trim_start().strip_prefix('!') else {
            continue;
        };
        if !after.trim_start().starts_with(['(', '[', '{']) {
            continue;
        }
        let open = code.len() - after.trim_start().len();
        out.push((at, closing(b, open) + 1));
    }
    out
}

/// The contract attributes of `code` (`code_only`), each as the byte span
/// of its argument list, inside the parentheses.
fn contract_args(code: &str) -> Vec<(usize, usize)> {
    let b = code.as_bytes();
    attributes(code)
        .iter()
        .filter(|a| CONTRACTS.contains(&a.name.as_str()))
        .filter_map(|a| {
            let open = a.start + code[a.start..a.end].find('(')?;
            Some((open + 1, closing(b, open)))
        })
        .collect()
}

/// The value checks of one file outside every contract attribute.
fn misplaced_value_checks(path: &str, source: &str) -> Vec<String> {
    let code = code_only(source);
    let contracts = contract_args(&code);
    value_checks(&code)
        .into_iter()
        .filter(|&(s, e)| !contracts.iter().any(|&(cs, ce)| cs <= s && e <= ce))
        .map(|(s, e)| {
            format!(
                "{path}:{}: {} outside a contract attribute; a false check stores its values \
                 for this thread's next panic, so it is written only in a contract's condition",
                line_of(&code, s),
                &code[s..e]
            )
        })
        .collect()
}

/// `(from, to)` with whitespace and every pair of parentheses enclosing the
/// whole of it trimmed off, in `code`.
fn bare(code: &str, (mut from, mut to): (usize, usize)) -> (usize, usize) {
    loop {
        let part = &code[from..to];
        from += part.len() - part.trim_start().len();
        to = from + part.trim().len();
        if to > from && code.as_bytes()[from] == b'(' && closing(code.as_bytes(), from) == to - 1 {
            from += 1;
            to -= 1;
        } else {
            return (from, to);
        }
    }
}

/// The top-level `&&` terms of `code[from..to]`, as byte spans of `code`,
/// each [`bare`].
fn and_terms(code: &str, from: usize, to: usize) -> Vec<(usize, usize)> {
    let b = code.as_bytes();
    let mut out = Vec::new();
    let (mut depth, mut start) = (0i32, from);
    let mut j = from;
    while j < to {
        match b[j] {
            b'(' | b'[' | b'{' => depth += 1,
            b')' | b']' | b'}' => depth -= 1,
            b'&' if depth == 0 && b.get(j + 1) == Some(&b'&') => {
                out.push(bare(code, (start, j)));
                start = j + 2;
                j += 1;
            }
            _ => {}
        }
        j += 1;
    }
    out.push(bare(code, (start, to)));
    out
}

/// The value checks of one file's contracts that are not a whole `&&` term
/// of their condition: negated, under `||`, in a closure, an operand.
fn buried_value_checks(path: &str, source: &str) -> Vec<String> {
    let code = code_only(source);
    let checks = value_checks(&code);
    let mut failures = Vec::new();
    for (from, to) in contract_args(&code) {
        let inside = &code[from..to];
        for (s, e) in top_level_args(inside, inside) {
            let (s, e) = (from + s, from + e);
            // A nested term may be a parenthesised chain of its own.
            let mut terms = and_terms(&code, s, e);
            let mut k = 0;
            while k < terms.len() {
                let (ts, te) = terms[k];
                let inner = and_terms(&code, ts, te);
                if inner.len() > 1 {
                    terms.extend(inner);
                }
                k += 1;
            }
            for &(cs, ce) in checks.iter().filter(|&&(cs, ce)| s <= cs && ce <= e) {
                if !terms.contains(&(cs, ce)) {
                    failures.push(format!(
                        "{path}:{}: {} is not a whole && term of its condition; negated, under \
                         ||, in a closure or as an operand, a false check can leave its values \
                         for this thread's next panic",
                        line_of(&code, cs),
                        &code[cs..ce]
                    ));
                }
            }
        }
    }
    failures
}

#[test]
fn a_value_check_is_only_in_a_contract() {
    let failures: Vec<String> = workspace_sources()
        .iter()
        .flat_map(|(p, s)| misplaced_value_checks(p, s))
        .collect();
    assert!(
        failures.is_empty(),
        "{} value check(s) outside a contract:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn a_value_check_outside_a_contract_is_found() {
    let src = concat!(
        "#[requires(le!(count, data.len()), \"count is within data\")]\n",
        "#[debug_ensures(fb_core_log::lt!(ret, n) && eq ! (a, b))]\n",
        "fn f(count: usize, data: &[f64]) {\n",
        "    let ok = le!(count, data.len());\n",
        "    assert!(fits_i32![count.checked_mul(count)], \"fits\");\n",
        "    if a.le != b || x.eq(y) || lt!{a, b} {}\n",
        "    // le!(a, b)\n",
        "    let s = \"eq!(a, b)\";\n",
        "}\n",
        "macro_rules! le { ($a:expr) => {}; }\n",
    );
    let f = misplaced_value_checks("a.rs", src);
    assert_eq!(f.len(), 3, "{f:#?}");
    assert!(
        f[0].starts_with("a.rs:4: le!(count, data.len()) outside"),
        "{f:#?}"
    );
    assert!(f[1].starts_with("a.rs:5: fits_i32!"), "{f:#?}");
    assert!(f[2].starts_with("a.rs:6: lt!{a, b}"), "{f:#?}");
}

#[test]
fn a_value_check_is_a_whole_term_of_its_condition() {
    let sources = workspace_sources();
    // `core/log/tests/panic_hook.rs` trips one of each, so the scan sees some.
    let seen: usize = sources
        .iter()
        .map(|(_, s)| value_checks(&code_only(s)).len())
        .sum();
    assert!(seen >= 4, "only {seen} value checks found");
    let failures: Vec<String> = sources
        .iter()
        .flat_map(|(p, s)| buried_value_checks(p, s))
        .collect();
    assert!(
        failures.is_empty(),
        "{} value check(s) not a whole && term:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn a_negated_or_buried_value_check_is_found() {
    let src = concat!(
        "#[requires(le!(a, b), \"whole\")]\n",
        "#[requires(n > 0 && (lt!(i, n)) && (eq!(a, b) && fits_i32!(n.checked_mul(n))))]\n",
        "#[requires(n > 0, eq!(a, b), \"one assertion each\")]\n",
        "#[requires(le!(a, if x && y { 1 } else { 2 }))]\n",
        "#[requires(!le!(a, b), \"negated\")]\n",
        "#[requires(! (lt!(a, b)), \"negated in parentheses\")]\n",
        "#[debug_requires(le!(a, b) || c, \"or\")]\n",
        "#[ensures(xs.iter().all(|x| eq!(*x, ret)), \"in a closure\")]\n",
        "#[invariant(lt!(a, b) == c, \"an operand\")]\n",
        "fn f() {}\n",
    );
    let f = buried_value_checks("a.rs", src);
    let lines: Vec<&str> = f.iter().map(|m| m.split(':').nth(1).unwrap()).collect();
    assert_eq!(lines, ["5", "6", "7", "8", "9"], "{f:#?}");
    assert!(
        f[0].starts_with("a.rs:5: le!(a, b) is not a whole && term"),
        "{f:#?}"
    );
}

#[test]
fn a_leading_precondition_is_a_requires() {
    let found: Vec<Leading> = workspace_sources()
        .iter()
        .flat_map(|(p, s)| leading_preconditions(p, s))
        .collect();
    let failures = leading_failures(&found, KEPT);
    assert!(
        failures.is_empty(),
        "{} leading precondition(s) to write as #[requires]:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn a_leading_precondition_is_judged_by_its_place_and_its_names() {
    let src = concat!(
        "pub fn a(n: usize, x: &[f64]) {\n",
        "    assert!(x.len() >= n, \"x holds n\");\n",
        "    assert_eq!(n % 2, 0);\n",
        "    let m = n;\n",
        "    assert!(m > 0, \"not leading\");\n",
        "}\n",
        "pub fn b(count: usize, data: &[f64]) {\n",
        "    assert!(count <= data.len(), \"count is {count}, data {}\", data.len());\n",
        "    assert!(count < u32::MAX as usize && count < LIMIT && ok(count), \"fits, {{}}\");\n",
        "}\n",
        "fn private(n: usize) { assert!(n > 0, \"n\"); }\n",
        "pub(crate) fn scoped(n: usize) { assert!(n > 0, \"n\"); }\n",
        "pub fn c(&self, n: usize) { assert!(self.len > n && local > 0, \"names a local\"); }\n",
        "pub fn d(mut n: usize) { assert!(n > 0, \"n\") }\n",
        "pub fn e(n: usize, m: usize) {\n",
        "    if n > i32::MAX as usize {\n        panic!(\"n is {n}\");\n    }\n",
        "    assert_ne!(m, 0, \"m\");\n",
        "    if m > n { panic!(\"m\") } else { return; }\n",
        "    assert!(n > 0);\n",
        "}\n",
        "pub fn g(n: usize) {\n",
        "    if n > 0 { log(n); }\n",
        "    assert!(n > 0);\n",
        "}\n",
        "pub fn h(n: usize) {\n",
        "    if let Some(k) = n.checked_mul(n) { panic!(\"k\") }\n",
        "    if local > n { panic!(\"names a local\") }\n",
        "    assert!(n > 0);\n",
        "}\n",
        "pub const fn k(n: usize) { assert!(n > 0, \"const\"); }\n",
        "#[cfg(test)]\nmod tests {\n    pub fn t(n: usize) { assert!(n > 0, \"n\"); }\n}\n",
    );
    let found = leading_preconditions("a/src/lib.rs", src);
    let got: Vec<(String, usize)> = found.iter().map(|l| (l.key.clone(), l.line)).collect();
    let key = |f: &str| format!("a/src/lib.rs:{f}");
    let want = [
        (key("a"), 2),
        (key("a"), 3),
        (key("b"), 8),
        (key("b"), 9),
        (key("d"), 14),
        (key("e"), 16),
        (key("e"), 19),
    ];
    assert_eq!(got, want);
    assert!(leading_preconditions("a/tests/t.rs", src).is_empty());
    assert!(leading_preconditions("a/src/x.rs", &format!("#![cfg(test)]\n{src}")).is_empty());
}

#[test]
fn each_shape_of_leading_precondition_fails() {
    // One function per shape the check widened to (#1640): an interpolated
    // message, `assert_eq!`, and an `if … { panic!() }` guard.
    let shapes = [
        (
            "interpolated",
            "pub fn f(n: usize, x: &[f64]) { assert!(n <= x.len(), \"n is {n}\"); }\n",
        ),
        (
            "assert_eq",
            "pub fn f(n: usize, x: &[f64]) { assert_eq!(x.len(), n); }\n",
        ),
        (
            "panic guard",
            "pub fn f(n: usize) { if n > LIMIT { panic!(\"n is {}\", n); } }\n",
        ),
    ];
    for (shape, src) in shapes {
        let f = leading_failures(&leading_preconditions("a/src/lib.rs", src), &[]);
        assert_eq!(f.len(), 1, "{shape}: {f:#?}");
        assert_eq!(
            f[0], "a/src/lib.rs:1: f opens with a check on its arguments; write it as #[requires]",
            "{shape}"
        );
    }
}

#[test]
fn a_kept_entry_silences_its_site_and_fails_without_one() {
    let src = "pub fn f(n: usize) { assert_eq!(n, 1); }\n";
    let found = leading_preconditions("a/src/lib.rs", src);
    let kept = [("a/src/lib.rs:f", "a reason")];
    assert_eq!(leading_failures(&found, &kept), Vec::<String>::new());
    let stale = [
        ("a/src/lib.rs:f", "a reason"),
        ("a/src/lib.rs:gone", "a reason"),
    ];
    assert_eq!(
        leading_failures(&found, &stale),
        ["a/src/lib.rs:gone: kept, but no leading precondition is there; drop it"]
    );
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

    // A plain comment between the contracts does not end the run.
    let commented = "/// Needs one row.\n#[debug_requires(m > 0, \"one column\")]\n\
                     // RELEASE: the launcher refuses it.\n#[debug_requires(n > 0, \"one row\")]\n\
                     pub fn f(m: usize, n: usize) {}\n";
    assert_eq!(restated("a.rs", commented).len(), 1);

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
