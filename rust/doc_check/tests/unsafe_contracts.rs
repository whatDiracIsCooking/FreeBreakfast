//! No debug-only contract on an extent an `unsafe` block trusts
//! (`docs/testing.md`, "Contracts are part of the spec").
//!
//! In a function whose body holds an `unsafe` block, a `#[debug_requires]`
//! whose condition names an integer-typed parameter, or a parameter's
//! `.len()`, is compiled out of a release build, so the block must not rest
//! on it. Each such contract is either always-on (`#[requires]`, which this
//! check passes) or carries a `// RELEASE:` comment directly above it saying
//! what holds the bound in a release build: the body clamps it, the
//! launcher refuses it. The marker is the shape of `// SAFETY:` and
//! `// INVARIANT:`.
//!
//! Every `.rs` of every workspace member is read. Only the crates
//! `docs/programming_model.md` ("Capabilities are enforced by lints") lists
//! can hold an `unsafe` block, since every other crate root forbids
//! `unsafe_code`, so the scan reaches no other crate's functions in effect.
//!
//! It is a text scan, and sees only what it looks for: an extent reached
//! through a method (`class.la()`) or a field, a parameter of a newtype or
//! a reference to an integer, and an `unsafe` block in a helper the function
//! calls (`square_block`, `matrix_extent`) are not seen.

#![forbid(unsafe_code)]

use fb_doc_check::{attributes, code_only, is_ident, line_of, words, workspace_sources};

/// The comment that says what holds a debug contract's bound in release.
const RELEASE: &str = "// RELEASE:";
/// The parameter types whose names, in a debug contract, are an extent.
const INTEGERS: [&str; 12] = [
    "usize", "isize", "u8", "u16", "u32", "u64", "u128", "i8", "i16", "i32", "i64", "i128",
];

/// The index one past the bracket that closes the one at `open` in `b`,
/// counting `open_c` and `close_c` alone, or `b.len()`.
fn closing(b: &[u8], open: usize, open_c: u8, close_c: u8) -> usize {
    let mut depth = 0i32;
    for (j, &c) in b.iter().enumerate().skip(open) {
        if c == open_c {
            depth += 1;
        } else if c == close_c {
            depth -= 1;
            if depth == 0 {
                return j + 1;
            }
        }
    }
    b.len()
}

/// The parameters of a parameter list's inside, as each one's name and
/// type, `self` left out.
fn parameters(list: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let mut depth = 0i32;
    let mut from = 0;
    let b = list.as_bytes();
    let mut push = |p: &str| {
        if let Some((pat, ty)) = p.split_once(':') {
            let name = pat
                .split(|c: char| !is_ident(c as u8))
                .rfind(|w| !w.is_empty());
            if let Some(name) = name {
                out.push((name.to_owned(), ty.trim().to_owned()));
            }
        }
    };
    for (j, &c) in b.iter().enumerate() {
        match c {
            b'(' | b'[' | b'{' | b'<' => depth += 1,
            b'>' if j > 0 && b[j - 1] == b'-' => {}
            b')' | b']' | b'}' | b'>' => depth -= 1,
            b',' if depth == 0 => {
                push(&list[from..j]);
                from = j + 1;
            }
            _ => {}
        }
    }
    push(&list[from..]);
    out
}

/// The debug contracts of one source that name an extent of their
/// function, in a function whose body holds an `unsafe` block, with no
/// `// RELEASE:` comment directly above them.
fn unmarked(path: &str, source: &str) -> Vec<String> {
    let lines: Vec<&str> = source.lines().collect();
    let code = code_only(source);
    let b = code.as_bytes();
    let attrs = attributes(&code);
    let ws = words(&code);
    let mut failures = Vec::new();
    for (k, &(at, w)) in ws.iter().enumerate() {
        if w != "fn" {
            continue;
        }
        let Some(&(name_at, name)) = ws.get(k + 1) else {
            continue;
        };
        // The parameter list, past any generics.
        let mut j = name_at + name.len();
        while j < b.len() && b[j].is_ascii_whitespace() {
            j += 1;
        }
        if b.get(j) == Some(&b'<') {
            let mut depth = 0i32;
            while j < b.len() {
                match b[j] {
                    b'<' => depth += 1,
                    b'>' if b[j - 1] != b'-' => depth -= 1,
                    _ => {}
                }
                j += 1;
                if depth == 0 {
                    break;
                }
            }
        }
        while j < b.len() && b[j].is_ascii_whitespace() {
            j += 1;
        }
        if b.get(j) != Some(&b'(') {
            continue;
        }
        let close = closing(b, j, b'(', b')');
        let params = parameters(&code[j + 1..close.saturating_sub(1)]);
        // A declaration or a `fn` type has no body before its `;`.
        let Some(open) = (close..b.len()).find(|&i| b[i] == b'{' || b[i] == b';') else {
            continue;
        };
        if b[open] == b';' {
            continue;
        }
        let body = &code[open..closing(b, open, b'{', b'}')];
        let bw = words(body);
        let holds_unsafe = bw
            .iter()
            .any(|&(i, w)| w == "unsafe" && body[i + w.len()..].trim_start().starts_with('{'));
        if !holds_unsafe {
            continue;
        }
        let integers: Vec<&str> = params
            .iter()
            .filter(|(_, ty)| INTEGERS.contains(&ty.as_str()))
            .map(|(n, _)| n.as_str())
            .collect();
        // The attributes above the item, up over each one that ends where
        // the next begins; between them is only blanked comment.
        let mut top = code[..at].rfind('\n').map_or(0, |p| p + 1);
        let mut above = Vec::new();
        while let Some(a) = attrs
            .iter()
            .rev()
            .find(|a| a.end <= top && code[a.end..top].trim().is_empty())
        {
            above.push(a);
            top = a.start;
        }
        for a in above.into_iter().filter(|a| a.name == "debug_requires") {
            let cond = &code[a.start..a.end];
            let names = words(cond).into_iter().find(|&(i, w)| {
                let field = cond[..i].ends_with('.');
                let after = cond[i + w.len()..].trim_start();
                !field
                    && (integers.contains(&w)
                        || (params.iter().any(|(n, _)| n == w) && after.starts_with(".len()")))
            });
            let Some((_, extent)) = names else {
                continue;
            };
            let first = line_of(&code, a.start) - 1;
            let marked = lines[..first]
                .iter()
                .rev()
                .map(|l| l.trim_start())
                .take_while(|l| l.starts_with("//") && !l.starts_with("///"))
                .any(|l| l.starts_with(RELEASE));
            if !marked {
                failures.push(format!(
                    "{path}:{}: #[debug_requires] on `{extent}` in {name}, whose body holds an \
                     unsafe block, with no {RELEASE} comment above it; make it #[requires] or \
                     say what holds the bound in a release build",
                    first + 1
                ));
            }
        }
    }
    failures
}

#[test]
fn no_debug_contract_bounds_an_extent_an_unsafe_block_trusts() {
    let sources = workspace_sources();
    assert!(sources.len() > 50, "only {} sources found", sources.len());
    let failures: Vec<String> = sources.iter().flat_map(|(p, s)| unmarked(p, s)).collect();
    assert!(
        failures.is_empty(),
        "{} debug contract(s) an unsafe block may rest on:\n{}",
        failures.len(),
        failures.join("\n")
    );
}

/// A function of `params` whose body is `body`, under `attrs`.
fn fixture(attrs: &str, params: &str, body: &str) -> String {
    format!(
        "/// Doc.\n{attrs}\n#[track_caller]\npub fn f<T: Fn() -> u8>({params}) -> u8 {{\n{body}\n}}\n"
    )
}

#[test]
fn an_extent_contract_needs_the_marker_or_to_be_always_on() {
    let unsafe_body = "    // SAFETY: x.\n    unsafe { g(n) }";
    let flagged = fixture(
        "#[debug_requires(n >= 1, \"one\")]",
        "s: &Stream, mut n: usize, a: &[f64]",
        unsafe_body,
    );
    let f = unmarked("a.rs", &flagged);
    assert_eq!(f.len(), 1, "{f:#?}");
    assert!(
        f[0].starts_with("a.rs:2:") && f[0].contains("`n`"),
        "{f:#?}"
    );

    let len = fixture(
        "#[debug_requires(\n    a.len() >= 3,\n    \"three\"\n)]",
        "a: &impl DeviceSlice<f64>",
        unsafe_body,
    );
    assert_eq!(unmarked("a.rs", &len).len(), 1);

    let marked = fixture(
        "#[debug_requires(n >= 1, \"one\")]\n// RELEASE: the launcher refuses it.\n\
         // More.\n#[debug_requires(n < 9, \"nine\")]",
        "n: u32",
        unsafe_body,
    );
    let f = unmarked("a.rs", &marked);
    assert_eq!(f.len(), 1, "only the first is unmarked: {f:#?}");
    assert!(f[0].starts_with("a.rs:2:"), "{f:#?}");

    let release_first = fixture(
        "// RELEASE: the body clamps it.\n#[debug_requires(n >= 1, \"one\")]",
        "n: i32",
        unsafe_body,
    );
    assert_eq!(unmarked("a.rs", &release_first), Vec::<String>::new());
}

#[test]
fn what_is_not_an_extent_contract_on_an_unsafe_body_passes() {
    let unsafe_body = "    // SAFETY: x.\n    unsafe { g() }";
    let cases = [
        // Always-on.
        fixture("#[requires(n >= 1, \"one\")]", "n: usize", unsafe_body),
        // No unsafe block: a safe wrapper's caller.
        fixture(
            "#[debug_requires(n >= 1, \"one\")]",
            "n: usize",
            "    let s = \"unsafe { }\";\n    g(n)",
        ),
        // A non-integer parameter.
        fixture(
            "#[debug_requires(floor >= 0.0, \"f\")]",
            "floor: f64",
            unsafe_body,
        ),
        // The name only inside the message, or as a field or a method's.
        fixture(
            "#[debug_requires(m > 0, \"n is positive\")]\n\
             #[debug_requires(class.n <= class.la(), \"in the table\")]",
            "n: usize, m: f64, class: &ClassRows",
            unsafe_body,
        ),
    ];
    for c in &cases {
        assert_eq!(unmarked("a.rs", c), Vec::<String>::new(), "{c}");
    }
}
