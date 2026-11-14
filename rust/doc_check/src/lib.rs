//! The documentation checks that need nothing of the tree but its text:
//! `tests/section_citations.rs`, which checks that every section the
//! workspace's documents and comments cite by name is a heading of the file
//! cited; `tests/unsafe_code.rs`, which checks that every crate root
//! forbids `unsafe_code` unless `docs/programming_model.md` lists its crate
//! as allowed it; `tests/contracts.rs`, which checks that no doc comment
//! restates a contract rustdoc already renders, and that every
//! `debug_assert!` is an internal invariant that says so; and
//! `tests/rules.rs`, which checks that every lint, type, test file and test
//! the register `docs/rules.md` names exists. A crate of its own so that it
//! builds and runs without the wwr library. The checks are its tests; the
//! library holds only what more than one of them reads.

#![forbid(unsafe_code)]

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

#[cfg(test)]
mod tests {
    use super::code_only;

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
}
