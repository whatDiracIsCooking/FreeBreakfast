#!/usr/bin/env python3
"""Generate each rust/ kernel library's -sys bindings from its C header.

    devtools/rust-bindings.py           # rewrite every bindings file
    devtools/rust-bindings.py --check   # exit 1, with a diff, if any is stale

A kernel library's one C header is rust/<module>/kernels/fb_<module>.h, and
its bindings are rust/<module>/sys/src/bindings.rs, committed
(rust/docs/gpu_layer.md, "Build"). The tier-3 test in rust/device_check runs
this with --check, so a header changed without its bindings fails there.

WHY NOT BINDGEN. The design names bindgen, but the devcontainer ships neither
bindgen nor libclang's Python bindings, and adding either is a new toolchain
dependency for one header. So this does what WarpWraps' C ABI generator does
for its own (deps/WarpWraps/src/capi/generate.py, spec/fragments.py): it reads
the header as text with stdlib `re`, no clang. That is sound only because the
headers it reads are held to a small, plain-C grammar, and anything outside it
is an error rather than a guess:

- `#include "x.h"`, followed when x.h is beside the including file, under
  the library's kernels/ directory or in rust/device_headers; any other
  include (the C ABI's own common.h, the standard headers) is not read, and
  the types it supplies are EXTERNAL below.
- `#define NAME ((type)literal)`: a typed integer constant.
- `typedef <type> name;`: an integer alias.
- `typedef struct name {` ... `} name;`: a plain struct, one `<type> field;`
  or `<type> *field;` per line, which becomes a `#[repr(C)]` struct of the
  same fields in the same order, so the two layouts follow one rule.
- `<type> name(<type> arg, ...);`: a function, on one or more lines.
- `///` comments, carried to the next item as its doc comment.

Everything else -- `#pragma once`, include guards, the `extern "C"` braces,
`/* */` comments -- is skipped. stdlib only, so it runs wherever python3 does.
"""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUST = REPO / "rust"
DEVICE_HEADERS = RUST / "device_headers"

# The C types the grammar allows, as Rust spells them.
SCALARS = {
    "void": "core::ffi::c_void",
    "bool": "bool",
    "double": "f64",
    "float": "f32",
    "int8_t": "i8",
    "int16_t": "i16",
    "int32_t": "i32",
    "int64_t": "i64",
    "uint8_t": "u8",
    "uint16_t": "u16",
    "uint32_t": "u32",
    "uint64_t": "u64",
    "size_t": "usize",
}

# Types a header takes from a library the crate already depends on, and the
# Rust path that crate binds them at.
EXTERNAL = {"wwr_stream": "wwr_sys::wwr_stream"}

IDENT = r"[A-Za-z_]\w*"
INCLUDE = re.compile(r'#include\s+"([^"]+)"')
DEFINE = re.compile(rf"#define\s+({IDENT})\s+\(\(({IDENT})\)\s*(-?\s*\d+)\)\s*$")
TYPEDEF = re.compile(rf"typedef\s+({IDENT})\s+({IDENT})\s*;")
STRUCT_OPEN = re.compile(rf"typedef\s+struct\s+({IDENT})\s*\{{")
STRUCT_CLOSE = re.compile(rf"\}}\s*({IDENT})\s*;")
FIELD = re.compile(rf"(const\s+)?({IDENT})\s*(\*?)\s*({IDENT})\s*;")
FUNCTION = re.compile(rf"({IDENT}(?:\s*\*)?)\s*\b({IDENT})\s*\(([^)]*)\)\s*;", re.S)
PARAM = re.compile(rf"(const\s+)?({IDENT})\s*(\*?)\s*({IDENT})")


class HeaderError(Exception):
    pass


class Module:
    """One header's items, in the order they first appear."""

    def __init__(self, root: Path) -> None:
        # The library's kernels/ directory, an include's second search base.
        self.root = root
        self.aliases: dict[str, str] = {}
        self.structs: set[str] = set()
        self.items: list[str] = []
        self.functions: list[str] = []
        self.externals: set[str] = set()
        self.seen: set[Path] = set()

    def rust_type(self, name: str, pointer: bool, const: bool, where: str) -> str:
        if name in SCALARS:
            base = SCALARS[name]
        elif name in self.aliases or name in self.structs:
            base = name
        elif name in EXTERNAL:
            self.externals.add(name)
            base = name
        else:
            raise HeaderError(f"{where}: unknown type '{name}'")
        if pointer:
            return f"*{'const' if const else 'mut'} {base}"
        if name == "void":
            raise HeaderError(f"{where}: a bare void is only a return type")
        return base


def doc_lines(doc: list[str], indent: str) -> list[str]:
    return [f"{indent}///{line}" if line else f"{indent}///" for line in doc]


def read_header(path: Path, module: Module) -> None:
    path = path.resolve()
    if path in module.seen:
        return
    module.seen.add(path)
    where_file = path.relative_to(REPO)
    text = path.read_text()
    # /* */ comments go first, each kept as its newlines, so line numbers hold.
    text = re.sub(
        r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), text, flags=re.S
    )
    doc: list[str] = []
    pending = ""
    pending_line = 0
    # The struct being read: its name, doc and fields so far.
    struct: tuple[str, list[str], list[str]] | None = None
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        where = f"{where_file}:{number}"
        if struct and line and not line.startswith("//"):
            name, struct_doc, fields = struct
            m = STRUCT_CLOSE.fullmatch(line)
            if m:
                if m.group(1) != name:
                    raise HeaderError(f"{where}: struct {name} closed as {m.group(1)}")
                module.items += (
                    doc_lines(struct_doc, "")
                    + ["#[repr(C)]", "#[derive(Debug, Clone, Copy)]"]
                    + [f"pub struct {name} {{"]
                    + fields
                    + ["}"]
                )
                module.structs.add(name)
                struct = None
                continue
            f = FIELD.fullmatch(line)
            if not f:
                raise HeaderError(f"{where}: a struct field must be <type> [*]name;")
            const, ctype, star, fname = f.groups()
            rtype = module.rust_type(ctype, bool(star), bool(const), where)
            fields += doc_lines(doc, "    ") + [f"    pub {fname}: {rtype},"]
            doc = []
            continue
        if pending:
            pending += " " + line
            if ";" in line:
                add_function(pending, module, doc, f"{where_file}:{pending_line}")
                doc, pending = [], ""
            continue
        if line.startswith("///"):
            doc.append(line[3:])
            continue
        if not line or line.startswith("//"):
            continue
        if line.startswith("#include"):
            m = INCLUDE.match(line)
            if m:
                for base in (path.parent, module.root, DEVICE_HEADERS):
                    if (base / m.group(1)).is_file():
                        read_header(base / m.group(1), module)
                        break
            continue
        if line.startswith("#define"):
            m = DEFINE.match(line)
            if not m:
                raise HeaderError(f"{where}: a #define must be NAME ((type)integer)")
            name, ctype, value = m.groups()
            value = value.replace(" ", "")  # clang-format writes (t) - 2
            rtype = module.rust_type(ctype, False, False, where)
            module.items += doc_lines(doc, "") + [
                f"pub const {name}: {rtype} = {value};"
            ]
            doc = []
            continue
        if line.startswith("#") or line in ('extern "C" {', "}"):
            continue
        if line.startswith("typedef struct"):
            m = STRUCT_OPEN.fullmatch(line)
            if not m:
                raise HeaderError(f"{where}: a struct opens as typedef struct name {{")
            struct, doc = (m.group(1), doc, []), []
            continue
        if line.startswith("typedef"):
            m = TYPEDEF.fullmatch(line)
            if not m or m.group(1) not in SCALARS:
                raise HeaderError(f"{where}: a typedef must alias a scalar type")
            target, name = m.groups()
            module.aliases[name] = SCALARS[target]
            module.items += doc_lines(doc, "") + [
                f"pub type {name} = {SCALARS[target]};"
            ]
            doc = []
            continue
        pending, pending_line = line, number
        if ";" in line:
            add_function(pending, module, doc, where)
            doc, pending = [], ""
    if pending:
        raise HeaderError(f"{where_file}:{pending_line}: unterminated declaration")
    if struct:
        raise HeaderError(f"{where_file}: struct {struct[0]} is never closed")


def add_function(decl: str, module: Module, doc: list[str], where: str) -> None:
    m = FUNCTION.fullmatch(decl.strip())
    if not m:
        raise HeaderError(f"{where}: not a declaration this grammar reads: {decl!r}")
    ret, name, params = m.groups()
    ret_pointer = ret.endswith("*")
    ret_type = module.rust_type(ret.rstrip("* "), ret_pointer, False, where)
    args = []
    if params.strip() not in ("", "void"):
        for param in params.split(","):
            p = PARAM.fullmatch(param.strip())
            if not p:
                raise HeaderError(f"{where}: unreadable parameter {param.strip()!r}")
            const, ctype, star, pname = p.groups()
            args.append(
                f"{pname}: {module.rust_type(ctype, bool(star), bool(const), where)}"
            )
    returns = "" if ret == "void" else f" -> {ret_type}"
    module.functions += doc_lines(doc, "    ") + [
        f"    pub fn {name}({', '.join(args)}){returns};"
    ]


def render(header: Path) -> str:
    module = Module(header.parent)
    read_header(header, module)
    rel = header.relative_to(REPO)
    lines = [
        f"// GENERATED by devtools/rust-bindings.py from {rel} -- do not edit.",
        "// Regenerate with `devtools/rust-bindings.py`; rust/device_check's",
        "// tier-3 test fails while this file and the header disagree.",
        "",
    ]
    lines += [f"pub use {EXTERNAL[name]};" for name in sorted(module.externals)]
    if module.externals:
        lines.append("")
    lines += module.items
    if module.functions:
        lines += ["", 'unsafe extern "C" {'] + module.functions + ["}"]
    return "\n".join(lines) + "\n"


def modules() -> list[tuple[Path, Path]]:
    """Each kernel library's header and the bindings generated from it."""
    found = []
    for kernels in sorted(RUST.glob("*/kernels")):
        name = kernels.parent.name
        header = kernels / f"fb_{name}.h"
        if header.is_file():
            found.append((header, kernels.parent / "sys" / "src" / "bindings.rs"))
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--check", action="store_true", help="diff instead of writing")
    check = parser.parse_args().check
    pairs = modules()
    if not pairs:
        print("rust-bindings: no rust/*/kernels/fb_*.h found", file=sys.stderr)
        return 2
    stale = 0
    for header, out in pairs:
        try:
            text = render(header)
        except HeaderError as e:
            print(f"rust-bindings: {e}", file=sys.stderr)
            return 2
        old = out.read_text() if out.is_file() else ""
        rel = out.relative_to(REPO)
        if old == text:
            print(f"rust-bindings: {rel} is current")
        elif check:
            stale += 1
            sys.stdout.writelines(
                difflib.unified_diff(
                    old.splitlines(keepends=True),
                    text.splitlines(keepends=True),
                    f"{rel} (committed)",
                    f"{rel} (from {header.relative_to(REPO)})",
                )
            )
        else:
            out.write_text(text)
            print(f"rust-bindings: wrote {rel}")
    if stale:
        print(
            f"rust-bindings: {stale} bindings file(s) stale; "
            "run devtools/rust-bindings.py"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
