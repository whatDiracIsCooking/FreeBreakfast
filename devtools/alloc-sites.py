#!/usr/bin/env python3
"""The tree's allocation sites: an inventory, and the pool-bypass lint.

docs/style.md "Allocations go through the pool" is the rule. Three commands:

  inventory [--json] <paths>
      One row per allocation site: file:line, kind, element type, the size
      expression as written, and the enclosing function. The footprint issues
      use it as their checklist, over a step's directory and the libraries it
      calls. The kinds:

        Buffers::device / ::pinned / ::host, StreamBuffers::device / ::pinned
            a named object (`x(n, dev)`, `x{n, dev}`), a temporary
            (`Buffers::device<T>(n, dev)`), a `std::make_shared` /
            `make_unique` of one, or an `emplace` / `emplace_back` into a
            container (vector, optional, ...) of them that this file or one
            in its directory declares by name;
        device_workspace / pinned_workspace
            a borrow (the type is std::byte, the size its byte count);
        device_regions / pinned_regions
            a carve (the type is its template arguments, the size its counts);
        counted_pinned
            a counted persistent pinned allocation (the type is its template
            argument, the size its count).

      Only what is spelled at the site is seen: a helper that returns a buffer
      (`device_zeros(res, n)`) is one row in the helper, not one per caller.

  check <paths>                  (the pre-commit hook `pool-bypass`)
      Flags allocation the pool cap cannot see:
        - outside src/core/wwr/: `wwrMalloc*`, `wwrHostAlloc`, a raw
          `cudaMalloc*` / `hipMalloc*` / `cudaHostAlloc` / `hipHostMalloc`;
        - outside src/core/: `Buffers::pinned` / `StreamBuffers::pinned`,
          which the pinned cap cannot see;
        - outside src/core/wwr/: `wwrMemGetInfo`, beyond the per-file counts in
          devtools/alloc-sites-memgetinfo.txt. Each entry names the issue that
          migrates it; an entry whose file has fewer calls than it allows, or
          no longer exists, is stale and flagged too, so a migration deletes it.
      A line carrying `pool-bypass-ok:` and a reason, or the line after a
      comment-only line that does, is exempt (as `dynamic-smem-ok:` is).

  changed <base> [paths]
      The sites added or resized between <base> and HEAD (default paths:
      src). Advisory, not a hook: for a reviewer of a step whose footprint the
      diff did not touch.

Comments and string literals are blanked before matching, and FelizJueves
kernels are skipped, with check-int-types.py's helpers (reused, not copied).
`check` exits 1 with one `path:line:col: message` per finding, 0 when clean.
Stdlib-only.
"""

from __future__ import annotations

import argparse
import bisect
import importlib.util
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

_INT_TYPES = Path(__file__).resolve().parent / "check-int-types.py"
_spec = importlib.util.spec_from_file_location("check_int_types", _INT_TYPES)
assert _spec is not None and _spec.loader is not None
_int_types = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_int_types)

blank_comments_and_literals = _int_types.blank_comments_and_literals
is_generated = _int_types.is_generated

REPO = Path(__file__).resolve().parents[1]
MEMGETINFO_ALLOWLIST = Path(__file__).resolve().parent / "alloc-sites-memgetinfo.txt"
CPP_SUFFIXES = frozenset({".cppm", ".cpp", ".cu", ".cuh", ".h", ".hpp"})

# The marker must carry a reason: a bare `pool-bypass-ok:` exempts nothing.
_SUPPRESS = re.compile(r"pool-bypass-ok:\s*\S")

_BUFFER = re.compile(r"(?<!\w)(StreamBuffers|Buffers)::(device|pinned|host)\s*<")
_MAKE = re.compile(r"(?<!\w)make_(?:shared|unique)\s*<\s*(?:const\s+)?")
_WORKSPACE = re.compile(r"(?<![\w])(device|pinned)_workspace\s*\(")
_COUNTED = re.compile(r"(?<![\w])counted_pinned\s*<")
_REGIONS = re.compile(r"(?<![\w])(device|pinned)_regions\s*<")
_CONTAINER = re.compile(r"(?<!\w)(?:std::)?(?:vector|optional|deque|list|array)\s*<")
_EMPLACE = re.compile(r"(?<![\w])([A-Za-z_]\w*)\s*(?:\.|->)\s*emplace(?:_back)?\s*\(")
_IDENT = re.compile(r"[A-Za-z_]\w*")
_MEMBER = re.compile(r"\s*([A-Za-z_]\w*)\s*;")
_DECLARATORS = re.compile(r"\s*([A-Za-z_]\w*(?:\s*,\s*[A-Za-z_]\w*)*)\s*[;={]")

_RAW_ALLOC = re.compile(
    r"(?<!\w)(?:wwrMalloc\w*|wwrHostAlloc|(?:cuda|hip)Malloc\w*"
    r"|cudaHostAlloc|hipHostMalloc)\b"
)
_PINNED = re.compile(r"(?<!\w)(?:Stream)?Buffers::pinned\b")
_MEMGETINFO = re.compile(r"(?<!\w)wwrMemGetInfo\b")

# A function parameter, as clang-format spells one: a type, then the name with
# any `&` / `*` hugging it (`DeviceResources &res`). A binary `n * m` has a
# space on both sides of the operator, so it is not one.
_PARAMETER = re.compile(
    r"^(?:const\s+)?[A-Za-z_][\w:]*(?:<.*>)?(?:\s+const)?(?:\s*\.\.\.)?"
    r"\s+(?:&&?|\*+)?[A-Za-z_]\w*(?:\s*=.*)?$",
    re.S,
)
# Keywords a `(...) {` follows that do not open a function body.
_NOT_FUNCTIONS = frozenset(
    "if for while switch catch return sizeof alignof decltype constexpr noexcept "
    "requires".split()
)
_FUNCTION_NAME = re.compile(
    r"((?:[\w:~]*::)?operator\s*(?:\(\)|\[\]|[^\w\s(]+)|~?[A-Za-z_][\w:~]*)$"
)
_TRAILING_QUALIFIERS = re.compile(
    r"\s*(?:\b(?:const|noexcept|override|final|mutable)\b|&&?)\s*$"
)


@dataclass(frozen=True)
class Site:
    file: str
    line: int
    kind: str
    type: str
    size: str
    function: str


# -- small parsers over the blanked text ------------------------------------


def _balanced(text: str, i: int) -> int:
    """Index just past the bracket group opening at text[i], or len(text)."""
    pairs = {"(": ")", "{": "}", "<": ">", "[": "]"}
    opener = text[i]
    closer = pairs[opener]
    depth = 0
    for j in range(i, len(text)):
        c = text[j]
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return j + 1
        elif opener == "<" and c in ";{}":
            break  # a less-than, not a template argument list
    return len(text)


def _split_top_level(args: str, keep_empty: bool = False) -> list[str]:
    """Split a call's argument text on its top-level commas.

    An argument blanked to spaces (a string literal) is dropped unless
    `keep_empty`, which keeps the arguments' positions.
    """
    out, depth, start = [], 0, 0
    for j, c in enumerate(args):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            out.append(args[start:j])
            start = j + 1
    out.append(args[start:])
    return [a.strip() for a in out if keep_empty or a.strip()]


def _squash(text: str) -> str:
    return " ".join(text.split())


def _skip_ws(text: str, i: int) -> int:
    while i < len(text) and text[i].isspace():
        i += 1
    return i


def _looks_like_parameters(args: str) -> bool:
    return any(_PARAMETER.match(a) for a in _split_top_level(args))


def _line_of(starts: list[int], offset: int) -> int:
    return bisect.bisect_right(starts, offset)


def _line_starts(text: str) -> list[int]:
    starts = [0]
    starts.extend(m.end() for m in re.finditer("\n", text))
    return starts


# -- the enclosing function --------------------------------------------------


def _open_tail(segment: str) -> str:
    """What follows the last `{` that `segment` leaves open."""
    cut = 0
    opens: list[int] = []
    for j, c in enumerate(segment):
        if c == "{":
            opens.append(j)
        elif c == "}" and opens:
            opens.pop()
    if opens:
        cut = opens[-1] + 1
    return segment[cut:]


def _function_label(segment: str) -> str | None:
    """The function a `{` closing `segment` opens the body of, or None."""
    seg = _open_tail(segment).rstrip()
    arrow = seg.rfind("->")
    if arrow != -1 and ")" in seg[:arrow] and "(" not in seg[arrow:]:
        seg = seg[:arrow].rstrip()  # a trailing return type
    # Searched over the tail only: a `$`-anchored search is quadratic.
    while m := _TRAILING_QUALIFIERS.search(seg[-64:]):
        seg = seg[: len(seg) - len(seg[-64:]) + m.start()]
    if not seg.endswith((")", "}")):
        return None  # a braced initializer (`n_{n}`), a class, a namespace
    # A constructor's initializer list: cut at the `:` after its parameters.
    depth = 0
    for j, c in enumerate(seg):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == ":" and depth == 0 and 0 < j < len(seg) - 1:
            if ":" not in (seg[j - 1], seg[j + 1]) and seg[:j].rstrip().endswith(")"):
                seg = seg[:j].rstrip()
                break
    if not seg.endswith(")"):
        return None
    depth = 0
    for j in range(len(seg) - 1, -1, -1):
        if seg[j] == ")":
            depth += 1
        elif seg[j] == "(":
            depth -= 1
            if depth == 0:
                m = _FUNCTION_NAME.search(seg[:j].rstrip()[-256:])
                if not m or m.group(1) in _NOT_FUNCTIONS:
                    return None
                return m.group(1)
    return None


def _function_spans(code: str) -> list[tuple[int, int, str]]:
    """(start, end, name) for every function body in `code`.

    A segment runs from the last top-level `;`, so a braced initializer
    (`n_{n}` in a constructor's initializer list, a `= {}` default argument)
    stays inside the segment of the body that follows it.
    """
    spans, stack = [], []
    seg_start = parens = 0
    for i, c in enumerate(code):
        if c == "(":
            parens += 1
        elif c == ")":
            parens = max(parens - 1, 0)
        elif c == "{":
            # A body's span opens at its segment, so it holds the signature
            # and a constructor's initializer list.
            label = _function_label(code[seg_start:i])
            stack.append((seg_start if label else i, label))
        elif c == "}":
            if stack:
                start, label = stack.pop()
                if label is not None:
                    spans.append((start, i, label))
                    seg_start = i + 1
        elif c == ";" and parens == 0:
            seg_start = i + 1
    return spans


def _enclosing(spans: list[tuple[int, int, str]], offset: int) -> str:
    best = None
    for start, end, label in spans:
        if start < offset < end and (best is None or start > best[0]):
            best = (start, label)
    return best[1] if best else ""


# -- the inventory -----------------------------------------------------------


def _buffer_type(code: str, m: re.Match) -> tuple[str, str, int]:
    """(kind, element type, end) for a `Buffers::x<T>` match."""
    lt = m.end() - 1
    end = _balanced(code, lt)
    return f"{m.group(1)}::{m.group(2)}", _squash(code[lt + 1 : end - 1]), end


# A declared name: (kind, type, member), member when it is a buffer declared
# with no initializer (constructed later, in an initializer list), else a
# container of buffers; None for a container of something else, which shadows
# a sibling file's name.
Names = dict[str, "tuple[str, str, bool] | None"]


def declared_names(code: str) -> Names:
    """The containers and uninitialized buffers that `code` declares."""
    names: Names = {}
    for m in _CONTAINER.finditer(code):
        end = _balanced(code, m.end() - 1)
        b = _BUFFER.search(code, m.end(), end)
        held = (*_buffer_type(code, b)[:2], False) if b else None
        if tail := _DECLARATORS.match(code, end):
            for name in re.split(r"\s*,\s*", tail.group(1)):
                names[name] = held
    for m in _BUFFER.finditer(code):
        kind, elem, end = _buffer_type(code, m)
        if tail := _MEMBER.match(code, end):
            names[tail.group(1)] = (kind, elem, True)
    return names


def find_sites(text: str, path: str = "", siblings: Names | None = None) -> list[Site]:
    """Every allocation site in `text`, in source order.

    `siblings` are the names the other files in its directory declare (a
    member declared in a .cppm, constructed in its .cpp); `text`'s own
    declarations take precedence.
    """
    code = blank_comments_and_literals(text)
    starts = _line_starts(code)
    spans = _function_spans(code)
    names = {**(siblings or {}), **declared_names(code)}
    found: list[tuple[int, str, str, str]] = []

    def row(offset: int, kind: str, elem: str, size: str) -> None:
        found.append((offset, kind, elem, size))

    def sizes(space: str, args: str) -> str:
        # The pinned spellings take the caller's name first: not a size.
        parts = _split_top_level(args, keep_empty=True)
        return ", ".join(map(_squash, parts[1:] if space == "pinned" else parts))

    def first_arg(args: str) -> str:
        parts = _split_top_level(args)
        return parts[0] if parts else ""

    make_ends = set()
    for m in _MAKE.finditer(code):
        b = _BUFFER.match(code, m.end())
        if not b:
            continue
        kind, elem, end = _buffer_type(code, b)
        close = _skip_ws(code, end)
        if code.startswith(">", close):
            paren = _skip_ws(code, close + 1)
            if code.startswith("(", paren):
                args = code[paren + 1 : _balanced(code, paren) - 1]
                row(m.start(), kind, elem, _squash(first_arg(args)))
                make_ends.add(b.start())

    for m in _BUFFER.finditer(code):
        if m.start() in make_ends:
            continue
        kind, elem, end = _buffer_type(code, m)
        j = _skip_ws(code, end)
        named = _IDENT.match(code, j)
        if named:
            j = _skip_ws(code, named.end())
        if j >= len(code) or code[j] not in "({":
            continue
        args = code[j + 1 : _balanced(code, j) - 1]
        if not args.strip():
            continue  # empty, or a function declaration
        if named:
            column = m.start() - starts[_line_of(starts, m.start()) - 1]
            line_head = code[starts[_line_of(starts, m.start()) - 1] : m.start()]
            at_namespace = column == 0 or line_head.strip() in {"export", "static"}
            if at_namespace or (code[j] == "(" and _looks_like_parameters(args)):
                continue  # a function returning a buffer
        row(m.start(), kind, elem, _squash(first_arg(args)))

    for m in _EMPLACE.finditer(code):
        if (held := names.get(m.group(1))) and not held[2]:
            args = code[m.end() : _balanced(code, m.end() - 1) - 1]
            row(m.start(1), held[0], held[1], _squash(first_arg(args)))

    # A member constructed in an initializer list: `: theta_(n, dev)`.
    members = [n for n, held in names.items() if held and held[2]]
    if members:
        alternation = "|".join(map(re.escape, sorted(members, key=len, reverse=True)))
        for m in re.finditer(rf"(?<![\w.>:])({alternation})\s*[({{]", code):
            k = m.start() - 1
            while k >= 0 and code[k].isspace():
                k -= 1
            if k < 0 or code[k] not in ":," or code[k - 1 : k + 1] == "::":
                continue
            j = m.end() - 1
            args = code[j + 1 : _balanced(code, j) - 1]
            if args.strip() and not args.strip().startswith("std::move("):
                kind, elem, _ = names[m.group(1)]
                row(m.start(1), kind, elem, _squash(first_arg(args)))

    for m in _WORKSPACE.finditer(code):
        args = code[m.end() : _balanced(code, m.end() - 1) - 1]
        if _looks_like_parameters(args):
            continue  # its definition
        row(m.start(), f"{m.group(1)}_workspace", "std::byte", sizes(m.group(1), args))

    for m in _REGIONS.finditer(code):
        end = _balanced(code, m.end() - 1)
        j = _skip_ws(code, end)
        if not code.startswith("(", j):
            continue
        args = code[j + 1 : _balanced(code, j) - 1]
        if _looks_like_parameters(args):
            continue
        elem = _squash(code[m.end() : end - 1])
        row(m.start(), f"{m.group(1)}_regions", elem, sizes(m.group(1), args))

    for m in _COUNTED.finditer(code):
        end = _balanced(code, m.end() - 1)
        j = _skip_ws(code, end)
        if not code.startswith("(", j):
            continue
        args = code[j + 1 : _balanced(code, j) - 1]
        if _looks_like_parameters(args):
            continue
        parts = _split_top_level(args, keep_empty=True)
        size = _squash(parts[-1]) if parts else ""
        row(m.start(), "counted_pinned", _squash(code[m.end() : end - 1]), size)

    found.sort()
    return [
        Site(path, _line_of(starts, o), kind, elem, size, _enclosing(spans, o))
        for o, kind, elem, size in found
    ]


# -- reading a tree ----------------------------------------------------------

Reader = Callable[[str], "str | None"]
Lister = Callable[[str], list[str]]


def _worktree_reader(path: str) -> str | None:
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.is_file() else None


def _worktree_lister(directory: str) -> list[str]:
    d = Path(directory or ".")
    if not d.is_dir():
        return []
    return [str(p) for p in sorted(d.iterdir()) if p.suffix in CPP_SUFFIXES]


def expand(paths: Iterable[str]) -> list[str]:
    """The C++ files under `paths`, each file once, in order."""
    out: list[str] = []
    for name in paths:
        p = Path(name)
        if p.is_dir():
            out.extend(str(f) for f in sorted(p.rglob("*")) if f.suffix in CPP_SUFFIXES)
        else:
            out.append(name)
    return list(dict.fromkeys(out))


def inventory(
    files: Iterable[str],
    reader: Reader = _worktree_reader,
    lister: Lister = _worktree_lister,
) -> list[Site]:
    """The sites in `files`, each read with the names its directory declares."""
    by_dir: dict[str, Names] = {}
    sites: list[Site] = []
    for name in files:
        text = reader(name)
        if text is None or is_generated(text):
            continue
        directory = str(Path(name).parent)
        if directory not in by_dir:
            names: Names = {}
            for sibling in lister(directory):
                if (sibling_text := reader(sibling)) is not None:
                    found = declared_names(blank_comments_and_literals(sibling_text))
                    names.update({k: v for k, v in found.items() if v})
            by_dir[directory] = names
        sites.extend(find_sites(text, name, by_dir[directory]))
    return sites


# -- the check ---------------------------------------------------------------


def repo_path(name: str) -> str:
    """`name` relative to the repository root, as posix; as given if outside."""
    p = Path(name).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return Path(name).as_posix()


def is_suppressed(raw_lines: list[str], index: int) -> bool:
    """True when line `index` carries the marker, or follows a comment-only
    line that does (so the reason need not share an already-long line)."""
    if _SUPPRESS.search(raw_lines[index]):
        return True
    if index == 0:
        return False
    above = raw_lines[index - 1].strip()
    return above.startswith("//") and _SUPPRESS.search(above) is not None


def read_allowlist(path: Path = MEMGETINFO_ALLOWLIST) -> dict[str, tuple[int, str]]:
    """path -> (calls allowed, owning issue). Each entry must name its owner."""
    entries: dict[str, tuple[int, str]] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = re.fullmatch(r"(\S+)\s+(\d+)\s+(#\d+)", line)
        if not m:
            raise ValueError(f"{path}:{lineno}: want `<path> <calls> #<issue>`")
        entries[m.group(1)] = (int(m.group(2)), m.group(3))
    return entries


def find_bypasses(
    text: str, rel: str, allowlist: dict[str, tuple[int, str]]
) -> list[tuple[int, int, str]]:
    """(line, column, message) for each bypass in `text`, at repo path `rel`."""
    if rel.startswith("src/core/wwr/"):
        return []
    in_core = rel.startswith("src/core/")
    raw_lines = text.split("\n")
    code_lines = blank_comments_and_literals(text).split("\n")
    found: list[tuple[int, int, str]] = []
    memgetinfo: list[tuple[int, int]] = []
    for lineno, code in enumerate(code_lines, 1):
        if is_suppressed(raw_lines, lineno - 1):
            continue
        for m in _RAW_ALLOC.finditer(code):
            found.append(
                (lineno, m.start() + 1, f"`{m.group(0)}` bypasses the device pool")
            )
        for m in _MEMGETINFO.finditer(code):
            memgetinfo.append((lineno, m.start() + 1))
        if in_core:
            continue
        for m in _PINNED.finditer(code):
            found.append(
                (lineno, m.start() + 1, "pinned memory outside src/core/ (#1130)")
            )
    allowed, owner = allowlist.get(rel, (0, ""))
    if len(memgetinfo) > allowed:
        what = (
            f"{len(memgetinfo)} calls here, the allowlist has {allowed}"
            if allowed
            else "not in devtools/alloc-sites-memgetinfo.txt"
        )
        for lineno, col in memgetinfo:
            found.append((lineno, col, f"`wwrMemGetInfo`, {what}"))
    elif len(memgetinfo) < allowed:
        found.append(
            (
                1,
                1,
                f"stale wwrMemGetInfo allowlist entry ({allowed} allowed, "
                f"{len(memgetinfo)} found, {owner}): lower or delete it",
            )
        )
    return sorted(found)


def check(names: Iterable[str], allowlist_path: Path = MEMGETINFO_ALLOWLIST) -> int:
    """The lint over `names` (files or directories). The allowlist among them
    (pre-commit passes it when it is staged) stands for the whole of src/."""
    names = list(names)
    if any(Path(n).resolve() == allowlist_path.resolve() for n in names):
        names = [n for n in names if Path(n).resolve() != allowlist_path.resolve()]
        names.append(os.path.relpath(REPO / "src"))
    allowlist = read_allowlist(allowlist_path)
    status = 0
    scanned: set[str] = set()
    roots = [repo_path(n).rstrip("/") + "/" for n in names if Path(n).is_dir()]
    for name in expand(names):
        rel = repo_path(name)
        scanned.add(rel)
        text = Path(name).read_text(encoding="utf-8")
        if is_generated(text):
            continue
        for lineno, col, message in find_bypasses(text, rel, allowlist):
            print(
                f"{name}:{lineno}:{col}: {message}; see docs/style.md "
                '"Allocations go through the pool", or exempt the line with '
                "`pool-bypass-ok: <reason>`"
            )
            status = 1
    # An entry under a scanned directory whose file is gone is stale too.
    for rel, (_, owner) in sorted(allowlist.items()):
        if rel not in scanned and any(rel.startswith(r) for r in roots):
            print(f"{allowlist_path}: stale entry {rel} ({owner}): the file is gone")
            status = 1
    return status


# -- changed -----------------------------------------------------------------


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout


def _git_reader(rev: str) -> Reader:
    def read(path: str) -> str | None:
        try:
            return _git("show", f"{rev}:{path}")
        except subprocess.CalledProcessError:
            return None

    return read


def _git_lister(rev: str) -> Lister:
    def ls(directory: str) -> list[str]:
        out = _git("ls-tree", "--name-only", rev, directory.rstrip("/") + "/")
        return [p for p in out.splitlines() if Path(p).suffix in CPP_SUFFIXES]

    return ls


def _added_lines(base: str, path: str) -> set[int]:
    lines: set[int] = set()
    diff = _git("diff", "--unified=0", base, "HEAD", "--", path)
    for m in re.finditer(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", diff, re.M):
        start, count = int(m.group(1)), int(m.group(2) or 1)
        lines.update(range(start, start + count))
    return lines


def diff_sites(
    old: list[Site], head: list[Site], touched: set[int]
) -> list[tuple[str, Site]]:
    """(added | resized, site) for each of one file's `head` sites on a
    `touched` line that `old` has no copy of; resized when `old` has a site of
    its kind, type and function whose size `head` no longer has."""
    old_full = {(s.kind, s.type, s.function, s.size) for s in old}
    head_full = {(s.kind, s.type, s.function, s.size) for s in head}
    gone = {key[:3] for key in old_full - head_full}
    out = []
    for s in head:
        key = (s.kind, s.type, s.function, s.size)
        if s.line in touched and key not in old_full:
            out.append(("resized" if key[:3] in gone else "added", s))
    return out


def changed(base: str, paths: list[str]) -> list[tuple[str, Site]]:
    """diff_sites over each C++ file that differs between `base` and HEAD."""
    files = [
        p
        for p in _git("diff", "--name-only", base, "HEAD", "--", *paths).splitlines()
        if Path(p).suffix in CPP_SUFFIXES
    ]
    out = []
    for path in files:
        head = inventory([path], _git_reader("HEAD"), _git_lister("HEAD"))
        old = inventory([path], _git_reader(base), _git_lister(base))
        out.extend(diff_sites(old, head, _added_lines(base, path)))
    return out


# -- the command line --------------------------------------------------------


def _row(site: Site) -> str:
    return "\t".join(
        [f"{site.file}:{site.line}", site.kind, site.type, site.size, site.function]
    )


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.partition("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory", help="one row per allocation site")
    inv.add_argument("--json", action="store_true")
    inv.add_argument("paths", nargs="+")
    chk = sub.add_parser("check", help="the pool-bypass lint")
    chk.add_argument("paths", nargs="*")
    chg = sub.add_parser("changed", help="sites added or resized since <base>")
    chg.add_argument("base")
    chg.add_argument("paths", nargs="*", default=["src"])
    args = parser.parse_args(argv)

    if args.command == "check":
        return check(args.paths)
    if args.command == "inventory":
        sites = inventory(expand(args.paths))
        if args.json:
            print(json.dumps([asdict(s) for s in sites], indent=2))
        else:
            for s in sites:
                print(_row(s))
        return 0
    for verdict, s in changed(args.base, args.paths):
        print(f"{verdict}\t{_row(s)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
