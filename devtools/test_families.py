#!/usr/bin/env python3
"""The test families (#984): each family of gtest cases, and its representative.

docs/testing.md "Families" is the contract. A family is

- a gtest suite, for a TEST or TEST_F;
- one TEST_P (or TYPED_TEST) across its parameters, with the
  INSTANTIATE_TEST_SUITE_P prefixes removed, so `Cases/Foo.Bar/x` and
  `More/Foo.Bar/y` are both the family `Foo.Bar`;
- or a part of one TEST_P, split off by a marker (below).

Its representative is the case marked with `FBF_FAMILY_REPRESENTATIVE` (from
test/support/slow_test.h), else the cheapest member by recorded times that
runs in the fast set, else none, and the family is flagged. A member does not
run there when its latest record says it reached `FBF_SKIP_IF_FAST_TESTS()`
(the record's `slow`; a member no record says that of counts as fast), and a
family whose every timed member is gated so is an error (#1125): it needs a
marker on a case that runs, or the exempt marker below.

The marker compiles to nothing; this script reads it from the source, and
places it in the test whose definition line (from gtest's own listing) is the
nearest one above it:

    FBF_FAMILY_REPRESENTATIVE();             in a TEST or TEST_F body
    FBF_FAMILY_REPRESENTATIVE("p");          in a TEST_P body: the instance p
    FBF_FAMILY_REPRESENTATIVE("p", "re");    ... of a family of its own, the
                                             instances whose param matches re

An instance is named by its param, or `Prefix/param` where two
instantiations share a param name. A split family is called `Foo.Bar[re]`.

A family with no case fit for the fast set is declared exempt instead (#1008),
by its form of the marker in any one of its cases, with the reason:

    FBF_FAMILY_EXEMPT("why");                the family of this case

An exempt family is reported as exempt, with its reason, not flagged, and
must not also have a marked representative.

Times come from the metrics branch's records (docs/testing.md "Metrics"):
`--metrics FILE`, or else the `metrics` branch of the source checkout when it
has one, filtered to this host and to the `nightly`, `weekly` and `manual`
kinds. `--log FILE` adds a gate's log: gtest's `[       OK ] Case (N ms)` and
ctest's `Passed  N sec` lines. A member's time is the median of its passes.
Whether it is gated comes from the same records, never from a log.

A case in a GPU binary (one whose file name ends `_gpu`) that never reaches
`FBF_REQUIRE_GPU()` is unguarded (#1037): docs/testing.md "Which tier a check
goes in" says it is most likely a host claim in the wrong binary. A case
reaches the guard when its body calls it, or calls a function or lambda that
does, at any depth, or when its fixture's class (or a base of it) does. The
analysis is by name, over the case's file and the headers beside it that it
includes by quotes, and reads code only: comments and string literals are
blanked first. A `--gtest-json` listing's binary is the file's stem.
`--unguarded` prints only those cases, one per test definition.

Exit status: 0 when every family has a representative, is exempt or is
flagged, 1 when a marker cannot be placed or contradicts another, or a
family's every timed member is gated, or under
`--unguarded` when a case is unguarded, 2 when nothing could be read.

Standard library only: it must run under a bare `python3`.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

SCHEMA = 1
MARKER = "FBF_FAMILY_REPRESENTATIVE"
EXEMPT = "FBF_FAMILY_EXEMPT"
_STRING = r'"(?:[^"\\]|\\.)*"'
_MACROS = "(" + MARKER + "|" + EXEMPT + ")"
# Either macro and its arguments, string literals only, so a regex argument
# may hold a parenthesis.
MARKER_RE = re.compile(
    r"\b"
    + _MACROS
    + r"\s*\(\s*((?:"
    + _STRING
    + r"\s*(?:,\s*"
    + _STRING
    + r"\s*)*)?)\)"
)
STRING_RE = re.compile(r'"((?:[^"\\]|\\.)*)"')
ANY_MARKER_RE = re.compile(r"\b" + _MACROS + r"\b")
FILTER_FLAG = "--gtest_filter="
TIMED_KINDS = ("nightly", "weekly", "manual")
GTEST_OK = re.compile(r"\[ +OK +\] (\S+) \((\d+) ms\)")
CTEST_PASSED = re.compile(r"Test +#\d+: (\S+) \.*\s+Passed\s+([0-9.]+) sec")
_ROOT = Path(__file__).resolve().parents[1]


def this_host() -> str:
    """This machine as its metrics records name it: metrics.host_name with
    devtools/config.sh's METRICS_HOST (an environment variable still wins),
    so the default filter matches what test.py nightly and pr --record wrote."""
    spec = importlib.util.spec_from_file_location(
        "fbf_metrics", _ROOT / "devtools" / "metrics.py"
    )
    metrics = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(metrics)
    override = subprocess.run(
        [
            "bash",
            "-c",
            '. "$1" && printf %s "$METRICS_HOST"',
            "_",
            str(_ROOT / "devtools" / "config.sh"),
        ],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return metrics.host_name(override)


@dataclass
class Case:
    name: str  # gtest's full name, as ctest's entry and the metrics carry it
    file: str  # relative to the source dir when inside it
    line: int
    test: str  # Suite.Test, prefixes and params removed
    param: str | None  # a parameterized case's param; None for TEST/TEST_F
    prefix: str | None  # its INSTANTIATE prefix, if any
    binary: str = ""  # the gtest binary's file name, when known
    fixture: str = ""  # the suite's own name: a TEST_F/TEST_P's class

    @property
    def in_gpu_binary(self) -> bool:
        return self.binary.endswith("_gpu")

    @property
    def default_family(self) -> str:
        return self.test if self.param is not None else self.test.split(".")[0]


@dataclass
class Marker:
    file: str
    line: int
    args: list[str] | None  # None: an argument that is not a string literal
    macro: str = MARKER  # or EXEMPT


@dataclass
class Family:
    name: str
    members: list[str] = field(default_factory=list)
    representative: str | None = None
    by: str | None = None  # "marked" | "cheapest" | "exempt"
    seconds: float | None = None
    reason: str | None = None  # an exempt family's


# --- reading ---------------------------------------------------------------


def cases_from_listing(listing: dict, source: Path, binary: str = "") -> list[Case]:
    """Cases from one gtest binary's `--gtest_list_tests --gtest_output=json`."""
    out = []
    for suite in listing.get("testsuites", []):
        for c in suite.get("testsuite", []):
            if "DISABLED_" in suite["name"] or "DISABLED_" in c["name"]:
                continue
            parts = suite["name"].split("/")
            typed = "type_param" in c
            valued = "value_param" in c
            if typed:  # Suite/N or Prefix/Suite/N
                param, parts = parts[-1], parts[:-1]
            test, _, vparam = c["name"].partition("/")
            if valued:
                param = vparam
            prefix = "/".join(parts[:-1]) or None
            out.append(
                Case(
                    name=f"{suite['name']}.{c['name']}",
                    file=rel(c.get("file", ""), source),
                    line=int(c.get("line", 0)),
                    test=f"{parts[-1]}.{test}",
                    param=param if (typed or valued) else None,
                    prefix=prefix,
                    binary=binary,
                    fixture=parts[-1],
                )
            )
    return out


def rel(path: str, source: Path) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(source).as_posix()
    except (ValueError, OSError):
        return path


def run_gtest_listing(exe: Path) -> dict | None:
    """gtest's own listing; it walks the registry and never starts the GPU."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "list.json"
        try:
            subprocess.run(
                [str(exe), "--gtest_list_tests", f"--gtest_output=json:{out}"],
                capture_output=True,
                check=False,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return json.loads(out.read_text()) if out.exists() else None


def ctest_cases(build: Path) -> tuple[dict[str, set[str]], list[str]]:
    """{gtest binary: the case names ctest runs from it}, and the ctest entries
    that are not one gtest case (the examples, run through a cmake script)."""
    proc = subprocess.run(
        ["ctest", "--test-dir", str(build), "--show-only=json-v1"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or "{" not in proc.stdout:
        raise SystemExit(f"test_families: ctest could not list {build}")
    listing = json.loads(proc.stdout[proc.stdout.index("{") :])
    by_binary: dict[str, set[str]] = defaultdict(set)
    other = []
    for t in listing.get("tests", []):
        cmd = t.get("command") or []
        filters = [a[len(FILTER_FLAG) :] for a in cmd if a.startswith(FILTER_FLAG)]
        if cmd and len(filters) == 1:
            by_binary[cmd[0]].add(filters[0])
        else:
            other.append(t["name"])
    return by_binary, other


def build_cases(build: Path, source: Path) -> tuple[list[Case], list[str]]:
    """Every gtest case a built build dir's ctest runs, and the ctest entries
    that are not one gtest case. Raises SystemExit when the dir or a binary
    cannot be listed. devtools/slow_rule.py reads the families through it."""
    if not (build / "CTestTestfile.cmake").exists():
        raise SystemExit(f"test_families: {build} is not a configured build dir")
    by_binary, other = ctest_cases(build)
    cases: list[Case] = []
    for exe, names in sorted(by_binary.items()):
        listing = run_gtest_listing(Path(exe))
        if listing is None:
            raise SystemExit(f"test_families: cannot list {exe}")
        cases += [
            c
            for c in cases_from_listing(listing, source, Path(exe).name)
            if c.name in names
        ]
    return cases, other


def markers_in(source: Path, files: set[str]) -> list[Marker]:
    """Every marker in the given test files. A line comment or a #define is
    not a marker, so the macro's own definition and its documentation are not."""
    out = []
    for f in sorted(files):
        path = source / f
        if not path.is_file():
            continue
        for n, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            code = line.split("//", 1)[0]
            if code.lstrip().startswith("#"):
                continue
            found = list(MARKER_RE.finditer(code))
            for m in found:
                args = [
                    re.sub(r"\\(.)", r"\1", a) for a in STRING_RE.findall(m.group(2))
                ]
                out.append(Marker(f, n, args, m.group(1)))
            for macro in ANY_MARKER_RE.findall(code)[len(found) :]:
                out.append(Marker(f, n, None, macro))  # not string literals: an error
    return out


def parse_log(text: str) -> dict[str, list[float]]:
    """Pass times from a gate's log: gtest's OK lines and ctest's Passed lines."""
    out: dict[str, list[float]] = defaultdict(list)
    for line in text.splitlines():
        if m := GTEST_OK.search(line):
            out[m.group(1)].append(int(m.group(2)) / 1000)
        elif m := CTEST_PASSED.search(line):
            out[m.group(1)].append(float(m.group(2)))
    return out


def parse_metrics(text: str, host: str | None) -> dict[str, list[float]]:
    """Pass times from metrics records (schema 1, one JSON object a line)."""
    out: dict[str, list[float]] = defaultdict(list)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("schema") != SCHEMA or rec.get("kind") not in TIMED_KINDS:
            continue
        if host is not None and rec.get("host") != host:
            continue
        for c in rec.get("cases", []):
            if c.get("status") == "pass" and isinstance(c.get("seconds"), int | float):
                out[c.get("name", "")].append(float(c["seconds"]))
    return out


def parse_slow(text: str, host: str | None) -> dict[str, bool]:
    """Each case's `slow` from the same records parse_metrics reads: whether
    it reached FBF_SKIP_IF_FAST_TESTS(), so a fast run skips it. The latest
    record that carries the field wins, so a case ungated since counts as
    fast again."""
    out: dict[str, bool] = {}
    for line in text.splitlines():
        try:
            rec = json.loads(line) if line.strip() else None
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict) or rec.get("schema") != SCHEMA:
            continue
        if rec.get("kind") not in TIMED_KINDS:
            continue
        if host is not None and rec.get("host") != host:
            continue
        for c in rec.get("cases", []):
            if isinstance(c.get("slow"), bool):
                out[c.get("name", "")] = c["slow"]
    return out


def metrics_branch_text(source: Path) -> str | None:
    """The `metrics` branch's records, when the checkout has the branch."""

    def git(*args: str) -> str | None:
        proc = subprocess.run(
            ["git", "-C", str(source), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        return proc.stdout if proc.returncode == 0 else None

    for ref in ("metrics", "origin/metrics"):
        names = git("ls-tree", "--name-only", f"{ref}:runs")
        if names is None:
            continue
        files = sorted(n for n in names.split() if n.endswith(".jsonl"))
        return "".join(git("show", f"{ref}:runs/{n}") or "" for n in files)
    return None


# --- the GPU guard (#1037) -------------------------------------------------

GUARD = "FBF_REQUIRE_GPU"
# A comment, a raw string, a string or a character literal: blanked before the
# guard is looked for, so a mention of it in prose is not a call.
_NOT_CODE = re.compile(
    r"//[^\n]*"
    r"|/\*.*?\*/"
    r'|R"([^(\s]*)\(.*?\)\1"'
    r'|"(?:[^"\\\n]|\\.)*"'
    r"|'(?:[^'\\\n]|\\.)*'",
    re.S,
)
_CALL = re.compile(r"\b([A-Za-z_]\w*)\s*(?:<[^;{}()]*>)?\s*\(")
_LAMBDA = re.compile(r"\b([A-Za-z_]\w*)\s*=\s*\[[^\]]*\]")
_CLASS = re.compile(r"\b(?:class|struct)\s+([A-Za-z_]\w*)\b([^;{}()]*)\{")
_QUOTED_INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.M)
_TEST_MACRO = re.compile(r"\b(TEST|TEST_F|TEST_P|TYPED_TEST|TYPED_TEST_P)\s*\(")
# What may sit between a definition's parameter list and its body.
_BEFORE_BODY = re.compile(
    r"(?:\s|const\b|noexcept\b|override\b|final\b|mutable\b|&|->\s*[\w:<>,*&\s]+?)*\{"
)
_KEYWORDS = frozenset(
    ("if", "for", "while", "switch", "catch", "return", "sizeof", "alignof")
    + ("decltype", "static_assert", "noexcept", "requires")
)


def code_only(text: str) -> str:
    """The text with comments and string and character literals blanked, every
    newline kept, so offsets and line numbers still hold."""
    return _NOT_CODE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def _past_match(code: str, i: int, open_: str, close: str) -> int:
    """The index just past the bracket that closes the one at code[i]."""
    depth = 0
    for k in range(i, len(code)):
        if code[k] == open_:
            depth += 1
        elif code[k] == close:
            depth -= 1
            if depth == 0:
                return k + 1
    return len(code)


def _block(code: str, i: int) -> str:
    """The brace block that opens at the first `{` at or after i."""
    j = code.find("{", i)
    return "" if j < 0 else code[j : _past_match(code, j, "{", "}")]


def _bases(clause: str) -> list[str]:
    """The base classes a class head names after its `:`, by their last name:
    `: public ::testing::TestWithParam<Case>` is TestWithParam."""
    if ":" not in clause:
        return []
    out = []
    for part in clause.split(":", 1)[1].split(","):
        names = re.findall(r"[A-Za-z_]\w*", re.sub(r"<.*", "", part))
        if names:
            out.append(names[-1])
    return out


def _member_blocks(body: str) -> str:
    """The brace blocks one level inside a class body: its methods' bodies and
    its members' initialisers, without the signatures between them, so a
    method's own name is not read as a call."""
    out = []
    i = 1
    while (j := body.find("{", i)) >= 0:
        end = _past_match(body, j, "{", "}")
        out.append(body[j:end])
        i = end
    return "\n".join(out)


def guarding_names(code: str) -> set[str]:
    """Every function, lambda or class defined in code (already code_only) that
    reaches the guard: whose body calls it or a name that does, or, for a
    class, one of whose bases does. A class's calls are its members' bodies',
    so a SetUp() that calls the guard makes the class reach it."""
    bodies: dict[str, list[str]] = defaultdict(list)
    bases: dict[str, set[str]] = defaultdict(set)
    for m in _CLASS.finditer(code):
        bodies[m.group(1)].append(_member_blocks(_block(code, m.end() - 1)))
        bases[m.group(1)].update(_bases(m.group(2)))
    for m in _LAMBDA.finditer(code):
        bodies[m.group(1)].append(_block(code, m.end()))
    for m in _CALL.finditer(code):
        if m.group(1) in _KEYWORDS or _TEST_MACRO.match(code, m.start()):
            continue
        end = _past_match(code, m.end() - 1, "(", ")")
        if _BEFORE_BODY.match(code, end):
            bodies[m.group(1)].append(_block(code, end))
    calls = {n: {c for b in bs for c in _CALL.findall(b)} for n, bs in bodies.items()}
    found: set[str] = set()
    grew = True
    while grew:
        grew = False
        for name, called in calls.items():
            if name not in found and (
                GUARD in called or called & found or bases[name] & found
            ):
                found.add(name)
                grew = True
    return found


def _with_headers(path: Path, seen: set[Path] | None = None) -> str:
    """A file's text, then each header beside it that it includes by quotes,
    recursively. A header found elsewhere on the include path is not read."""
    seen = set() if seen is None else seen
    if path in seen or not path.is_file():
        return ""
    seen.add(path)
    text = path.read_text(errors="replace")
    parts = [text]
    for name in _QUOTED_INCLUDE.findall(text):
        parts.append(_with_headers(path.parent / name, seen))
    return "\n".join(parts)


def unguarded(cases: list[Case], source: Path) -> list[Case]:
    """The GPU-binary cases that never reach FBF_REQUIRE_GPU(), one per test
    definition: a TEST_P's instances, or a TYPED_TEST's types, share theirs."""
    by_file: dict[str, list[Case]] = defaultdict(list)
    for c in cases:
        if c.in_gpu_binary:
            by_file[c.file].append(c)
    out = []
    for f, members in sorted(by_file.items()):
        path = source / f
        if not path.is_file():
            continue
        own = code_only(path.read_text(errors="replace"))
        found = guarding_names(code_only(_with_headers(path)))
        starts = [0]
        for line in own.splitlines(keepends=True):
            starts.append(starts[-1] + len(line))
        seen: set[tuple[int, str]] = set()
        for c in sorted(members, key=lambda c: (c.line, c.name)):
            if (c.line, c.test) in seen or not 0 < c.line < len(starts):
                continue
            seen.add((c.line, c.test))
            m = _TEST_MACRO.search(own, starts[c.line - 1])
            if m is None:
                continue
            body = _block(own, _past_match(own, m.end() - 1, "(", ")"))
            called = set(_CALL.findall(body))
            # A plain TEST's suite is only a name, not a class it runs in.
            fixture = m.group(1) != "TEST" and c.fixture in found
            if GUARD in called or called & found or fixture:
                continue
            out.append(c)
    return out


# --- the families ----------------------------------------------------------


def families(
    cases: list[Case],
    markers: list[Marker],
    times: dict[str, list[float]] | None = None,
    slow: dict[str, bool] | None = None,
) -> tuple[dict[str, Family], list[str], list[str]]:
    """(families by name, errors, notes). `slow` is parse_slow's: a member
    it calls slow is never picked as the cheapest."""
    errors: list[str] = []
    notes: list[str] = []
    by_file: dict[str, list[Case]] = defaultdict(list)
    for c in cases:
        by_file[c.file].append(c)

    marked: dict[str, tuple[Marker, str]] = {}  # family -> (marker, case)
    exempt: dict[str, tuple[Marker, str]] = {}  # family -> (marker, reason)
    splits: dict[str, list[tuple[str, re.Pattern]]] = defaultdict(list)
    for mk in markers:
        where = f"{mk.file}:{mk.line}"
        if mk.args is None:
            errors.append(f"{where}: {mk.macro}'s arguments are string literals only")
            continue
        above = [c for c in by_file.get(mk.file, []) if c.line <= mk.line]
        if not above:
            errors.append(f"{where}: {mk.macro} is not inside any listed test")
            continue
        line = max(c.line for c in above)
        owners = [c for c in above if c.line == line]
        tests = {c.test for c in owners}
        if len(tests) != 1:
            errors.append(f"{where}: {mk.macro} is ambiguous among {sorted(tests)}")
            continue
        test = tests.pop()
        if mk.macro == EXEMPT:
            if len(mk.args) != 1 or not mk.args[0].strip():
                errors.append(f"{where}: {EXEMPT} takes one string literal, the reason")
                continue
            family = owners[0].default_family
            if family in exempt:
                first = exempt[family][0]
                errors.append(
                    f"{where}: {family} is already exempt at {first.file}:{first.line}"
                )
                continue
            exempt[family] = (mk, mk.args[0])
            continue
        if owners[0].param is None:
            if mk.args:
                errors.append(f"{where}: {MARKER} in a TEST takes no arguments")
                continue
            family, case = owners[0].default_family, owners[0].name
        else:
            if not mk.args or len(mk.args) > 2:
                errors.append(
                    f"{where}: {MARKER} in {test}, a parameterized test, "
                    'names the instance: ("param"[, "regex"])'
                )
                continue
            hits = [c for c in owners if mk.args[0] in (c.param, qualified(c))]
            if len(hits) != 1:
                what = "matches no instance" if not hits else "is ambiguous"
                errors.append(f'{where}: {MARKER}("{mk.args[0]}") {what} of {test}')
                continue
            case, family = hits[0].name, test
            if len(mk.args) == 2:
                try:
                    pattern = re.compile(mk.args[1])
                except re.error as e:
                    errors.append(f"{where}: bad family regex {mk.args[1]!r}: {e}")
                    continue
                if not pattern.search(qualified(hits[0])):
                    errors.append(
                        f"{where}: the family regex {mk.args[1]!r} does not match "
                        f"its own representative {case}"
                    )
                    continue
                family = f"{test}[{mk.args[1]}]"
                splits[test].append((family, pattern))
        if family in marked:
            first = marked[family][0]
            errors.append(
                f"{where}: {family} already has a representative marked at "
                f"{first.file}:{first.line}"
            )
            continue
        marked[family] = (mk, case)

    out: dict[str, Family] = {}
    for c in sorted(cases, key=lambda c: c.name):
        name = c.default_family
        hit = [f for f, p in splits.get(c.test, []) if p.search(qualified(c))]
        if len(hit) > 1:
            errors.append(f"{c.name} is in two split families: {hit}")
        if hit:
            name = hit[0]
        out.setdefault(name, Family(name)).members.append(c.name)

    times = times or {}
    slow = slow or {}
    for fam in out.values():
        if fam.name in exempt:
            mk, fam.reason = exempt[fam.name]
            fam.by = "exempt"
            if fam.name in marked:
                errors.append(
                    f"{mk.file}:{mk.line}: {fam.name} is exempt and also has a "
                    "marked representative"
                )
            continue
        if fam.name in marked:
            fam.representative, fam.by = marked[fam.name][1], "marked"
            if fam.representative not in fam.members:
                errors.append(
                    f"{fam.name}: its marked representative {fam.representative} "
                    "is not one of its members"
                )
            t = times.get(fam.representative)
            fam.seconds = statistics.median(t) if t else None
            timed = [m for m in fam.members if times.get(m)]
            if timed and not t:
                notes.append(
                    f"{fam.name}: the marked {fam.representative} has no pass "
                    "time where other members do (skipped in a fast run?)"
                )
            continue
        timed = [(statistics.median(times[m]), m) for m in fam.members if times.get(m)]
        fast = [(t, m) for t, m in timed if not slow.get(m)]
        if fast:
            fam.seconds, fam.representative = min(fast)
            fam.by = "cheapest"
        elif timed:
            errors.append(
                f"{fam.name}: every timed member reaches FBF_SKIP_IF_FAST_TESTS(),"
                f" so none runs in the fast set: mark one that does with {MARKER},"
                f" or exempt the family with {EXEMPT}"
            )
    return out, errors, notes


def qualified(c: Case) -> str:
    return f"{c.prefix}/{c.param}" if c.prefix else (c.param or "")


# --- the report ------------------------------------------------------------


def report(fams: dict[str, Family], out, *, flagged_only: bool = False) -> None:
    rows = []
    for f in sorted(fams.values(), key=lambda f: f.name):
        if f.by == "marked":
            how = "marked" + (f" ({f.seconds:.3f} s)" if f.seconds is not None else "")
        elif f.by == "cheapest":
            how = f"cheapest ({f.seconds:.3f} s)"
        elif f.by == "exempt":
            how = f"exempt: {f.reason}"
        else:
            how = "FLAGGED: not marked, and no recorded time"
        if flagged_only and f.by is not None:
            continue
        rows.append((f.name, f.representative or "-", how, len(f.members)))
    for name, rep, how, n in rows:
        print(f"{name}  [{n}]  ->  {rep}  {how}", file=out)


def summary(fams: dict[str, Family], times_from: str) -> str:
    count = defaultdict(int)
    for f in fams.values():
        count[f.by] += 1
    members = sum(len(f.members) for f in fams.values())
    return (
        f"test_families: {len(fams)} families over {members} cases: "
        f"{count['marked']} marked, {count['cheapest']} cheapest by time, "
        f"{count['exempt']} exempt, {count[None]} flagged; times: {times_from}"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="devtools/test_families.py",
        description="List every gtest family and its representative.",
    )
    ap.add_argument(
        "build_dir",
        nargs="?",
        type=Path,
        help="a built build dir: its ctest entries and gtest binaries are listed",
    )
    ap.add_argument(
        "--gtest-json",
        action="append",
        type=Path,
        default=[],
        help="a saved `--gtest_list_tests --gtest_output=json` (repeatable), "
        "instead of a build dir",
    )
    ap.add_argument(
        "--source",
        type=Path,
        default=_ROOT,
        help="the source tree the markers are read from (default: this checkout)",
    )
    ap.add_argument(
        "--metrics",
        action="append",
        type=Path,
        default=[],
        help="a metrics .jsonl file (repeatable; default: the metrics branch)",
    )
    ap.add_argument(
        "--no-metrics", action="store_true", help="do not read the metrics branch"
    )
    ap.add_argument(
        "--log",
        action="append",
        type=Path,
        default=[],
        help="a gate's log to take pass times from (repeatable)",
    )
    ap.add_argument(
        "--host",
        default=None,
        help="the host whose metrics records count (default: this one, as"
        " devtools/metrics.py names it)",
    )
    ap.add_argument("--json", action="store_true", help="print the families as JSON")
    ap.add_argument(
        "--flagged", action="store_true", help="print only the flagged families"
    )
    ap.add_argument(
        "--unguarded",
        action="store_true",
        help="print only the GPU-binary cases that never reach FBF_REQUIRE_GPU(),"
        " and exit 1 when there are any",
    )
    args = ap.parse_args(argv)
    source = args.source.resolve()

    cases: list[Case] = []
    other: list[str] = []
    if args.build_dir is not None:
        try:
            cases, other = build_cases(args.build_dir.resolve(), source)
        except SystemExit as e:
            print(e, file=sys.stderr)
            return 2
    for path in args.gtest_json:
        cases += cases_from_listing(json.loads(path.read_text()), source, path.stem)
    if not cases:
        print(
            "test_families: no cases: pass a build dir or --gtest-json", file=sys.stderr
        )
        return 2

    bare = unguarded(cases, source)
    if args.unguarded:
        for c in bare:
            print(f"{c.file}:{c.line}  {c.test}  ({c.binary})")
        gpu = sum(c.in_gpu_binary for c in cases)
        print(
            f"test_families: {len(bare)} unguarded test definitions over {gpu} "
            "GPU-binary cases (no FBF_REQUIRE_GPU() reached)",
            file=sys.stderr,
        )
        return 1 if bare else 0

    if args.host is None:
        args.host = this_host()
    times: dict[str, list[float]] = defaultdict(list)
    slow: dict[str, bool] = {}
    sources = []
    texts = [p.read_text() for p in args.metrics]
    if args.metrics:
        sources.append(f"{len(args.metrics)} metrics file(s), host {args.host}")
    elif not args.no_metrics:
        branch = metrics_branch_text(source)
        if branch is not None:
            texts.append(branch)
            sources.append(f"the metrics branch, host {args.host}")
    for text in texts:
        for k, v in parse_metrics(text, args.host).items():
            times[k] += v
        slow.update(parse_slow(text, args.host))
    for path in args.log:
        for k, v in parse_log(path.read_text(errors="replace")).items():
            times[k] += v
        sources.append(str(path))

    markers = markers_in(source, {c.file for c in cases})
    fams, errors, notes = families(cases, markers, times, slow)
    times_from = ", ".join(sources) if sources else "none (no metrics records yet)"

    if args.json:
        doc = {
            "schema": SCHEMA,
            "times": times_from,
            "families": {
                f.name: {
                    "members": f.members,
                    "representative": f.representative,
                    "by": f.by,
                    "seconds": f.seconds,
                    "reason": f.reason,
                }
                for f in sorted(fams.values(), key=lambda f: f.name)
            },
            "flagged": sorted(f.name for f in fams.values() if f.by is None),
            "exempt": sorted(f.name for f in fams.values() if f.by == "exempt"),
            "not_gtest": sorted(other),
            "unguarded": [
                {"test": c.test, "file": c.file, "line": c.line, "binary": c.binary}
                for c in bare
            ],
            "errors": errors,
            "notes": notes,
        }
        print(json.dumps(doc, indent=1))
    else:
        report(fams, sys.stdout, flagged_only=args.flagged)
    for n in notes:
        print(f"note: {n}", file=sys.stderr)
    if other:
        print(
            f"note: {len(other)} ctest entries are not gtest cases: "
            + ", ".join(sorted(other)),
            file=sys.stderr,
        )
    if bare:
        print(
            f"note: {len(bare)} GPU-binary test definitions never reach "
            "FBF_REQUIRE_GPU() (--unguarded lists them)",
            file=sys.stderr,
        )
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    print(summary(fams, times_from), file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
