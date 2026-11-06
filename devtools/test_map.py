#!/usr/bin/env python3
"""The test-dependency map (#983): which tests each source file can affect.

From a configured, built (or at least scanned) build directory it reads

- every unit's `.ddi` (the dependency scanner's P1689 output): the modules it
  provides and the modules it imports directly. An implementation unit
  provides nothing; its imports belong to the module its target provides;
- every unit's `.ddi.d`, and for units the scanner never sees (`.cu` device
  code) `ninja -t deps`: the headers each unit includes;
- the CMake File API codemodel (`codemodel-v2`): which target compiles each
  source, and what each target links. A `.cu` in a device library reaches a
  test only through a link edge, never through an import;
- `ctest --show-only=json-v1` and each gtest binary's `--gtest_list_tests`
  JSON: which ctest entries each test file defines.

and writes one JSON map, source file -> owner (module or target) -> test files
(with a distance) -> ctest entries. docs/testing.md "Test selection" is the
contract it serves. Every file under the reported roots that it cannot place
is listed, with a reason, never dropped. It is not wired into any gate.

Standard library only: it must run under a bare `python3`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from collections import defaultdict, deque
from collections.abc import Callable, Iterable
from pathlib import Path

SCHEMA = 1
QUERY_CLIENT = "client-free-breakfast-test-map"
FILTER_FLAG = "--gtest_filter="
DEFAULT_ROOTS = ("src", "test")
# The kinds of file the map does not place by design, and why. Anything else
# it cannot place is "other", and is the part of the report worth reading.
RUN_EVERYTHING = {
    "cmake": "build configuration (CMakeLists.txt, *.cmake, *.in): runs everything",
    "data": "data compiled in or read at run time: runs everything",
}
NOT_CPP = {
    "unconfigured": "its directory is not configured in this build (another preset)",
    "config": "tool configuration (.clang-format and the like): no build effect",
    "doc": "documentation: no build effect",
    "python": "Python: the pytest tier, not this map",
}


# --- small readers -----------------------------------------------------------


def parse_depfile(text: str) -> list[str]:
    """The prerequisites of a make-style depfile, in order. The first is the
    unit's own source."""
    text = text.replace("\\\n", " ")
    _, sep, rest = text.partition(": ")
    if not sep:
        return []
    deps, token, i = [], "", 0
    while i < len(rest):
        ch = rest[i]
        if ch == "\\" and i + 1 < len(rest) and rest[i + 1] in " #\\":
            token += rest[i + 1]
            i += 2
            continue
        if ch == "$" and rest[i + 1 : i + 2] == "$":
            token += "$"
            i += 2
            continue
        if ch.isspace():
            if token:
                deps.append(token)
            token = ""
            if ch == "\n":
                break  # one rule; anything after is a phony line
        else:
            token += ch
        i += 1
    if token:
        deps.append(token)
    return deps


def parse_ninja_deps(text: str) -> dict[str, list[str]]:
    """`ninja -t deps`: object (relative to the build dir) -> its deps."""
    out: dict[str, list[str]] = {}
    current: list[str] | None = None
    for line in text.splitlines():
        if not line.strip():
            current = None
        elif not line[0].isspace() and ": #deps" in line:
            current = out.setdefault(line.split(": #deps", 1)[0], [])
        elif current is not None:
            current.append(line.strip())
    return out


def target_of_object(obj: str) -> str | None:
    """The target an object belongs to, from CMake's `CMakeFiles/<t>.dir/`."""
    head, sep, _ = obj.rpartition(".dir/")
    if not sep or "CMakeFiles/" not in head:
        return None
    return head.rsplit("CMakeFiles/", 1)[1]


def module_of(logical_name: str) -> str:
    """A partition `m:p` belongs to module `m`."""
    return logical_name.split(":", 1)[0]


# --- the build's own records -------------------------------------------------


def ensure_query(build: Path) -> Path:
    """Write the File API codemodel query; CMake answers it at configure."""
    query = build / ".cmake/api/v1/query" / QUERY_CLIENT / "query.json"
    query.parent.mkdir(parents=True, exist_ok=True)
    if not query.exists():
        query.write_text('{"requests":[{"kind":"codemodel","version":2}]}\n')
    return query


def load_codemodel(build: Path) -> dict | None:
    """The latest codemodel-v2 reply, or None when there is none yet."""
    reply = build / ".cmake/api/v1/reply"
    indexes = sorted(reply.glob("index-*.json"))
    if not indexes:
        return None
    index = json.loads(indexes[-1].read_text())
    for obj in index.get("objects", []):
        if obj.get("kind") == "codemodel":
            return json.loads((reply / obj["jsonFile"]).read_text())
    return None


def load_targets(build: Path, codemodel: dict) -> dict[str, dict]:
    """Every target, concrete and abstract (imported, interface), by name,
    with its links as names."""
    reply = build / ".cmake/api/v1/reply"
    config = codemodel["configurations"][0]
    entries = config.get("targets", []) + config.get("abstractTargets", [])
    raw = {e["id"]: json.loads((reply / e["jsonFile"]).read_text()) for e in entries}
    names = {tid: t["name"] for tid, t in raw.items()}
    targets = {}
    for t in raw.values():
        # codemodel 2.9 has the direct links; older replies only the build
        # dependencies, a superset.
        links = t.get("linkLibraries", t.get("dependencies", []))
        iface = t.get("interfaceLinkLibraries", [])
        targets[t["name"]] = {
            "type": t.get("type", ""),
            "abstract": bool(t.get("abstract")) or t.get("type") == "INTERFACE_LIBRARY",
            "sources": [s["path"] for s in t.get("sources", [])],
            "artifacts": [a["path"] for a in t.get("artifacts", [])],
            "links": [names[x["id"]] for x in links if x.get("id") in names],
            "interface_links": [names[x["id"]] for x in iface if x.get("id") in names],
        }
    return targets


def read_units(build: Path) -> list[dict]:
    """One record per scanned unit: object, target, source, provides,
    requires, headers."""
    units = []
    for ddi in sorted(build.rglob("*.ddi")):
        doc = json.loads(ddi.read_text())
        depfile = ddi.with_name(ddi.name + ".d")
        deps = parse_depfile(depfile.read_text()) if depfile.exists() else []
        for rule in doc.get("rules", []):
            obj = rule.get("primary-output", "")
            units.append(
                {
                    "object": obj,
                    "target": target_of_object(obj),
                    "source": deps[0] if deps else None,
                    "provides": [p["logical-name"] for p in rule.get("provides", [])],
                    "requires": [r["logical-name"] for r in rule.get("requires", [])],
                    "headers": deps[1:],
                }
            )
    return units


def run_ninja_deps(build: Path) -> str | None:
    ninja = shutil.which("ninja")
    if ninja is None or not (build / "build.ninja").exists():
        return None
    proc = subprocess.run(
        [ninja, "-C", str(build), "-t", "deps"],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout if proc.returncode == 0 else None


def run_ctest_listing(build: Path) -> dict | None:
    ctest = shutil.which("ctest")
    if ctest is None:
        return None
    proc = subprocess.run(
        [ctest, "--test-dir", str(build), "--show-only=json-v1"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0 or "{" not in proc.stdout:
        return None
    return json.loads(proc.stdout[proc.stdout.index("{") :])


def run_gtest_listing(exe: Path) -> dict | None:
    """gtest's own listing, with each case's file. It walks the registry only,
    so it never initialises the GPU runtime."""
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


# --- the map -----------------------------------------------------------------


class Tree:
    """Paths as the map writes them: relative to the source dir when inside
    it, and absolute (system and build-dir files, which no diff touches)
    otherwise."""

    def __init__(self, source: Path, build: Path) -> None:
        self.source = source.resolve()
        self.build = build.resolve()

    def rel(self, path: str | Path, base: Path | None = None) -> str | None:
        p = Path(path)
        if not p.is_absolute():
            p = (base or self.source) / p
        p = Path(os.path.normpath(p))
        if p.is_relative_to(self.build) or not p.is_relative_to(self.source):
            return None
        return p.relative_to(self.source).as_posix()


def classify_unmapped(
    path: str, source: Path | None = None, configured: set[str] | None = None
) -> str:
    """Why a file is not in the map. `configured` is the codemodel's source
    directories: a file whose nearest CMakeLists.txt is outside them belongs
    to a directory this preset never adds (experimental/, say)."""
    name = path.rsplit("/", 1)[-1]
    if source is not None and configured is not None:
        for parent in Path(path).parents:
            if (source / parent / "CMakeLists.txt").exists():
                if parent.as_posix() not in configured:
                    return "unconfigured"
                break
    if name.startswith("."):
        return "config"
    if name == "CMakeLists.txt" or name.endswith((".cmake", ".in")):
        return "cmake"
    if name.endswith((".md", ".txt", ".rst")):
        return "doc"
    if name.endswith(".py"):
        return "python"
    parts = path.split("/")
    if "data" in parts[:-1]:
        return "data"
    return "other"


def walk_roots(source: Path, roots: Iterable[str], skip: Path) -> list[str]:
    files = []
    for root in roots:
        base = source / root
        for dirpath, dirnames, filenames in os.walk(base):
            d = Path(dirpath)
            dirnames[:] = sorted(
                n
                for n in dirnames
                if not n.startswith(".")
                and n != "__pycache__"
                and not (d / n).resolve().is_relative_to(skip)
            )
            files += [(d / f).relative_to(source).as_posix() for f in sorted(filenames)]
    return files


def extract(
    build: Path,
    *,
    roots: Iterable[str] = DEFAULT_ROOTS,
    ctest_listing: dict | None = None,
    gtest_listing: Callable[[Path], dict | None] = run_gtest_listing,
    ninja_deps: Callable[[Path], str | None] = run_ninja_deps,
) -> dict:
    """The map, as one JSON-ready dict. `ctest_listing` defaults to running
    ctest; the two callables are what the toy-tree tests replace."""
    build = build.resolve()
    notes: list[str] = []
    codemodel = load_codemodel(build)
    if codemodel is None:
        raise SystemExit(
            f"test_map: no codemodel reply in {build}/.cmake/api/v1/reply -- "
            "the query is written now; re-run cmake on the build dir"
        )
    tree = Tree(Path(codemodel["paths"]["source"]), build)
    targets = load_targets(build, codemodel)
    units = read_units(build)

    # Which module each target provides (one per target; std's is synthetic).
    provider: dict[str, str] = {}  # module -> target
    target_module: dict[str, str] = {}  # target -> module
    for u in units:
        for p in u["provides"]:
            m = module_of(p)
            provider.setdefault(m, u["target"])
            if not m.startswith("std"):
                target_module.setdefault(u["target"], m)

    def node(target: str) -> str:
        return target_module.get(target, target)

    # ctest entries -> (binary target, gtest case); each case -> its file.
    if ctest_listing is None:
        ctest_listing = run_ctest_listing(build)
        if ctest_listing is None:
            notes.append("ctest listing unavailable: no ctest entries mapped")
    artifact_target = {
        os.path.normpath(build / a): name
        for name, t in targets.items()
        for a in t["artifacts"]
    }
    entries_by_case: dict[tuple[str, str], list[str]] = defaultdict(list)
    unmapped_entries: list[dict] = []
    gtest_bins: dict[str, Path] = {}
    for test in (ctest_listing or {}).get("tests", []):
        command = test.get("command") or []
        filters = [a[len(FILTER_FLAG) :] for a in command if a.startswith(FILTER_FLAG)]
        exe = os.path.normpath(command[0]) if command else ""
        tgt = artifact_target.get(exe)
        if tgt is None or len(filters) != 1:
            unmapped_entries.append(
                {"name": test["name"], "reason": "not one gtest case of a built target"}
            )
            continue
        gtest_bins[tgt] = Path(exe)
        entries_by_case[(tgt, filters[0])].append(test["name"])

    case_file: dict[tuple[str, str], str] = {}
    for tgt, exe in sorted(gtest_bins.items()):
        listing = gtest_listing(exe)
        if listing is None:
            notes.append(f"gtest listing failed for {exe}")
            continue
        for suite in listing.get("testsuites", []):
            for case in suite.get("testsuite", []):
                rel = tree.rel(case.get("file", ""), build)
                if rel:
                    case_file[(tgt, f"{suite['name']}.{case['name']}")] = rel

    test_entries: dict[str, list[str]] = defaultdict(list)
    for key, names in entries_by_case.items():
        f = case_file.get(key)
        if f is None:
            unmapped_entries += [
                {"name": n, "reason": "case not found in its binary's listing"}
                for n in names
            ]
        else:
            test_entries[f] += names
    test_target = {f: t for (t, _), f in case_file.items()}

    # Graph nodes are owners: a module (its target) or a non-module target.
    # Edges point from a dependent to what it depends on.
    edges: dict[str, set[str]] = defaultdict(set)
    files: dict[str, dict] = {}
    unresolved: set[str] = set()

    def link_closure(name: str) -> set[str]:
        """The non-module targets `name` links, looking through interface and
        imported targets, which compile nothing of their own."""
        out: set[str] = set()
        stack, seen = list(targets[name]["links"]), set()
        while stack:
            dep = stack.pop()
            if dep in seen or dep not in targets:
                continue
            seen.add(dep)
            t = targets[dep]
            if t["abstract"]:
                stack += t["interface_links"] or t["links"]
            elif dep not in target_module and t["sources"]:
                out.add(dep)
        return out

    def add_file(path: str, kind: str, owner: str) -> None:
        rec = files.setdefault(path, {"kinds": set(), "owners": set()})
        rec["kinds"].add(kind)
        rec["owners"].add(owner)

    tests: dict[str, dict] = {}
    for u in units:
        src = tree.rel(u["source"], build) if u["source"] else None
        tgt = u["target"]
        if tgt is None or tgt not in targets:
            if tgt and tgt.startswith("__cmake_"):
                continue  # CMake's own `import std` target; nothing to place
            notes.append(f"unit {u['object']}: target not in the codemodel")
            continue
        if src in test_target:
            owner = f"test:{src}"
            # One file compiled into two binaries is one test file of both.
            rec = tests.setdefault(
                src,
                {"targets": [], "requires": [], "entries": sorted(test_entries[src])},
            )
            rec["targets"].append(tgt)
            rec["requires"] += [r for r in u["requires"] if r not in rec["requires"]]
            kind = "test"
        else:
            owner = node(tgt)
            kind = (
                "module-interface"
                if any(":" not in p for p in u["provides"])
                else "module-partition"
                if u["provides"]
                else "module-implementation"
                if tgt in target_module
                else "source"
            )
        for r in u["requires"]:
            if module_of(r) not in provider:
                unresolved.add(r)
            elif node(provider[module_of(r)]) != owner:
                edges[owner].add(node(provider[module_of(r)]))
        if src:
            add_file(src, kind, owner)
        for h in u["headers"]:
            rel = tree.rel(h, build)
            if rel:
                add_file(rel, "header", owner)

    # A test file also depends on its own binary (helper units) and on the
    # non-module targets the binary links.
    for f, rec in tests.items():
        own = f"test:{f}"
        links = set().union(*(link_closure(t) for t in rec["targets"]))
        edges[own] |= {node(t) for t in rec["targets"]} | {node(t) for t in links}
        rec["links"] = sorted(links)
    for name in targets:
        if name in target_module or targets[name]["sources"]:
            edges[node(name)] |= {node(t) for t in link_closure(name)}

    # Sources and headers the scanner never saw: device code, unscanned C++.
    scanned = {u["object"] for u in units}
    deps_text = ninja_deps(build)
    if deps_text is None:
        notes.append("ninja -t deps unavailable: headers of unscanned units not read")
    for obj, deps in parse_ninja_deps(deps_text or "").items():
        tgt = target_of_object(obj)
        if obj in scanned or tgt not in targets:
            continue
        owner = node(tgt)
        if deps:
            src = tree.rel(deps[0], build)
            if src:
                add_file(src, "source", owner)
        for h in deps[1:]:
            rel = tree.rel(h, build)
            if rel:
                add_file(rel, "header", owner)
    # Every compiled source the codemodel names, so a .cu is placed even when
    # its depfile could not be read.
    for name, t in targets.items():
        for s in t["sources"]:
            rel = tree.rel(s)
            if rel and rel not in files and not t["abstract"]:
                add_file(rel, "source", node(name))

    # Distances: from each test file's direct dependencies (distance 0),
    # forward along the edges. A test file is distance 0 from itself.
    reach: dict[str, dict[str, int]] = {}
    for f in tests:
        dist = {f"test:{f}": 0}
        queue = deque()
        for d in edges[f"test:{f}"]:
            if d not in dist:
                dist[d] = 0
                queue.append(d)
        while queue:
            n = queue.popleft()
            for d in edges.get(n, ()):
                if d not in dist:
                    dist[d] = dist[n] + 1
                    queue.append(d)
        reach[f] = dist

    file_map = {}
    for path in sorted(files):
        rec = files[path]
        owners = sorted(rec["owners"])
        hit = {}
        for f, dist in reach.items():
            ds = [dist[o] for o in owners if o in dist]
            if ds:
                hit[f] = min(ds)
        file_map[path] = {
            "kinds": sorted(rec["kinds"]),
            "owners": owners,
            "tests": dict(sorted(hit.items())),
        }

    owners_out = {}
    for name in sorted(set(edges) | {node(t) for t in targets}):
        if name.startswith("test:"):
            continue
        is_module = name in provider and name in target_module.values()
        owners_out[name] = {
            "kind": "module" if is_module else "target",
            "target": provider[name] if is_module else name,
            "depends_on": sorted(edges.get(name, ())),
        }

    configured = {
        d["source"] for d in codemodel["configurations"][0].get("directories", [])
    } or None
    unmapped = []
    for path in walk_roots(tree.source, roots, build):
        if path not in files:
            cat = classify_unmapped(path, tree.source, configured)
            unmapped.append({"path": path, "category": cat})
    return {
        "schema": SCHEMA,
        "source_dir": str(tree.source),
        "build_dir": str(build),
        "files": file_map,
        "owners": owners_out,
        "tests": {f: tests[f] for f in sorted(tests)},
        "unmapped": {
            "files": unmapped,
            "entries": unmapped_entries,
            "modules": sorted(unresolved),
        },
        "notes": notes,
    }


# --- the report --------------------------------------------------------------


def report(result: dict, out=sys.stderr) -> None:
    files = result["unmapped"]["files"]
    by_cat: dict[str, list[str]] = defaultdict(list)
    for f in files:
        by_cat[f["category"]].append(f["path"])
    tests = result["tests"]
    print(
        f"test_map: {len(result['files'])} files, {len(result['owners'])} owners, "
        f"{len(tests)} test files, "
        f"{sum(len(t['entries']) for t in tests.values())} ctest entries",
        file=out,
    )
    reasons = {**RUN_EVERYTHING, **NOT_CPP, "other": "NOT PLACED: runs everything"}
    for cat in ("other", *RUN_EVERYTHING, *NOT_CPP):
        paths = by_cat.get(cat, [])
        print(f"unmapped {cat}: {len(paths)}  ({reasons[cat]})", file=out)
        if cat == "other":
            for p in paths:
                print(f"  {p}", file=out)
    for key in ("entries", "modules"):
        items = result["unmapped"][key]
        print(f"unmapped {key}: {len(items)}", file=out)
        for item in items:
            print(f"  {item['name'] if isinstance(item, dict) else item}", file=out)
    for n in result["notes"]:
        print(f"note: {n}", file=out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="devtools/test_map.py",
        description="Write the test-dependency map of a built build directory.",
    )
    ap.add_argument("build_dir", type=Path, help="a configured and built build dir")
    ap.add_argument(
        "-o", "--output", type=Path, help="write the map here (default: stdout)"
    )
    ap.add_argument(
        "--root",
        action="append",
        dest="roots",
        help="tree to report unplaced files under (repeatable; default: src, test)",
    )
    ap.add_argument(
        "--ctest-json",
        type=Path,
        help="a saved `ctest --show-only=json-v1` instead of running ctest",
    )
    ap.add_argument(
        "--file", action="append", default=[], help="print one file's entry and exit"
    )
    args = ap.parse_args(argv)
    build = args.build_dir.resolve()
    if not (build / "CMakeCache.txt").exists():
        print(f"test_map: {build} is not a configured build dir", file=sys.stderr)
        return 2
    ensure_query(build)
    if load_codemodel(build) is None:
        cmake = shutil.which("cmake")
        if cmake is None:
            print(
                "test_map: no codemodel reply and no cmake to make one", file=sys.stderr
            )
            return 2
        subprocess.run([cmake, str(build)], check=True, stdout=subprocess.DEVNULL)
    listing = None
    if args.ctest_json:
        text = args.ctest_json.read_text()
        listing = json.loads(text[text.index("{") :])
    result = extract(build, roots=args.roots or DEFAULT_ROOTS, ctest_listing=listing)
    if args.file:
        for f in args.file:
            entry = result["files"].get(f)
            print(json.dumps({f: entry, "test": result["tests"].get(f)}, indent=2))
        return 0
    text = json.dumps(result, separators=(",", ":")) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        sys.stdout.write(text)
    report(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
