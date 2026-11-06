"""devtools/test_map.py, the test-dependency map (#983), over a toy build tree
generated in tmp_path: P1689 `.ddi` files, their `.ddi.d` depfiles, a File
API codemodel reply, a ctest listing and gtest listings. Nothing here builds
or needs a GPU, and nothing reads this checkout's git history.

The toy tree:

    toy.util   module   src/util/util.cppm
    toy.base   module   src/base/base.cppm   includes src/base/shared.h
                        src/base/base.cpp    implementation unit; imports toy.util
                        links toy.kern.device, and toy.base.headers (an
                        INTERFACE library that links toy.extra.device)
    toy.top    module   src/top/top.cppm     imports toy.base
                        src/top/part.cppm    partition toy.top:part
    toy.kern.device     src/kern/kern.cu     includes src/kern/kern.h (ninja)
    toy.extra.device    src/extra/extra.cu
    toy_tests  binary   test/a_test.cpp  imports toy.top
                        test/b_test.cpp  imports toy.base; includes test/helpers.h
                        test/c_test.cpp  imports std only
                        test/main.cpp    no cases
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _ROOT / "devtools" / "test_map.py"
_spec = importlib.util.spec_from_file_location("devtools_test_map", _TOOL)
tm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tm)

STD = "/usr/lib/llvm/share/libc++/v1/std.cppm"

# target -> (type, directory, [(source, provides, requires, headers)], links)
TARGETS = {
    "__cmake_cxx23": ("OBJECT_LIBRARY", "", [(STD, ["std"], [], [])], []),
    "toy.util": (
        "STATIC_LIBRARY",
        "src/util",
        [("src/util/util.cppm", ["toy.util"], ["std"], [])],
        [],
    ),
    "toy.base": (
        "STATIC_LIBRARY",
        "src/base",
        [
            ("src/base/base.cppm", ["toy.base"], ["std"], ["src/base/shared.h"]),
            ("src/base/base.cpp", [], ["std", "toy.base", "toy.util"], []),
        ],
        ["toy.util", "toy.kern.device", "toy.base.headers"],
    ),
    "toy.base.headers": ("INTERFACE_LIBRARY", "src/base", [], ["toy.extra.device"]),
    "toy.top": (
        "STATIC_LIBRARY",
        "src/top",
        [
            ("src/top/top.cppm", ["toy.top"], ["toy.base", "toy.top:part"], []),
            ("src/top/part.cppm", ["toy.top:part"], ["std"], []),
        ],
        ["toy.base"],
    ),
    "toy.kern.device": ("STATIC_LIBRARY", "src/kern", [("src/kern/kern.cu",)], []),
    "toy.extra.device": ("STATIC_LIBRARY", "src/extra", [("src/extra/extra.cu",)], []),
    "toy_tests": (
        "EXECUTABLE",
        "test",
        [
            ("test/a_test.cpp", [], ["std", "toy.top"], []),
            ("test/b_test.cpp", [], ["toy.base"], ["test/helpers.h", STD]),
            ("test/c_test.cpp", [], ["std"], []),
            ("test/main.cpp", [], [], []),
        ],
        ["toy.top", "toy.base", "GTest::gtest"],
    ),
}
CASES = {
    "test/a_test.cpp": ["A.One", "A.Two"],
    "test/b_test.cpp": ["P/B.Param/0", "P/B.Param/1"],
    "test/c_test.cpp": ["C.Only"],
}
UNPLACED = [
    "src/README.md",
    "src/CMakeLists.txt",
    "src/orphan.h",
    "src/kern/.clang-format",
    "test/data/x.json",
    "test/experimental/CMakeLists.txt",
    "test/experimental/x_gpu.cpp",
]
# The codemodel's configured directories; test/experimental/ is not one.
CONFIGURED = [".", "src", "test"]


def _obj(directory: str, target: str, source: str) -> str:
    stem = Path(source).name
    return f"{directory}/CMakeFiles/{target}.dir/{stem}.o".lstrip("/")


def make_tree(root: Path) -> tuple[Path, dict, str]:
    """Write the toy source and build trees; return the build dir, the ctest
    listing and the `ninja -t deps` text."""
    src, build = root / "repo", root / "repo" / "build"
    reply = build / ".cmake/api/v1/reply"
    reply.mkdir(parents=True)
    (build / "CMakeCache.txt").write_text("")
    entries, ninja = [], []
    for name, (ttype, directory, units, links) in TARGETS.items():
        sources = []
        for unit in units:
            source = unit[0]
            path = Path(source) if source.startswith("/") else src / source
            if not source.startswith("/"):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("")
            sources.append({"path": source})
            obj = _obj(directory, name, source)
            if len(unit) == 1:  # device code: no scan, headers from ninja
                ninja.append(
                    f"{obj}: #deps 2, deps mtime 1 (VALID)\n    {path}\n"
                    + "".join(
                        f"    {src / h}\n" for h in ["src/kern/kern.h"] if "kern" in obj
                    )
                )
                continue
            _, provides, requires, headers = unit
            rule = {
                "primary-output": obj,
                "requires": [{"logical-name": r} for r in requires],
            }
            if provides:
                rule["provides"] = [
                    {"is-interface": True, "logical-name": p, "source-path": str(path)}
                    for p in provides
                ]
            ddi = build / f"{obj}.ddi"
            ddi.parent.mkdir(parents=True, exist_ok=True)
            ddi.write_text(json.dumps({"revision": 0, "rules": [rule], "version": 1}))
            heads = [h if h.startswith("/") else str(src / h) for h in headers]
            for h in heads:
                if not h.startswith("/usr"):
                    Path(h).parent.mkdir(parents=True, exist_ok=True)
                    Path(h).write_text("")
            ddi.with_name(ddi.name + ".d").write_text(
                f"{obj}.ddi: \\\n  " + " \\\n  ".join([str(path), *heads]) + "\n"
            )
        doc = {
            "name": name,
            "type": ttype,
            "sources": sources,
            "linkLibraries": [{"id": f"{x}::@1"} for x in links if x in TARGETS],
        }
        if ttype == "INTERFACE_LIBRARY":
            doc["interfaceLinkLibraries"] = doc.pop("linkLibraries")
        if ttype == "EXECUTABLE":
            doc["artifacts"] = [{"path": f"{directory}/{name}"}]
        (reply / f"target-{name}.json").write_text(json.dumps(doc))
        entries.append(
            {"id": f"{name}::@1", "name": name, "jsonFile": f"target-{name}.json"}
        )
    (reply / "codemodel-v2-1.json").write_text(
        json.dumps(
            {
                "paths": {"source": str(src), "build": str(build)},
                "configurations": [
                    {
                        "targets": entries,
                        "abstractTargets": [],
                        "directories": [{"source": d} for d in CONFIGURED],
                    }
                ],
            }
        )
    )
    (reply / "index-1.json").write_text(
        json.dumps(
            {"objects": [{"kind": "codemodel", "jsonFile": "codemodel-v2-1.json"}]}
        )
    )
    for p in UNPLACED:
        (src / p).parent.mkdir(parents=True, exist_ok=True)
        (src / p).write_text("")
    exe = str(build / "test" / "toy_tests")
    tests = [
        {"name": case, "command": [exe, f"--gtest_filter={case}"]}
        for cases in CASES.values()
        for case in cases
    ]
    tests.append({"name": "example_rhf", "command": ["/bin/sh", "run.sh"]})
    return build, {"kind": "ctestInfo", "tests": tests}, "\n".join(ninja)


def gtest_listing(src: Path) -> dict:
    suites: dict[str, list] = {}
    for f, cases in CASES.items():
        for case in cases:
            suite, name = case.split(".", 1)
            suites.setdefault(suite, []).append({"name": name, "file": str(src / f)})
    return {"testsuites": [{"name": s, "testsuite": c} for s, c in suites.items()]}


@pytest.fixture
def toy(tmp_path: Path) -> dict:
    build, listing, ninja = make_tree(tmp_path)
    return tm.extract(
        build,
        ctest_listing=listing,
        gtest_listing=lambda exe: gtest_listing(build.parent),
        ninja_deps=lambda b: ninja,
    )


A, B, C = "test/a_test.cpp", "test/b_test.cpp", "test/c_test.cpp"


# --- the four routes the issue names ------------------------------------------


def test_interface_change_reaches_importers_by_distance(toy: dict) -> None:
    entry = toy["files"]["src/base/base.cppm"]
    assert entry["owners"] == ["toy.base"]
    assert entry["kinds"] == ["module-interface"]
    assert entry["tests"] == {A: 1, B: 0}


def test_implementation_unit_import_belongs_to_its_module(toy: dict) -> None:
    # toy.util is imported only by base.cpp, never by an interface.
    assert toy["files"]["src/base/base.cpp"]["kinds"] == ["module-implementation"]
    assert toy["files"]["src/base/base.cpp"]["owners"] == ["toy.base"]
    assert "toy.util" in toy["owners"]["toy.base"]["depends_on"]
    assert toy["files"]["src/util/util.cppm"]["tests"] == {A: 2, B: 1}


def test_header_takes_its_includers_owners(toy: dict) -> None:
    assert toy["files"]["src/base/shared.h"] == {
        "kinds": ["header"],
        "owners": ["toy.base"],
        "tests": {A: 1, B: 0},
    }
    # A header a test file includes belongs to that test file alone.
    assert toy["files"]["test/helpers.h"]["owners"] == [f"test:{B}"]
    assert toy["files"]["test/helpers.h"]["tests"] == {B: 0}


def test_cu_reached_only_through_linking(toy: dict) -> None:
    kern = toy["files"]["src/kern/kern.cu"]
    assert kern["owners"] == ["toy.kern.device"]
    assert kern["tests"] == {A: 2, B: 1}
    assert toy["owners"]["toy.kern.device"]["kind"] == "target"
    # Its header comes from ninja's deps, not from a scan.
    assert toy["files"]["src/kern/kern.h"]["tests"] == {A: 2, B: 1}
    # Through an INTERFACE library, which compiles nothing of its own.
    assert toy["files"]["src/extra/extra.cu"]["tests"] == {A: 2, B: 1}
    assert "toy.base.headers" not in toy["owners"]["toy.base"]["depends_on"]


# --- the rest of the map ------------------------------------------------------


def test_partition_belongs_to_its_module(toy: dict) -> None:
    part = toy["files"]["src/top/part.cppm"]
    assert part["kinds"] == ["module-partition"]
    assert part["owners"] == ["toy.top"]
    assert part["tests"] == {A: 0}


def test_test_file_records_requires_and_entries(toy: dict) -> None:
    assert toy["tests"][A]["targets"] == ["toy_tests"]
    assert toy["tests"][A]["requires"] == ["std", "toy.top"]
    assert toy["tests"][A]["entries"] == ["A.One", "A.Two"]
    assert toy["tests"][B]["entries"] == ["P/B.Param/0", "P/B.Param/1"]
    # A changed test file selects itself, and only itself.
    assert toy["files"][A]["tests"] == {A: 0}
    # Importing only std, it is reached by itself and its binary's helper.
    assert [p for p, f in toy["files"].items() if C in f["tests"]] == [
        C,
        "test/main.cpp",
    ]


def test_helper_unit_reaches_every_test_in_its_binary(toy: dict) -> None:
    assert toy["files"]["test/main.cpp"]["owners"] == ["toy_tests"]
    assert toy["files"]["test/main.cpp"]["tests"] == {A: 0, B: 0, C: 0}


def test_unplaced_files_are_reported_never_dropped(toy: dict) -> None:
    got = {f["path"]: f["category"] for f in toy["unmapped"]["files"]}
    assert got == {
        "src/README.md": "doc",
        "src/CMakeLists.txt": "cmake",
        "src/orphan.h": "other",
        "src/kern/.clang-format": "config",
        "test/data/x.json": "data",
        "test/experimental/CMakeLists.txt": "unconfigured",
        "test/experimental/x_gpu.cpp": "unconfigured",
    }
    assert toy["unmapped"]["entries"] == [
        {"name": "example_rhf", "reason": "not one gtest case of a built target"}
    ]
    assert toy["unmapped"]["modules"] == []
    assert not any(p.startswith(("build/", "/")) for p in toy["files"])


def test_unresolved_module_is_reported(tmp_path: Path) -> None:
    build, listing, ninja = make_tree(tmp_path)
    ddi = build / _obj("test", "toy_tests", "test/c_test.cpp")
    path = Path(f"{ddi}.ddi")
    doc = json.loads(path.read_text())
    doc["rules"][0]["requires"].append({"logical-name": "toy.gone"})
    path.write_text(json.dumps(doc))
    result = tm.extract(
        build,
        ctest_listing=listing,
        gtest_listing=lambda exe: gtest_listing(build.parent),
        ninja_deps=lambda b: ninja,
    )
    assert result["unmapped"]["modules"] == ["toy.gone"]


# --- the readers --------------------------------------------------------------


def test_parse_depfile() -> None:
    text = "out.o.ddi: \\\n  /a/b.cpp /a/with\\ space.h \\\n  /c.h\nphony:\n"
    assert tm.parse_depfile(text) == ["/a/b.cpp", "/a/with space.h", "/c.h"]


def test_parse_ninja_deps() -> None:
    text = (
        "x/CMakeFiles/t.dir/k.cu.o: #deps 2, deps mtime 1 (VALID)\n"
        "    /s/k.cu\n    /s/k.h\n\n"
    )
    assert tm.parse_ninja_deps(text) == {
        "x/CMakeFiles/t.dir/k.cu.o": ["/s/k.cu", "/s/k.h"]
    }


def test_target_of_object() -> None:
    assert tm.target_of_object("src/a/CMakeFiles/fb.a.dir/a.cpp.o") == "fb.a"
    assert tm.target_of_object("loose.o") is None


# --- the command line ---------------------------------------------------------


def test_imports_only_the_standard_library() -> None:
    tree = ast.parse(_TOOL.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module.split(".")[0])
    assert names - {"__future__"} <= sys.stdlib_module_names, names


def test_cli_runs_the_gtest_binary_under_isolated_python(tmp_path: Path) -> None:
    build, listing, _ = make_tree(tmp_path)
    exe = build / "test" / "toy_tests"
    exe.parent.mkdir(parents=True, exist_ok=True)
    # A stand-in gtest binary: writes the listing where --gtest_output says.
    payload = json.dumps(json.dumps(gtest_listing(build.parent)))
    exe.write_text(
        f"#!{sys.executable}\nimport sys, json\n"
        "out = [a for a in sys.argv if a.startswith('--gtest_output=json:')][0]\n"
        f"open(out.split(':', 1)[1], 'w').write({payload})\n"
    )
    exe.chmod(0o755)
    saved = tmp_path / "ctest.json"
    saved.write_text("ctest chatter\n" + json.dumps(listing))
    out = tmp_path / "map.json"
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "PRE_COMMIT"))
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            str(_TOOL),
            str(build),
            "--ctest-json",
            str(saved),
            "-o",
            str(out),
        ],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text())
    assert result["tests"][A]["entries"] == ["A.One", "A.Two"]
    assert "unmapped other: 1" in proc.stderr
    assert "  src/orphan.h" in proc.stderr
    assert (build / ".cmake/api/v1/query" / tm.QUERY_CLIENT / "query.json").exists()


def test_cli_refuses_an_unconfigured_dir(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, "-I", str(_TOOL), str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "not a configured build dir" in proc.stderr
