"""devtools/alloc-sites.py: the inventory's rows, and what the pool-bypass lint flags.

The inventory is the footprint issues' checklist and the lint the only gate on
docs/style.md "Allocations go through the pool", so these cases pin both
directions: every spelling of an allocation is one row (and a look-alike none),
and every raw call is a finding until a `pool-bypass-ok:` reason exempts it.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[2] / "devtools" / "alloc-sites.py"
_spec = importlib.util.spec_from_file_location("alloc_sites", _SCRIPT)
assert _spec is not None and _spec.loader is not None
sites = importlib.util.module_from_spec(_spec)
sys.modules["alloc_sites"] = sites  # dataclass resolves its module by name
_spec.loader.exec_module(sites)

_DATA = Path(__file__).resolve().parent / "data" / "alloc_sites"


def rows(source: str) -> list[tuple]:
    found = sites.find_sites(source)
    return [(s.line, s.kind, s.type, s.size, s.function) for s in found]


def test_every_spelling_is_one_row() -> None:
    text = (_DATA / "spellings.cpp").read_text(encoding="utf-8")
    assert rows(text) == [
        (12, "Buffers::device", "double", "count", "make_buffer"),
        (17, "Buffers::device", "double", "n * 2", "Holder"),
        (17, "StreamBuffers::pinned", "int32_t", "n", "Holder"),
        (24, "Buffers::device", "double", "n * m", "every_spelling"),
        (25, "Buffers::pinned", "int32_t", "n + 1", "every_spelling"),
        (27, "Buffers::host", "float", "static_cast<size_t>(m)", "every_spelling"),
        (30, "Buffers::device", "double", "n * i", "every_spelling"),
        (33, "Buffers::device", "uint64_t", "m", "every_spelling"),
        (34, "StreamBuffers::device", "double", "n", "every_spelling"),
        (35, "device_workspace", "std::byte", "n * sizeof(double)", "every_spelling"),
        (
            37,
            "device_regions",
            "double, int32_t",
            "static_cast<size_t>(n) * static_cast<size_t>(m) * sizeof(double), "
            "static_cast<size_t>(m)",
            "every_spelling",
        ),
        (41, "counted_pinned", "double", "n * m", "every_spelling"),
    ]


@pytest.mark.parametrize(
    "source",
    [
        "Result<Buffers::device<double>> make(DeviceResources &res, size_t n);",
        "void f(const Buffers::device<double> &b) {}",
        "Buffers::device<double> slab(DeviceResources &res, const Blocks &blocks) {}",
        "void f() { Buffers::device<double> moved = std::move(other); }",
        "void f() { std::vector<std::optional<Buffers::device<double>>> v(n); }",
        "void f() { Buffers::device<double> empty; }",
        "auto device_regions(const Counts... counts) {}",
    ],
)
def test_look_alikes_are_not_sites(source: str) -> None:
    assert rows(source) == []


def test_member_built_in_a_siblings_initializer_list(tmp_path: Path) -> None:
    (tmp_path / "a.cppm").write_text("struct A {\n  Buffers::device<double> x_;\n};\n")
    (tmp_path / "a.cpp").write_text("A::A(Device d, size_t n) : x_(2 * n, d) {}\n")
    found = sites.inventory([str(tmp_path / "a.cpp")])
    assert [(s.line, s.kind, s.size, s.function) for s in found] == [
        (1, "Buffers::device", "2 * n", "A::A")
    ]


def test_own_declaration_shadows_a_siblings(tmp_path: Path) -> None:
    (tmp_path / "a.cppm").write_text("std::vector<Buffers::device<double>> folded;\n")
    (tmp_path / "b.cpp").write_text(
        "void f() {\n  std::optional<Fold> folded;\n  folded.emplace(make());\n}\n"
    )
    assert sites.inventory([str(tmp_path / "b.cpp")]) == []


def test_diff_sites_names_added_and_resized() -> None:
    def site(line: int, size: str, function: str = "f") -> object:
        return sites.Site("x.cpp", line, "Buffers::device", "double", size, function)

    old = [site(3, "n"), site(4, "m")]
    head = [site(3, "n"), site(4, "2 * m"), site(5, "k", "g"), site(9, "n")]
    assert sites.diff_sites(old, head, touched={4, 5, 9}) == [
        ("resized", head[1]),
        ("added", head[2]),
    ]


# -- the check ---------------------------------------------------------------

_FLAGGED = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 25}


def bypass_lines(text: str, rel: str = "src/x/x.cpp", allow=None) -> set[int]:
    return {line for line, _, _ in sites.find_bypasses(text, rel, allow or {})}


def test_bypass_fixture_flags_each_raw_call() -> None:
    text = (_DATA / "bypass.cpp").read_text(encoding="utf-8")
    assert bypass_lines(text) == _FLAGGED


def test_a_reason_exempts_the_line() -> None:
    lines = (_DATA / "bypass.cpp").read_text(encoding="utf-8").split("\n")
    marked = [
        f"{line}  // pool-bypass-ok: the fixture" if i in _FLAGGED else line
        for i, line in enumerate(lines, 1)
    ]
    assert bypass_lines("\n".join(marked)) == set()


@pytest.mark.parametrize(
    ("source", "flagged"),
    [
        ("// pool-bypass-ok: until #1130\nBuffers::pinned<int> k(n);", set()),
        ("Buffers::pinned<int> k(n);  // pool-bypass-ok:", {1}),
        ("f();  // pool-bypass-ok: not comment-only\nBuffers::pinned<int> k(n);", {2}),
    ],
)
def test_marker_needs_a_reason_and_a_comment_only_line(
    source: str, flagged: set[int]
) -> None:
    assert bypass_lines(source) == flagged


def test_core_may_hold_pinned_and_wwr_may_call_the_runtime() -> None:
    pinned = "Buffers::pinned<int32_t> info(1);"
    raw = "wwrMalloc(p, n); wwrMemGetInfo(&f, &t);"
    assert bypass_lines(pinned, "src/core/resources/r.cppm") == set()
    assert bypass_lines(raw, "src/core/resources/r.cppm") == {1}
    assert bypass_lines(raw, "src/core/wwr/w.cppm") == set()


def test_memgetinfo_allowlist_counts_per_file() -> None:
    two = "wwrMemGetInfo(&f, &t);\nwwrMemGetInfo(&f, &t);\n"
    assert bypass_lines(two, allow={"src/x/x.cpp": (2, "#1")}) == set()
    assert bypass_lines(two, allow={"src/x/x.cpp": (1, "#1")}) == {1, 2}
    stale = sites.find_bypasses("int x;", "src/x/x.cpp", {"src/x/x.cpp": (1, "#1")})
    assert [m for _, _, m in stale] == [
        "stale wwrMemGetInfo allowlist entry (1 allowed, 0 found, #1): "
        "lower or delete it"
    ]


def test_allowlist_entries_name_their_owner(tmp_path: Path) -> None:
    good = tmp_path / "good.txt"
    good.write_text("# comment\nsrc/a.cpp  2 #1047\n")
    assert sites.read_allowlist(good) == {"src/a.cpp": (2, "#1047")}
    bad = tmp_path / "bad.txt"
    bad.write_text("src/a.cpp 2\n")
    with pytest.raises(ValueError, match="#<issue>"):
        sites.read_allowlist(bad)
    # The checked-in one parses, and every entry it has names its owner. The
    # "Pipeline memory caps" milestone migrated its last sites (#1186), so it
    # holds comments only: a re-added line fails here.
    checked = sites.read_allowlist()
    assert isinstance(checked, dict)
    for path, (_, owner) in checked.items():
        assert owner.startswith("#"), path
    assert checked == {}


def test_main_check_reports_and_fails(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "k.cpp"
    bad.write_text("void f() { hipMalloc(&p, n); }\n")
    good = tmp_path / "ok.cpp"
    good.write_text("void f() { Buffers::device<double> x(n, dev); }\n")
    allow = tmp_path / "allow.txt"
    allow.write_text("")
    assert sites.check([str(good)], allow) == 0
    assert sites.check([str(bad)], allow) == 1
    assert "k.cpp:1:12: `hipMalloc` bypasses the device pool" in capsys.readouterr().out


def test_main_inventory_json(capsys) -> None:
    assert sites.main(["inventory", "--json", str(_DATA / "spellings.cpp")]) == 0
    assert '"kind": "device_regions"' in capsys.readouterr().out


def test_pinned_spellings_drop_the_callers_name() -> None:
    source = (
        "void f() {\n"
        '  auto a = res.pinned_workspace("f", n * 8);\n'
        '  auto b = res.pinned_regions<double, int32_t>("f", n, m);\n'
        '  auto c = res.counted_pinned<double>("f", n);\n'
        "}\n"
    )
    assert rows(source) == [
        (2, "pinned_workspace", "std::byte", "n * 8", "f"),
        (3, "pinned_regions", "double, int32_t", "n, m", "f"),
        (4, "counted_pinned", "double", "n", "f"),
    ]
