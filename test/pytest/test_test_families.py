"""devtools/test_families.py, the family lister (#984), over toy gtest listings
and toy sources written to tmp_path. Nothing here builds or needs a GPU, and
nothing reads this checkout's git history: the metrics-branch test makes its
own repository."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _ROOT / "devtools" / "test_families.py"
_spec = importlib.util.spec_from_file_location("devtools_test_families", _TOOL)
tf = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = tf  # dataclasses resolve their module by name
_spec.loader.exec_module(tf)

SOURCE = """\
// toy
TEST(Plain, Refuses) {
}

TEST(Plain, Validates) {
  FBF_REQUIRE_GPU();
  FBF_FAMILY_REPRESENTATIVE();
}

TEST(Other, Cheap) {
}

TEST(Other, Dear) {
  // FBF_FAMILY_REPRESENTATIVE(); in a comment is not a marker
}

TEST_P(Param, Matches) {
  FBF_FAMILY_REPRESENTATIVE("big_batch0", "big");
}

TEST_P(Param, Other) {
}
"""
# What gtest's listing reports for each case: its macro's line.


def listing(file: str) -> dict:
    def case(name, line, **extra):
        return {"name": name, "file": file, "line": line, **extra}

    return {
        "testsuites": [
            {"name": "Plain", "testsuite": [case("Refuses", 2), case("Validates", 5)]},
            {"name": "Other", "testsuite": [case("Cheap", 9), case("Dear", 12)]},
            {
                "name": "Cases/Param",
                "testsuite": [
                    case(f"Matches/{p}", 16, value_param=p)
                    for p in (
                        "small_batch0",
                        "small_batch3",
                        "big_batch0",
                        "big_batch3",
                    )
                ]
                + [case("Other/x", 20, value_param="x")],
            },
            {
                "name": "More/Param",
                "testsuite": [case("Matches/small_batch0", 16, value_param="y")],
            },
            {"name": "Typed/0", "testsuite": [case("Works", 30, type_param="int")]},
            {"name": "Typed/1", "testsuite": [case("Works", 30, type_param="float")]},
            {"name": "Plain", "testsuite": [case("DISABLED_Off", 3)]},
        ]
    }


@pytest.fixture
def tree(tmp_path):
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "a_test.cpp").write_text(SOURCE)
    return tmp_path


def run(tree, source=SOURCE, times=None, slow=None):
    (tree / "test" / "a_test.cpp").write_text(source)
    cases = tf.cases_from_listing(listing(str(tree / "test" / "a_test.cpp")), tree)
    markers = tf.markers_in(tree, {c.file for c in cases})
    return tf.families(cases, markers, times, slow)


def test_default_families(tree):
    cases = tf.cases_from_listing(listing(str(tree / "test" / "a_test.cpp")), tree)
    by = {c.name: c for c in cases}
    assert "Plain.DISABLED_Off" not in by
    assert by["Plain.Refuses"].default_family == "Plain"
    assert by["Plain.Refuses"].file == "test/a_test.cpp"
    # Every instantiation prefix and every param is one TEST_P's family.
    assert by["Cases/Param.Matches/big_batch0"].default_family == "Param.Matches"
    assert by["More/Param.Matches/small_batch0"].default_family == "Param.Matches"
    assert by["More/Param.Matches/small_batch0"].prefix == "More"
    assert by["Cases/Param.Other/x"].default_family == "Param.Other"
    # A typed test is one family across its types, as a TEST_P across params.
    assert by["Typed/1.Works"].default_family == "Typed.Works"
    assert by["Typed/1.Works"].param == "1"


def test_marked_split_cheapest_and_flagged(tree):
    times = {
        "Other.Cheap": [0.5],
        "Other.Dear": [3.0],
        "Cases/Param.Matches/small_batch3": [0.2, 0.4, 0.3],
        "Cases/Param.Matches/small_batch0": [0.35],
        "Cases/Param.Matches/big_batch3": [0.01],  # cheaper, but the marker wins
    }
    fams, errors, notes = run(tree, times=times)
    assert errors == []
    assert fams["Plain"].representative == "Plain.Validates"
    assert fams["Plain"].by == "marked"
    assert fams["Other"].representative == "Other.Cheap"
    assert fams["Other"].by == "cheapest"
    assert fams["Other"].seconds == 0.5
    split = fams["Param.Matches[big]"]
    assert split.representative == "Cases/Param.Matches/big_batch0"
    assert split.by == "marked"
    assert sorted(split.members) == [
        "Cases/Param.Matches/big_batch0",
        "Cases/Param.Matches/big_batch3",
    ]
    # The rest of the TEST_P: the median 0.3 s beats 0.35 s.
    rest = fams["Param.Matches"]
    assert rest.representative == "Cases/Param.Matches/small_batch3"
    assert rest.seconds == 0.3
    assert "More/Param.Matches/small_batch0" in rest.members
    # No marker and no time: flagged, not guessed.
    assert fams["Param.Other"].by is None
    assert fams["Typed.Works"].by is None
    # The marked big_batch0 has no time where big_batch3 does.
    assert any("big_batch0" in n for n in notes)


def test_no_times_flags_every_unmarked_family(tree):
    fams, errors, _ = run(tree)
    assert errors == []
    flagged = sorted(f.name for f in fams.values() if f.by is None)
    assert flagged == ["Other", "Param.Matches", "Param.Other", "Typed.Works"]


def test_a_gated_cheapest_member_is_passed_over(tree):
    # #1125: the nightly's times are unskipped, so the cheapest member can be
    # one FBF_SKIP_IF_FAST_TESTS() gates; the fast set would then run nothing.
    times = {
        "Other.Cheap": [0.5],
        "Other.Dear": [3.0],
        "Cases/Param.Matches/small_batch3": [0.1],
        "Cases/Param.Matches/small_batch0": [0.35],
        "More/Param.Matches/small_batch0": [0.9],
    }
    slow = {
        "Other.Cheap": True,
        "Other.Dear": False,
        # A TEST_P gated by its param: one instance gated, the next not.
        "Cases/Param.Matches/small_batch3": True,
        "Cases/Param.Matches/small_batch0": False,
        # No record says More/... either way: it counts as fast.
    }
    fams, errors, _ = run(tree, times=times, slow=slow)
    assert errors == []
    assert fams["Other"].representative == "Other.Dear"
    assert fams["Other"].by == "cheapest"
    assert fams["Other"].seconds == 3.0
    rest = fams["Param.Matches"]
    assert rest.representative == "Cases/Param.Matches/small_batch0"
    assert rest.seconds == 0.35
    # A marked representative is the marker's, gated or not: the gate's
    # `representative(s) skipped` line is what catches that.
    assert fams["Plain"].representative == "Plain.Validates"


def test_a_family_with_no_fast_member_is_an_error(tree):
    times = {"Other.Cheap": [0.5], "Other.Dear": [3.0]}
    slow = {"Other.Cheap": True, "Other.Dear": True}
    fams, errors, _ = run(tree, times=times, slow=slow)
    assert fams["Other"].representative is None
    assert len(errors) == 1
    assert errors[0].startswith("Other: every timed member reaches")
    assert "FBF_FAMILY_REPRESENTATIVE" in errors[0]
    assert "FBF_FAMILY_EXEMPT" in errors[0]
    # A marker, or an exemption, settles it: the lister takes the marker
    # as given (here on a gated case, which the gate would report skipped).
    for fix in ("FBF_FAMILY_REPRESENTATIVE();", 'FBF_FAMILY_EXEMPT("why");'):
        source = SOURCE.replace("TEST(Other, Dear) {\n", "TEST(Other, Dear) {\n" + fix)
        assert run(tree, source, times, slow)[1] == [], fix


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            SOURCE.replace("// toy", "FBF_FAMILY_REPRESENTATIVE();"),
            "not inside any listed test",
        ),
        (SOURCE.replace('"big_batch0", "big"', "kName"), "string literals only"),
        (
            SOURCE.replace(
                'FBF_FAMILY_REPRESENTATIVE("big_batch0", "big")',
                "FBF_FAMILY_REPRESENTATIVE()",
            ),
            "names the instance",
        ),
        (
            SOURCE.replace('"big_batch0", "big"', '"huge_batch0"'),
            "matches no instance",
        ),
        (SOURCE.replace('"big_batch0", "big"', '"small_batch0"'), "is ambiguous"),
        (
            SOURCE.replace('"big_batch0", "big"', '"big_batch0", "small"'),
            "does not match",
        ),
        (
            SOURCE.replace("FBF_REQUIRE_GPU();", 'FBF_FAMILY_REPRESENTATIVE("x");'),
            "takes no arguments",
        ),
        (
            SOURCE.replace(
                "TEST(Plain, Refuses) {\n",
                "TEST(Plain, Refuses) {\n  FBF_FAMILY_REPRESENTATIVE();\n",
            ),
            "already has a representative",
        ),
    ],
)
def test_marker_errors(tree, source, message):
    _, errors, _ = run(tree, source)
    assert any(message in e for e in errors), errors


# Other's two cases, one exempting the family; the same line count as SOURCE.
EXEMPT_SOURCE = SOURCE.replace(
    "TEST(Other, Cheap) {\n}",
    'TEST(Other, Cheap) {\n  FBF_FAMILY_EXEMPT("a timing"); }',
)


def test_an_exempt_family_is_not_flagged(tree):
    # A time does not make it a cheapest family: the exemption wins.
    fams, errors, _ = run(tree, EXEMPT_SOURCE, times={"Other.Cheap": [0.1]})
    assert errors == []
    assert fams["Other"].by == "exempt"
    assert fams["Other"].reason == "a timing"
    assert fams["Other"].representative is None
    assert "1 exempt, 3 flagged" in tf.summary(fams, "none")


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (
            EXEMPT_SOURCE.replace('"a timing"', ""),
            "takes one string literal, the reason",
        ),
        (
            EXEMPT_SOURCE.replace('"a timing"', '"a", "b"'),
            "takes one string literal, the reason",
        ),
        (EXEMPT_SOURCE.replace('"a timing"', '" "'), "takes one string literal"),
        (EXEMPT_SOURCE.replace('"a timing"', "kWhy"), "string literals only"),
        (
            EXEMPT_SOURCE.replace(
                "  // FBF_FAMILY_REPRESENTATIVE(); in a comment is not a marker",
                "  FBF_FAMILY_REPRESENTATIVE();",
            ),
            "is exempt and also has a marked representative",
        ),
        (
            EXEMPT_SOURCE.replace(
                "  // FBF_FAMILY_REPRESENTATIVE(); in a comment is not a marker",
                '  FBF_FAMILY_EXEMPT("again");',
            ),
            "is already exempt",
        ),
    ],
)
def test_exempt_errors(tree, source, message):
    _, errors, _ = run(tree, source)
    assert any(message in e for e in errors), errors


def test_a_qualified_param_settles_an_ambiguity(tree):
    source = SOURCE.replace('"big_batch0", "big"', '"More/small_batch0"')
    fams, errors, _ = run(tree, source)
    assert errors == []
    assert fams["Param.Matches"].representative == "More/Param.Matches/small_batch0"


def test_markers_skip_comments_and_defines(tmp_path):
    (tmp_path / "h.h").write_text(
        "#define FBF_FAMILY_REPRESENTATIVE(...) static_assert(true)\n"
        '/// FBF_FAMILY_REPRESENTATIVE("doc")\n'
        '  FBF_FAMILY_REPRESENTATIVE("p", "r(e)?"); // trailing\n'
        '  FBF_FAMILY_EXEMPT("why (#1)");\n'
    )
    got = tf.markers_in(tmp_path, {"h.h"})
    assert [(m.line, m.args, m.macro) for m in got] == [
        (3, ["p", "r(e)?"], tf.MARKER),
        (4, ["why (#1)"], tf.EXEMPT),
    ]


def test_parse_log():
    text = (
        "fb:   [gpu] [       OK ] Cases/Foo.Bar/x (1234 ms)\n"
        "[  SKIPPED ] Foo.Slow (0 ms)\n"
        "[       OK ] Foo.Fast (0 ms)\n"
        "  3/342 Test   #12: Host.Case ..........   Passed    0.25 sec\n"
        "  4/342 Test   #13: Host.Other .........***Skipped   0.01 sec\n"
    )
    assert tf.parse_log(text) == {
        "Cases/Foo.Bar/x": [1.234],
        "Foo.Fast": [0.0],
        "Host.Case": [0.25],
    }


def record(host="box", kind="nightly", schema=1, cases=()):
    return json.dumps(
        {"schema": schema, "kind": kind, "host": host, "cases": list(cases)}
    )


def test_parse_metrics_filters_host_kind_and_status():
    ok = {"name": "A.b", "status": "pass", "seconds": 2.0}
    lines = [
        record(cases=[ok, {"name": "A.c", "status": "skip", "seconds": 0.0}]),
        record(
            kind="weekly", cases=[{"name": "A.b", "status": "pass", "seconds": 4.0}]
        ),
        record(host="vm", cases=[{"name": "A.b", "status": "pass", "seconds": 9.0}]),
        record(
            kind="shadow", cases=[{"name": "A.b", "status": "pass", "seconds": 9.0}]
        ),
        record(schema=2, cases=[{"name": "A.b", "status": "pass", "seconds": 9.0}]),
        "not json",
        "",
    ]
    assert tf.parse_metrics("\n".join(lines), "box") == {"A.b": [2.0, 4.0]}


def test_parse_slow_takes_the_latest_record_on_this_host():
    def case(name, slow=None):
        c = {"name": name, "status": "pass", "seconds": 1.0}
        return c if slow is None else dict(c, slow=slow)

    lines = [
        record(cases=[case("A.gated", True), case("A.ungated", True)]),
        record(cases=[case("A.ungated", False), case("A.unknown")]),
        record(host="vm", cases=[case("A.gated", False)]),
        record(kind="shadow", cases=[case("A.gated", False)]),
        "not json",
    ]
    assert tf.parse_slow("\n".join(lines), "box") == {
        "A.gated": True,
        "A.ungated": False,
    }


def test_metrics_branch(tmp_path, monkeypatch):
    # The pre-push hook exports GIT_DIR; a toy repository must not see it.
    for k in [k for k in os.environ if k.startswith("GIT_")]:
        monkeypatch.delenv(k)
    for k, v in {
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "t",
        "GIT_AUTHOR_EMAIL": "t@t",
        "GIT_COMMITTER_NAME": "t",
        "GIT_COMMITTER_EMAIL": "t@t",
    }.items():
        monkeypatch.setenv(k, v)

    def git(*args):
        subprocess.run(
            ["git", "-C", str(tmp_path), *args], check=True, capture_output=True
        )

    git("init", "-q", "-b", "main")
    assert tf.metrics_branch_text(tmp_path) is None
    git("checkout", "-q", "--orphan", "metrics")
    (tmp_path / "runs").mkdir()
    (tmp_path / "runs" / "2026-10.jsonl").write_text(record() + "\n")
    (tmp_path / "runs" / "2026-11.jsonl").write_text(record(host="x") + "\n")
    git("add", "runs")
    git("commit", "-q", "-m", "m")
    text = tf.metrics_branch_text(tmp_path)
    assert text.splitlines() == [record(), record(host="x")]


def test_cli(tree, tmp_path):
    saved = tmp_path / "list.json"
    saved.write_text(json.dumps(listing(str(tree / "test" / "a_test.cpp"))))
    log = tmp_path / "gate.log"
    log.write_text(
        "[       OK ] Other.Dear (40 ms)\n[       OK ] Other.Cheap (900 ms)\n"
    )
    cmd = [
        sys.executable,
        str(_TOOL),
        "--gtest-json",
        str(saved),
        "--source",
        str(tree),
        "--no-metrics",
        "--log",
        str(log),
    ]
    proc = subprocess.run([*cmd, "--json"], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    doc = json.loads(proc.stdout)
    assert doc["families"]["Other"]["representative"] == "Other.Dear"
    assert doc["flagged"] == ["Param.Matches", "Param.Other", "Typed.Works"]
    assert "2 marked, 1 cheapest by time, 0 exempt, 3 flagged" in proc.stderr
    text = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert "Plain  [2]  ->  Plain.Validates  marked" in text.stdout
    # A marker that cannot be placed fails the run.
    (tree / "test" / "a_test.cpp").write_text(
        SOURCE.replace("// toy", "FBF_FAMILY_REPRESENTATIVE();")
    )
    bad = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert bad.returncode == 1
    assert "not inside any listed test" in bad.stderr


def test_cli_reads_slow_from_the_metrics(tree, tmp_path):
    saved = tmp_path / "list.json"
    saved.write_text(json.dumps(listing(str(tree / "test" / "a_test.cpp"))))

    def lister(*slow):
        cases = [
            {"name": n, "status": "pass", "seconds": t, "slow": g}
            for (n, t), g in zip(
                (("Other.Cheap", 0.5), ("Other.Dear", 3.0)), slow, strict=True
            )
        ]
        metrics = tmp_path / "runs.jsonl"
        metrics.write_text(record(cases=cases) + "\n")
        cmd = [sys.executable, str(_TOOL), "--gtest-json", str(saved)]
        cmd += ["--source", str(tree), "--metrics", str(metrics), "--host", "box"]
        return subprocess.run(
            [*cmd, "--json"], capture_output=True, text=True, check=False
        )

    one = lister(True, False)
    assert one.returncode == 0, one.stderr
    assert json.loads(one.stdout)["families"]["Other"]["representative"] == (
        "Other.Dear"
    )
    both = lister(True, True)
    assert both.returncode == 1
    assert json.loads(both.stdout)["errors"][0].startswith("Other: every timed")
    assert "error: Other: every timed member" in both.stderr


GPU_SOURCE = """\
#include "helpers.h"

struct Guarded : ::testing::Test {
  void SetUp() override { FBF_REQUIRE_GPU(); }
};

struct Derived : public Guarded {};

struct Unguarded : ::testing::Test {
  void SetUp() override {}
};

void inner() { FBF_REQUIRE_GPU(); }
void outer(int x) { inner(); }

template <typename T>
void templated(T t) {
  outer(0);
}

TEST(Plain, Direct) {
  FBF_REQUIRE_GPU();
}

TEST(Plain, HostClaim) {
  EXPECT_EQ(1, 1);
}

TEST_F(Guarded, InSetUp) {
}

TEST_F(Derived, ThroughItsBase) {
}

TEST_F(Unguarded, NoSetUpGuard) {
}

TEST(Plain, ThroughTwoCalls) {
  outer(1);
}

TEST(Plain, ThroughATemplate) {
  templated<double>(1.0);
}

TEST(Plain, ThroughALambda) {
  const auto run = [&](int n) {
    FBF_REQUIRE_GPU();
  };
  run(1);
}

TEST(Plain, ThroughAHeader) {
  from_header();
}

TEST(Plain, OnlyInProse) {
  // FBF_REQUIRE_GPU();
  const char *s = "FBF_REQUIRE_GPU()";
}

TEST(Guarded, APlainTestNamedLikeTheFixture) {
}

TEST_P(Param, Shared) {
}
"""
HELPERS = "#pragma once\ninline void from_header() { FBF_REQUIRE_GPU(); }\n"


def gpu_tree(tmp_path: Path, binary: str = "free_breakfast_tests_gpu"):
    """GPU_SOURCE and its header in tmp_path, and its cases as gtest lists
    them: each at its macro's line, the TEST_P with three params."""
    src = tmp_path / "test" / "a_gpu.cpp"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(GPU_SOURCE)
    (src.parent / "helpers.h").write_text(HELPERS)
    suites: dict[str, list[dict]] = {}
    for n, line in enumerate(GPU_SOURCE.splitlines(), 1):
        if not line.startswith("TEST"):
            continue
        macro, args = line.split("(", 1)
        suite, test = (a.strip() for a in args.split(")")[0].split(","))
        if macro == "TEST_P":
            for p in ("a", "b", "c"):
                entry = {"name": f"{test}/{p}", "file": str(src), "line": n}
                suites.setdefault(f"Cases/{suite}", []).append(
                    entry | {"value_param": p}
                )
        else:
            suites.setdefault(suite, []).append(
                {"name": test, "file": str(src), "line": n}
            )
    listing = {"testsuites": [{"name": s, "testsuite": c} for s, c in suites.items()]}
    return tf.cases_from_listing(listing, tmp_path, binary), listing


def test_unguarded_gpu_cases(tmp_path):
    cases, _ = gpu_tree(tmp_path)
    got = [c.test for c in tf.unguarded(cases, tmp_path)]
    assert got == [  # in line order
        "Plain.HostClaim",
        # Guarded's SetUp guards; this one's own SetUp is not a call to it.
        "Unguarded.NoSetUpGuard",
        "Plain.OnlyInProse",  # a comment and a string literal are not a call
        "Guarded.APlainTestNamedLikeTheFixture",  # a TEST's suite is no class
        "Param.Shared",  # once, for its three instances
    ]


def test_a_host_binary_is_never_unguarded(tmp_path):
    cases, _ = gpu_tree(tmp_path, binary="free_breakfast_tests_host")
    assert tf.unguarded(cases, tmp_path) == []


def test_code_only_keeps_offsets():
    text = 'a // FBF_REQUIRE_GPU()\nb "x(" /* y\n z */ R"d(q)d" \'(\' c\n'
    code = tf.code_only(text)
    assert len(code) == len(text)
    assert code.count("\n") == text.count("\n")
    assert "FBF" not in code and "(" not in code
    assert code.split() == ["a", "b", "c"]


def test_cli_unguarded(tmp_path):
    _, listing = gpu_tree(tmp_path)
    saved = tmp_path / "free_breakfast_tests_gpu.json"  # the binary is the stem
    saved.write_text(json.dumps(listing))
    cmd = [sys.executable, str(_TOOL), "--gtest-json", str(saved)]
    cmd += ["--source", str(tmp_path), "--no-metrics"]
    proc = subprocess.run([*cmd, "--unguarded"], capture_output=True, text=True)
    assert proc.returncode == 1, proc.stderr
    lines = proc.stdout.splitlines()
    assert len(lines) == 5
    assert lines[0] == "test/a_gpu.cpp:25  Plain.HostClaim  (free_breakfast_tests_gpu)"
    assert "5 unguarded test definitions over" in proc.stderr
    doc = json.loads(
        subprocess.run([*cmd, "--json"], capture_output=True, text=True).stdout
    )
    assert [u["test"] for u in doc["unguarded"]][:2] == [
        "Plain.HostClaim",
        "Unguarded.NoSetUpGuard",
    ]
    # Every case guarded: exit 0, nothing listed.
    (tmp_path / "test" / "a_gpu.cpp").write_text(
        GPU_SOURCE.replace("{\n}", "{\n  FBF_REQUIRE_GPU();\n}")
        .replace("EXPECT_EQ(1, 1);", "FBF_REQUIRE_GPU();")
        .replace("  // FBF_REQUIRE_GPU();\n", "  FBF_REQUIRE_GPU();\n")
    )
    clean = subprocess.run([*cmd, "--unguarded"], capture_output=True, text=True)
    assert clean.returncode == 0, clean.stdout
    assert clean.stdout == ""


def test_cli_needs_cases(tmp_path):
    proc = subprocess.run(
        [sys.executable, str(_TOOL), "--source", str(tmp_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 2


def test_the_trees_markers_parse():
    """Every marker in this checkout's test sources (the files, not git) has
    zero to two string arguments; where each lands needs a build's listing."""
    files = {p.relative_to(_ROOT).as_posix() for p in (_ROOT / "test").rglob("*.cpp")}
    markers = tf.markers_in(_ROOT, files)
    assert markers, "no FBF_FAMILY_REPRESENTATIVE in test/"
    for m in markers:
        assert len(m.args) <= (2 if m.macro == tf.MARKER else 1), m


def test_the_default_host_is_the_one_metrics_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The records' host is metrics.host_name, not the hostname (#985): a
    # container's hostname is its id, so a gethostname() default filtered out
    # every record this box wrote.
    monkeypatch.setenv("METRICS_HOST", "the-box")
    assert tf.this_host() == "the-box"
    monkeypatch.delenv("METRICS_HOST")
    metrics_spec = importlib.util.spec_from_file_location(
        "devtools_metrics_for_families", _TOOL.parent / "metrics.py"
    )
    metrics = importlib.util.module_from_spec(metrics_spec)
    metrics_spec.loader.exec_module(metrics)
    assert tf.this_host() == metrics.host_name()
