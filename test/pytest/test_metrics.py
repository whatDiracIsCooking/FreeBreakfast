"""devtools/metrics.py: the record built from committed fixture reports
(gtest JSON, ctest and pytest JUnit, .ninja_log), and its append to the
orphan `metrics` branch, in throwaway repositories under tmp_path. Nothing
here builds, needs a GPU, or reads this checkout's history."""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_METRICS_PY = _ROOT / "devtools" / "metrics.py"
_DATA = Path(__file__).resolve().parent / "data" / "metrics"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


metrics = _load("devtools_metrics", _METRICS_PY)
gates = _load("devtools_test_py_for_metrics", _ROOT / "devtools" / "test.py")

_SHA = "0123456789abcdef0123456789abcdef01234567"


def _clean_env(**extra: str) -> dict[str, str]:
    # The pre-push hook exports GIT_DIR and PRE_COMMIT_*; a test that runs git
    # under them acts on this checkout's own repository (#694).
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "PRE_COMMIT"))
    }
    env.update(extra)
    return env


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout


@pytest.fixture(autouse=True)
def _no_git_env(monkeypatch: pytest.MonkeyPatch) -> None:
    # metrics.py strips these itself; a stale GIT_DIR here would prove it.
    for name in list(os.environ):
        if name.startswith(("GIT_", "PRE_COMMIT")):
            monkeypatch.delenv(name)


# --- standard library only --------------------------------------------------


def test_imports_only_the_standard_library() -> None:
    tree = ast.parse(_METRICS_PY.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module.split(".")[0])
    assert names - {"__future__"} <= sys.stdlib_module_names, names


def test_imports_under_isolated_python() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            "import importlib.util as u, sys;"
            "s = u.spec_from_file_location('m', sys.argv[1]);"
            "m = u.module_from_spec(s); s.loader.exec_module(m); print(m.SCHEMA)",
            str(_METRICS_PY),
        ],
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "1"


# --- the documented schema ---------------------------------------------------


def _doc_fields() -> list[str]:
    """The `Field` column of docs/testing.md's record table, in order."""
    text = (_ROOT / "docs" / "testing.md").read_text()
    table = text.split("A record, schema version 1:", 1)[1].split("\n\n", 2)[1]
    fields = []
    for row in table.splitlines()[2:]:
        fields += re.findall(r"`([a-z]+)`", row.split("|")[1])
    return fields


def test_fields_are_the_documented_ones() -> None:
    assert _doc_fields() == list(metrics.FIELDS)


# --- the reports --------------------------------------------------------------


def test_gtest_json_cases() -> None:
    cases = metrics.gtest_cases(
        _DATA / "free_breakfast_tests_gpu.json", "free_breakfast_tests_gpu"
    )
    assert cases == [
        {
            "name": "Overlap.Symmetric",
            "binary": "free_breakfast_tests_gpu",
            "status": "pass",
            "seconds": 0.012,
            "slow": False,
        },
        {
            "name": "Overlap.NeedsADevice",
            "binary": "free_breakfast_tests_gpu",
            "status": "skip",
            "seconds": 0.0,
            "slow": False,
        },
        {
            "name": "Exchange/ExchangeGolden.Matches/1",
            "binary": "free_breakfast_tests_gpu",
            "status": "fail",
            "seconds": 3.25,
            "slow": True,
        },
        {
            "name": "Exchange/ExchangeGolden.DISABLED_Huge/0",
            "binary": "free_breakfast_tests_gpu",
            "status": "skip",
            "seconds": 0.0,
            "slow": False,
        },
    ]


def test_ctest_junit_cases_take_their_binary_from_the_listing() -> None:
    binaries = metrics.ctest_binaries((_DATA / "listing.json").read_text())
    assert binaries == {
        "Basis.ParsesSto3g": "free_breakfast_basis_tests_host",
        "Basis.RejectsCartesian": "free_breakfast_basis_tests_host",
        "Gpu.Serial": "free_breakfast_tests_gpu",
    }
    cases = metrics.junit_cases(_DATA / "ctest.xml", binaries)
    assert [(c["name"], c["binary"], c["status"], c["seconds"]) for c in cases] == [
        ("Basis.ParsesSto3g", "free_breakfast_basis_tests_host", "pass", 0.031),
        ("Basis.RejectsCartesian", "free_breakfast_basis_tests_host", "fail", 0.02),
        ("Gpu.Serial", "free_breakfast_tests_gpu", "skip", 0.41),
        ("Unlisted.Case", "ctest", "pass", 0.001),
    ]


def test_gtest_xml_slow_reads_each_entry_report() -> None:
    # One report per ctest entry; a param's `/` nests it; a report that does
    # not parse adds nothing.
    assert metrics.gtest_xml_slow(_DATA / "gtest-xml") == {
        "Basis.ParsesSto3g": False,
        "Cases/Foo.Bar/x": True,
        "Gpu.Serial": True,
    }


def test_with_slow_marks_only_the_cases_it_names() -> None:
    binaries = metrics.ctest_binaries((_DATA / "listing.json").read_text())
    cases = metrics.with_slow(
        metrics.junit_cases(_DATA / "ctest.xml", binaries),
        metrics.gtest_xml_slow(_DATA / "gtest-xml"),
    )
    assert {c["name"]: c.get("slow") for c in cases} == {
        "Basis.ParsesSto3g": False,
        "Basis.RejectsCartesian": None,
        "Gpu.Serial": True,
        "Unlisted.Case": None,
    }


def test_gtest_xml_slow_of_no_directory(tmp_path: Path) -> None:
    assert metrics.gtest_xml_slow(tmp_path / "absent") == {}


def test_every_ctest_entry_writes_its_report_where_test_py_reads_it() -> None:
    # test/CMakeLists.txt's _fb_gtest_xml_dir is test.py's GTEST_XML_DIR under
    # the build directory, and every registration ctest runs passes it (the
    # host binaries, the RUN_SERIAL cases), so no ctest-run case is left with
    # its membership unrecorded (#1016). A batched GPU registration must not:
    # the flag would make batchable() refuse its every entry.
    text = (_ROOT / "test" / "CMakeLists.txt").read_text()
    assert (
        f'set(_fb_gtest_xml_dir "${{PROJECT_BINARY_DIR}}/{gates.GTEST_XML_DIR}")'
        in text
    )
    ctest_run = batched = 0
    for cml in sorted((_ROOT / "test").glob("*/CMakeLists.txt")):
        if cml.parent.name == "experimental":
            continue
        for call in re.findall(
            r"^gtest_discover_tests\((.*?)^\)", cml.read_text(), re.S | re.M
        ):
            by_ctest = "_tests_host" in call or "RUN_SERIAL TRUE" in call
            ctest_run += by_ctest
            batched += not by_ctest
            assert ("XML_OUTPUT_DIR ${_fb_gtest_xml_dir}" in call) == by_ctest, cml
    assert ctest_run >= 8
    assert batched >= 4


def test_pytest_junit_cases_are_node_ids() -> None:
    cases = metrics.junit_cases(_DATA / "pytest.xml")
    assert [(c["name"], c["binary"], c["status"], c["seconds"]) for c in cases] == [
        ("test/pytest/test_a.py::test_ok[x-1]", "pytest", "pass", 0.11),
        ("test/pytest/test_a.py::TestGroup::test_in_class", "pytest", "pass", 0.002),
        ("test/pytest/test_a.py::test_skipped", "pytest", "skip", 0.0),
        ("test/pytest/sub/test_b.py::test_broken", "pytest", "fail", 1.5),
        ("test/pytest/sub/test_b.py::test_errors", "pytest", "fail", 0.0),
    ]


def test_ninja_log_without_an_offset_reads_the_last_run() -> None:
    files = metrics.ninja_files(_DATA / "ninja_log")
    # The fixture's first run built a.o and b.o; the second rebuilt b.o and
    # built c.o and d.o, which is the run that counts.
    assert files == [
        {"file": "src/b.o", "seconds": 0.2},
        {"file": "src/c.o", "seconds": 1.25},
        {"file": "src/d.o", "seconds": 0.5},
    ]


def test_ninja_log_with_an_offset_reads_what_was_appended(tmp_path: Path) -> None:
    log = tmp_path / ".ninja_log"
    shutil.copy(_DATA / "ninja_log", log)
    since = metrics.ninja_offset(log)
    # CMake regenerating restarts ninja's clock mid-build: both parts count,
    # and an output built twice keeps its last time.
    with log.open("a") as f:
        f.write("0\t900\t1\tbuild.ninja\tff\n")
        f.write("10\t4010\t1\tsrc/e.o\taa\n")
        f.write("20\t520\t1\tsrc/e.o\tbb\n")
    assert metrics.ninja_files(log, since) == [
        {"file": "build.ninja", "seconds": 0.9},
        {"file": "src/e.o", "seconds": 0.5},
    ]


def test_ninja_log_rewritten_since_the_offset_falls_back(tmp_path: Path) -> None:
    log = tmp_path / ".ninja_log"
    shutil.copy(_DATA / "ninja_log", log)
    since = metrics.ninja_offset(log)
    # A recompaction writes a new file, so a new inode.
    fresh = tmp_path / "recompacted"
    shutil.copy(_DATA / "ninja_log", fresh)
    log.unlink()
    fresh.rename(log)
    assert metrics.ninja_files(log, since) == metrics.ninja_files(log)


def test_no_ninja_log_has_no_offset(tmp_path: Path) -> None:
    assert metrics.ninja_offset(tmp_path / ".ninja_log") is None


# --- the build directory -----------------------------------------------------


@pytest.fixture
def build_dir(tmp_path: Path) -> Path:
    d = tmp_path / "build"
    shutil.copytree(_DATA / "build", d)
    return d


def test_toolchain_is_the_build_directorys_own(build_dir: Path) -> None:
    assert metrics.toolchain(build_dir, "cuda") == {
        "clang": "20.1.8",
        "cmake": "4.2.3",
        "nvcc": "13.0.88",
    }


def test_launcher_follows_the_sccache_probe(build_dir: Path) -> None:
    assert metrics.launcher_on(build_dir)
    cache = build_dir / "CMakeCache.txt"
    cache.write_text(
        cache.read_text().replace(
            "FREE_BREAKFAST_SCCACHE:FILEPATH=/usr/local/bin/sccache",
            "FREE_BREAKFAST_SCCACHE:FILEPATH=FREE_BREAKFAST_SCCACHE-NOTFOUND",
        )
    )
    assert not metrics.launcher_on(build_dir)


def test_gpu_info_reads_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        metrics,
        "_run",
        lambda cmd: "NVIDIA GeForce RTX 3080, 580.126.20, 10240\n",
    )
    assert metrics.gpu_info("cuda") == {
        "name": "NVIDIA GeForce RTX 3080",
        "driver": "580.126.20",
        "vram_mib": 10240,
    }


def test_gpu_info_reads_rocm_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    doc = {
        "card0": {
            "Card Series": "AMD Radeon RX 9070",
            "VRAM Total Memory (B)": str(16 * 1024**3),
        },
        "system": {"Driver version": "6.12.12"},
    }
    monkeypatch.setattr(metrics, "_run", lambda cmd: json.dumps(doc))
    assert metrics.gpu_info("hip") == {
        "name": "AMD Radeon RX 9070",
        "driver": "6.12.12",
        "vram_mib": 16384,
    }


def test_gpu_info_without_a_tool_is_null(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(metrics, "_run", lambda cmd: None)
    for backend in ("cuda", "hip"):
        assert metrics.gpu_info(backend) == {
            "name": None,
            "driver": None,
            "vram_mib": None,
        }


# --- the record ---------------------------------------------------------------


def _fixture_record() -> dict:
    """The record of a run that wrote exactly the fixture reports, as
    test.py pr --record collects them (gates.run_cases)."""
    results = Path(_DATA)
    return metrics.make_record(
        kind="weekly",
        started="2026-11-01T03:00:00Z",
        finished="2026-11-01T04:10:00Z",
        commit=_SHA,
        backend="cuda",
        preset="default",
        gpu={
            "name": "NVIDIA GeForce RTX 3080",
            "driver": "580.126.20",
            "vram_mib": 10240,
        },
        toolchain=metrics.toolchain(_DATA / "build", "cuda"),
        build=metrics.build_info(
            mode="clean",
            launcher=False,
            jobs=8,
            seconds=1234.5678,
            files=metrics.ninja_files(_DATA / "ninja_log"),
        ),
        cases=gates.run_cases(metrics, results),
        result="fail",
        host="box",
    )


def test_fixture_reports_make_the_documented_record() -> None:
    record = _fixture_record()
    expected = json.loads((_DATA / "expected_record.json").read_text())
    assert record == expected
    assert list(record) == list(metrics.FIELDS)


def test_run_cases_collects_every_report() -> None:
    cases = gates.run_cases(metrics, _DATA)
    # ctest's four, the one GPU batch report, pytest's five.
    assert len(cases) == 4 + 4 + 5
    assert {c["binary"] for c in cases} >= {"pytest", "free_breakfast_tests_gpu"}


def test_run_cases_of_an_empty_directory(tmp_path: Path) -> None:
    assert gates.run_cases(metrics, tmp_path) == []


def test_a_warm_build_carries_no_files() -> None:
    build = metrics.build_info(
        mode="warm", launcher=True, jobs=12, seconds=3.0, files=[{"file": "x"}]
    )
    assert build == {"mode": "warm", "launcher": "on", "jobs": 12, "seconds": 3.0}


@pytest.mark.parametrize(
    "change, why",
    [
        ({"kind": "hourly"}, "kind"),
        ({"result": "ok"}, "result"),
        ({"commit": "0123abc"}, "full SHA"),
        ({"cases": [{"name": "X.Y", "status": "flaky"}]}, "status"),
    ],
)
def test_make_record_refuses(change: dict, why: str) -> None:
    kw = dict(
        kind="manual",
        started="2026-11-01T03:00:00Z",
        finished="2026-11-01T03:01:00Z",
        commit=_SHA,
        backend="cuda",
        preset="default",
        gpu={},
        toolchain={},
        build=None,
        cases=[],
        result="pass",
    )
    kw.update(change)
    with pytest.raises(metrics.MetricsError, match=why):
        metrics.make_record(**kw)


# --- the metrics branch -------------------------------------------------------


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A main checkout with one commit, ignoring .claude/worktrees/ as this
    tree's .gitignore does."""
    r = tmp_path / "main"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "t@example.com")
    _git(r, "config", "user.name", "t")
    _git(r, "config", "commit.gpgsign", "false")
    (r / ".gitignore").write_text(".claude/worktrees/\n")
    (r / "src.txt").write_text("code\n")
    _git(r, "add", ".")
    _git(r, "commit", "-q", "-m", "main")
    return r


def _main_state(r: Path) -> tuple[str, str, str, bytes]:
    """HEAD, the status, the branch HEAD names, and the index's bytes."""
    return (
        _git(r, "rev-parse", "HEAD"),
        _git(r, "status", "--porcelain=v1"),
        _git(r, "symbolic-ref", "HEAD"),
        (r / ".git" / "index").read_bytes(),
    )


def _record(started: str = "2026-11-01T03:00:00Z", result: str = "pass") -> dict:
    return metrics.make_record(
        kind="manual",
        started=started,
        finished=started,
        commit=_SHA,
        backend="cuda",
        preset="default",
        gpu={},
        toolchain={},
        build=None,
        cases=[{"name": "A.B", "binary": "x", "status": "pass", "seconds": 1.0}],
        result=result,
        host="box",
    )


def test_first_append_makes_an_orphan_branch_and_touches_nothing_on_main(
    repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Either would aim git at the wrong repository if it leaked through.
    monkeypatch.setenv("GIT_DIR", str(repo / "nowhere"))
    monkeypatch.setenv("GIT_INDEX_FILE", str(repo / "nowhere.index"))
    wt = repo / ".claude" / "worktrees" / "metrics"
    before = _main_state(repo)
    sha, where = metrics.append(_record(), repo, wt)
    assert _main_state(repo) == before
    assert where == wt / "runs" / "2026-11.jsonl"
    lines = where.read_text().splitlines()
    assert [json.loads(line) for line in lines] == [_record()]
    assert _git(repo, "rev-parse", "refs/heads/metrics").strip() == sha
    # An orphan: no parent, no history shared with main.
    assert _git(repo, "rev-list", "--count", "metrics").strip() == "1"
    merge_base = subprocess.run(
        ["git", "-C", str(repo), "merge-base", "main", "metrics"],
        env=_clean_env(),
        capture_output=True,
        check=False,
    )
    assert merge_base.returncode == 1
    assert _git(repo, "ls-tree", "-r", "--name-only", "metrics").split() == [
        "runs/2026-11.jsonl"
    ]
    # The worktree is left clean, on the branch.
    assert _git(wt, "status", "--porcelain").strip() == ""
    assert _git(wt, "symbolic-ref", "--short", "HEAD").strip() == "metrics"


def test_appends_accumulate_one_line_and_one_commit_each(repo: Path) -> None:
    wt = repo / ".claude" / "worktrees" / "metrics"
    metrics.append(_record(), repo, wt)
    metrics.append(_record(result="fail"), repo, wt)
    sha, where = metrics.append(_record(started="2026-12-01T00:00:00Z"), repo, wt)
    assert len((wt / "runs" / "2026-11.jsonl").read_text().splitlines()) == 2
    assert where.name == "2026-12.jsonl"
    assert _git(repo, "rev-list", "--count", "metrics").strip() == "3"
    assert _git(repo, "log", "-1", "--format=%s", "metrics").strip() == (
        f"metrics: manual {_SHA[:12]} default pass"
    )
    assert _git(wt, "status", "--porcelain").strip() == ""


def test_a_deleted_worktree_is_remade_on_the_branch(repo: Path) -> None:
    wt = repo / ".claude" / "worktrees" / "metrics"
    first, _ = metrics.append(_record(), repo, wt)
    shutil.rmtree(wt)
    second, where = metrics.append(_record(), repo, wt)
    assert _git(repo, "rev-parse", f"{second}^").strip() == first
    assert len(where.read_text().splitlines()) == 2


def test_the_branch_is_taken_from_origin_when_only_it_has_one(
    repo: Path, tmp_path: Path
) -> None:
    wt = repo / ".claude" / "worktrees" / "metrics"
    first, _ = metrics.append(_record(), repo, wt)
    clone = tmp_path / "clone"
    _git(tmp_path, "clone", "-q", str(repo), str(clone))
    _git(clone, "config", "user.email", "t@example.com")
    _git(clone, "config", "user.name", "t")
    assert _git(clone, "branch", "--list", "metrics").strip() == ""
    second, _ = metrics.append(_record(), clone, clone / "metrics-wt")
    assert _git(clone, "rev-parse", f"{second}^").strip() == first


def test_a_worktree_on_another_branch_is_refused(repo: Path) -> None:
    wt = repo / ".claude" / "worktrees" / "metrics"
    _git(repo, "worktree", "add", "-q", "-b", "other", str(wt))
    with pytest.raises(metrics.MetricsError, match="not metrics"):
        metrics.append(_record(), repo, wt)


def test_an_unrelated_directory_is_refused(repo: Path) -> None:
    wt = repo / ".claude" / "worktrees" / "metrics"
    wt.mkdir(parents=True)
    (wt / "notes.txt").write_text("mine\n")
    with pytest.raises(metrics.MetricsError, match="is not the metrics worktree"):
        metrics.append(_record(), repo, wt)


# --- test.py pr --record --------------------------------------------------------


def test_pr_takes_record() -> None:
    args, _ = gates.parse(["pr", "--record"])
    assert args.record
    assert not gates.parse(["pr"])[0].record


def test_record_refuses_a_dirty_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(gates, "tree_rev", lambda: ("abc1234+dirty", "deadbeef"))
    args, rest = gates.parse(
        ["pr", "--record", "--only", "python", "--report", str(tmp_path)]
    )
    cfg = gates.read_config(_clean_env())
    assert gates.cmd_pr(args, cfg, rest) == 2
    assert "--record needs a clean tree" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []
