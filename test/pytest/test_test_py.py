"""devtools/test.py: argument parsing, push dispatch, the GPU batch split,
pr's representative set, config.sh reading and the card lock. The watchdog
has its own file, test_gpu_batch_watchdog.py. Nothing here builds or needs a GPU."""

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
_TEST_PY = _ROOT / "devtools" / "test.py"
_spec = importlib.util.spec_from_file_location("devtools_test_py", _TEST_PY)
gates = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gates)


def _clean_env(**extra: str) -> dict[str, str]:
    # The pre-push hook exports GIT_DIR and PRE_COMMIT_*; a test that runs git
    # under them acts on this checkout's own repository (#694).
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "PRE_COMMIT"))
    }
    env.update(extra)
    return env


# --- standard library only --------------------------------------------------


def test_imports_only_the_standard_library() -> None:
    tree = ast.parse(_TEST_PY.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.add(node.module.split(".")[0])
    assert names - {"__future__"} <= sys.stdlib_module_names, names


def test_help_runs_under_isolated_python() -> None:
    # -I: no site-packages, no user site, no cwd on sys.path -- a bare python3.
    proc = subprocess.run(
        [sys.executable, "-I", str(_TEST_PY), "--help"],
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "push" in proc.stdout and "pr" in proc.stdout


# --- argument parsing --------------------------------------------------------


def test_pr_takes_ctest_args_after_dashdash() -> None:
    args, rest = gates.parse(["pr", "--preset", "hip", "--tidy", "--", "-R", "X"])
    assert args.gate == "pr" and args.preset == "hip" and args.tidy
    assert rest == ["-R", "X"]


def test_pr_defaults() -> None:
    args, rest = gates.parse(["pr"])
    assert args.preset is None and args.jobs is None and args.only is None
    assert not (args.fresh or args.clean or args.no_test or args.tidy or args.all)
    assert rest == []


def test_push_flags() -> None:
    args, _ = gates.parse(["push", "--dry-run", "-j", "8", "--preset", "default"])
    assert args.dry_run and args.jobs == "8" and args.preset == "default"
    assert not args.all and gates.parse(["push", "--all"])[0].all


def test_push_refuses_passthrough() -> None:
    with pytest.raises(SystemExit):
        gates.parse(["push", "--", "-k", "x"])


def test_a_gate_is_required() -> None:
    with pytest.raises(SystemExit):
        gates.parse([])


def test_only_is_checked() -> None:
    with pytest.raises(SystemExit):
        gates.parse(["pr", "--only", "rust"])
    assert gates.parse(["pr", "--only", "cargo"])[0].only == "cargo"


def test_cargo_lint_takes_nothing() -> None:
    assert gates.parse(["cargo-lint"])[0].gate == "cargo-lint"
    with pytest.raises(SystemExit):
        gates.parse(["cargo-lint", "rust/Cargo.toml"])


# --- push dispatch -----------------------------------------------------------


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (["src/core/a.cppm"], (True, False, False)),
        (["src/x/k.cu", "src/x/k.cuh"], (True, False, False)),
        (["test/gtest/CMakeLists.txt"], (True, False, False)),
        (["CMakePresets.json"], (True, False, False)),
        (["cmake/Thing.cmake"], (True, False, False)),
        (["test/pytest/test_x.py"], (False, True, False)),
        (["devtools/test.py", "src/a.cpp"], (True, True, False)),
        (["docs/testing.md", ".pre-commit-config.yaml"], (False, False, False)),
        # A submodule bump is its gitlink path, which matches nothing; files
        # inside a submodule are its own repository's business.
        (["deps/WarpWraps"], (False, False, False)),
        (["deps/FelizJueves/tools/emit/x.py"], (False, False, False)),
        # experimental/ and its tests are outside the smoke's preset, so their
        # C++ fires nothing; the root line that adds them still does, and so
        # does anything outside them pushed alongside.
        (["experimental/rust/s_overlap/ffi/fb_overlap.cpp"], (False, False, False)),
        (["experimental/CMakeLists.txt"], (False, False, False)),
        (["test/experimental/sort.cpp"], (False, False, False)),
        (["experimental/x/CMakeLists.txt", "src/a.cpp"], (True, False, False)),
        (["CMakeLists.txt"], (True, False, False)),
        (["experimental/cope_rearrangement/scan.py"], (False, True, False)),
        (["src/experimental_thing.cpp"], (True, False, False)),
        # Anything under rust/ fires the cargo half alone: Rust, its .cu and
        # its own CMake project (never the C++ smoke), its docs and its one
        # .py (never the Python half). What is pushed alongside still fires
        # its own half.
        (["rust/core/log/src/lib.rs"], (False, False, True)),
        (["rust/recipes/kernels/merge_sort.cu"], (False, False, True)),
        (["rust/CMakeLists.txt", "rust/CMakePresets.json"], (False, False, True)),
        (["rust/cmake/x.cmake", "rust/device_headers/a.h"], (False, False, True)),
        (["rust/README.md", "rust/Cargo.toml"], (False, False, True)),
        (["rust/docs/md2html.py"], (False, False, True)),
        (["rust/docs/md2html.py", "src/a.cpp"], (True, False, True)),
        # fb-device-check runs the bindings generator.
        (["devtools/rust-bindings.py"], (False, True, True)),
        (["src/rust.cpp", "devtools/rust.py"], (True, True, False)),
        (["src/core/log/log.cppm", "src/README.md"], (True, False, False)),
        (["experimental/rust/s_overlap/src/lib.rs"], (False, False, False)),
        ([], (False, False, False)),
        (None, (True, True, True)),
    ],
)
def test_halves(files: list[str] | None, expected: tuple[bool, bool, bool]) -> None:
    assert gates.halves(files) == expected


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        env=_clean_env(
            GIT_AUTHOR_NAME="t",
            GIT_AUTHOR_EMAIL="t@example.com",
            GIT_COMMITTER_NAME="t",
            GIT_COMMITTER_EMAIL="t@example.com",
        ),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A throwaway repository: a base commit, a C++ commit and a .py commit.
    Never the real checkout, whose history a CI clone does not have."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    (r / "README.md").write_text("base\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "base")
    (r / "src").mkdir()
    (r / "src" / "a.cppm").write_text("export module a;\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "cpp")
    (r / "tool.py").write_text("x = 1\n")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "py")
    return r


def _range(repo: Path, old: str, new: str) -> dict[str, str]:
    return _clean_env(
        PRE_COMMIT="1",
        PRE_COMMIT_FROM_REF=_git(repo, "rev-parse", old),
        PRE_COMMIT_TO_REF=_git(repo, "rev-parse", new),
    )


def test_changed_files_reads_pre_commits_refs(repo: Path) -> None:
    files = gates.changed_files(_range(repo, "HEAD~2", "HEAD~1"), repo)
    assert files == ["src/a.cppm"]
    assert gates.halves(files) == (True, False, False)
    files = gates.changed_files(_range(repo, "HEAD~1", "HEAD"), repo)
    assert files == ["tool.py"]
    assert gates.halves(files) == (False, True, False)
    assert gates.changed_files(_range(repo, "HEAD~2", "HEAD"), repo) == [
        "src/a.cppm",
        "tool.py",
    ]


def test_by_hand_diffs_against_origin_main(repo: Path) -> None:
    base = _git(repo, "rev-parse", "HEAD~2")
    _git(repo, "update-ref", "refs/remotes/origin/main", base)
    assert gates.changed_files(_clean_env(), repo) == ["src/a.cppm", "tool.py"]


def test_by_hand_without_origin_main_fails_clearly(repo: Path) -> None:
    # A shallow or remote-less clone has no origin/main to diff against.
    with pytest.raises(gates.DiffError, match="git fetch origin main"):
        gates.changed_files(_clean_env(), repo)


def test_hooked_unreadable_range_tests_both_halves(
    repo: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Under pre-commit a range git cannot read is never a reason to test less.
    env = _clean_env(
        PRE_COMMIT="1", PRE_COMMIT_FROM_REF="0" * 40, PRE_COMMIT_TO_REF="HEAD"
    )
    assert gates.changed_files(env, repo) is None
    assert "testing both halves" in capsys.readouterr().err


def test_a_whole_history_push_runs_both_halves(repo: Path) -> None:
    # pre-commit sets no refs when the push includes the root commit.
    assert gates.changed_files(_clean_env(PRE_COMMIT="1"), repo) is None


def test_shallow_clone(repo: Path, tmp_path: Path) -> None:
    # The CI case: a depth-1 clone lacks the parent commit a range names.
    clone = tmp_path / "shallow"
    _git(tmp_path, "clone", "-q", "--depth", "1", f"file://{repo}", str(clone))
    parent = _git(repo, "rev-parse", "HEAD~1")
    hooked = _clean_env(
        PRE_COMMIT="1", PRE_COMMIT_FROM_REF=parent, PRE_COMMIT_TO_REF="HEAD"
    )
    assert gates.changed_files(hooked, clone) is None
    # By hand it diffs against the clone's own origin/main: HEAD, so empty.
    assert gates.changed_files(_clean_env(), clone) == []


def test_dry_run_dispatch(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    env = _range(repo, "HEAD~2", "HEAD")
    real = gates.changed_files
    monkeypatch.setattr(gates, "changed_files", lambda _env: real(env, repo))
    args, _ = gates.parse(["push", "--dry-run"])
    assert gates.cmd_push(args, {}) == 0
    out = capsys.readouterr().out
    assert "2 changed file(s) -> C++ half: yes, Python half: yes, cargo half: no" in out
    assert "push:   src/a.cppm" in out and "push:   tool.py" in out


def test_push_by_hand_without_a_base_exits_2(
    repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    real = gates.changed_files
    monkeypatch.setattr(gates, "changed_files", lambda env: real(_clean_env(), repo))
    args, _ = gates.parse(["push", "--dry-run"])
    assert gates.cmd_push(args, {}) == 2
    err = capsys.readouterr().err
    assert "push: cannot diff origin/main...HEAD" in err
    assert "Traceback" not in err


# --- the cargo half ------------------------------------------------------------

# A stand-in `cargo` on PATH: a two-member workspace, one member linking
# wwr-sys, and a `cargo test` that exits FAKE_CARGO_TEST_RC. Each call's
# arguments go to FAKE_CARGO_LOG. Without --no-default-features, `test` is
# tier 4: fb-core-wwr's one device case is ignored unless --include-ignored.
_FAKE_CARGO = """
import json, os, sys
args = sys.argv[1:]
with open(os.environ["FAKE_CARGO_LOG"], "a") as f:
    f.write(" ".join(args) + "\\n")
ws = "/ws/rust"
if args[0] == "metadata":
    print(json.dumps({
        "packages": [
            {"name": "fb-core-log", "id": "log", "links": None,
             "manifest_path": ws + "/core/log/Cargo.toml"},
            {"name": "fb-core-wwr", "id": "wwr", "links": None,
             "manifest_path": ws + "/core/wwr/Cargo.toml"},
            {"name": "wwr-sys", "id": "sys", "links": "wwr_capi",
             "manifest_path": "/ws/deps/wwr-sys/Cargo.toml"},
        ],
        "workspace_members": ["log", "wwr"],
        "workspace_root": ws,
        "resolve": {"nodes": [
            {"id": "log", "deps": []},
            {"id": "wwr", "deps": [{"pkg": "log"}, {"pkg": "sys"}]},
            {"id": "sys", "deps": []},
        ]},
    }))
elif args[0] == "tree":
    name = args[args.index("-p") + 1]
    print(f"{name} v0.1.0 (/x)")
    if name == "fb-core-wwr":
        print("wwr-sys v1.0.0 (/x)")
elif args[0] == "test" and "--no-default-features" not in args:
    print("     Running unittests src/lib.rs (/t/deps/fb_core_wwr-0123abcd)")
    print("test status::tests::names ... ok")
    if "--include-ignored" in args:
        print("test runtime::stream::tests::syncs ... ok")
        print("test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured")
    else:
        print("test runtime::stream::tests::syncs ... ignored, tier 4: needs a GPU")
        print("test result: ok. 1 passed; 0 failed; 1 ignored; 0 measured")
    sys.exit(int(os.environ.get("FAKE_CARGO_TIER4_RC", "0")))
elif args[0] == "test":
    print("     Running unittests src/lib.rs (/t/deps/fb_core_log-0123abcd)")
    print("test result: ok. 3 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out")
    print("     Running tests/empty.rs (/t/deps/empty-89abcdef)")
    print("test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out")
    print("   Doc-tests fb_core_log")
    print("test result: ok. 2 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out")
    sys.exit(int(os.environ.get("FAKE_CARGO_TEST_RC", "0")))
"""


@pytest.fixture
def fake_cargo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The stand-in cargo first on PATH; returns its argument log."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    cargo = bindir / "cargo"
    cargo.write_text(f"#!{sys.executable}\n{_FAKE_CARGO}")
    cargo.chmod(0o755)
    calls = tmp_path / "cargo-calls.log"
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_CARGO_LOG", str(calls))
    monkeypatch.setattr(gates, "tree_rev", lambda: ("abc1234", ""))
    return calls


def test_link_free_crates_leave_out_a_linking_graph(fake_cargo: Path) -> None:
    assert gates.link_free_crates() == (["fb-core-log"], ["fb-core-log"])
    trees = [c for c in fake_cargo.read_text().splitlines() if c.startswith("tree")]
    assert sum("--no-default-features" in c for c in trees) == 2
    assert all("-e normal,dev,build" in c for c in trees)


def test_cargo_counts() -> None:
    lines = [
        "     Running unittests src/lib.rs (/t/debug/deps/fb_atom-0123abcd)",
        "test result: ok. 14 passed; 0 failed; 1 ignored; 0 measured; 0 filtered out",
        "     Running tests/xyz.rs (/t/debug/deps/xyz-4567)",
        "test result: FAILED. 6 passed; 1 failed; 0 ignored; 0 measured",
        "     Running tests/threads.rs (/t/debug/deps/threads-89ab)",
        "threads: ok",
        "   Doc-tests fb_atom",
        "test result: ok. 0 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out",
        "     Running tests/last.rs (/t/debug/deps/last-cdef)",
    ]
    rows, totals = gates.cargo_counts(lines)
    assert rows == [
        "fb_atom unittests src/lib.rs: 14 passed, 0 failed, 1 ignored",
        "xyz tests/xyz.rs: 6 passed, 1 failed, 0 ignored",
        "threads tests/threads.rs: no libtest counts (harness = false, or died)",
        "last tests/last.rs: no libtest counts (harness = false, or died)",
    ]
    assert totals == [20, 1, 1]


def _push_cargo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, files: list[str]
) -> int:
    monkeypatch.setattr(gates, "changed_files", lambda _env: files)

    def other_half(*_a, **_k):
        raise AssertionError("only the cargo half fires")

    monkeypatch.setattr(gates, "python_half", other_half)
    monkeypatch.setattr(gates, "cpp_push", other_half)
    monkeypatch.setattr(gates, "cargo_tier4", lambda *_a: 0)
    args, _ = gates.parse(["push"])
    return gates.cmd_push(
        args,
        {
            "SLOW_TIER_REPORTS": str(tmp_path / "reports"),
            "BUILD_JOBS": "4",
            "CMAKE_PRESET": "hip",
        },
    )


def test_push_runs_the_cargo_half_and_prints_its_counts(
    fake_cargo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert _push_cargo(monkeypatch, tmp_path, ["rust/core/log/src/lib.rs"]) == 0
    out = capsys.readouterr().out
    assert "C++ half: no, Python half: no, cargo half: yes" in out
    assert "fb_core_log unittests src/lib.rs: 3 passed, 0 failed, 1 ignored" in out
    assert "doc-tests fb_core_log: 2 passed, 0 failed, 0 ignored" in out
    assert "empty" not in out
    assert "cargo test: 5 passed, 0 failed, 1 ignored" in out
    assert "test.py push PASS  rev=abc1234  ran=cargo" in out
    calls = fake_cargo.read_text().splitlines()
    test = next(c for c in calls if c.startswith("test"))
    assert test == "test -j 4 --no-default-features -p fb-core-log"
    assert next(c for c in calls if c.startswith("doc")).endswith(
        "--no-deps -p fb-core-log"
    )


def test_a_failing_cargo_half_fails_the_push_and_is_named(
    fake_cargo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FAKE_CARGO_TEST_RC", "101")
    assert _push_cargo(monkeypatch, tmp_path, ["rust/Cargo.toml"]) == 101
    err = capsys.readouterr().err
    assert "cargo half FAILED: cargo test exited 101" in err
    assert "test.py push FAIL(rc=101)  failed=cargo" in err
    # Nothing documented after a failed test step.
    assert not any(c.startswith("doc") for c in fake_cargo.read_text().splitlines())


def test_the_cargo_half_skips_without_cargo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    real = gates.shutil.which
    monkeypatch.setattr(
        gates.shutil, "which", lambda n, *a, **k: None if n == "cargo" else real(n)
    )
    monkeypatch.setattr(gates, "tree_rev", lambda: ("abc1234", ""))
    assert _push_cargo(monkeypatch, tmp_path, ["rust/README.md"]) == 0
    assert "skipping the cargo half" in capsys.readouterr().out


# --- the cargo half's tier 4 -------------------------------------------------


def _package(name: str, path: str, links: str | None = None) -> dict:
    return {
        "name": name,
        "id": name,
        "links": links,
        "manifest_path": f"/ws/rust/{path}/Cargo.toml",
    }


# A workspace shaped like rust/'s: log below wwr; arena and thread_pool on
# wwr; planner on both; recipes-sys and integrals-sys on wwr-sys, with wwr a
# dev-dependency; atom on log alone.
_META = {
    "workspace_root": "/ws/rust",
    "workspace_members": [
        "fb-core-log",
        "fb-core-wwr",
        "fb-core-arena",
        "fb-core-thread-pool",
        "fb-core-planner",
        "fb-recipes-sys",
        "fb-integrals-sys",
        "fb-atom",
        "fb-doc-check",
        "fb-device-check",
    ],
    "packages": [
        _package("fb-core-log", "core/log"),
        _package("fb-core-wwr", "core/wwr"),
        _package("fb-core-arena", "core/arena"),
        _package("fb-core-thread-pool", "core/thread_pool"),
        _package("fb-core-planner", "core/planner"),
        _package("fb-recipes-sys", "recipes/sys", "fb_recipes_kernels"),
        _package("fb-integrals-sys", "integrals/sys", "fb_integrals_kernels"),
        _package("fb-atom", "atom"),
        _package("fb-doc-check", "doc_check"),
        _package("fb-device-check", "device_check"),
        _package("wwr-sys", "../deps/wwr-sys", "wwr_capi"),
    ],
    "resolve": {
        "nodes": [
            {"id": node, "deps": [{"pkg": d} for d in deps]}
            for node, deps in {
                "fb-core-log": [],
                "fb-core-wwr": ["fb-core-log", "wwr-sys"],
                "fb-core-arena": ["fb-core-wwr", "wwr-sys"],
                "fb-core-thread-pool": ["fb-core-wwr", "fb-core-log"],
                "fb-core-planner": ["fb-core-arena", "fb-core-thread-pool"],
                "fb-recipes-sys": ["wwr-sys", "fb-core-wwr"],
                "fb-integrals-sys": ["wwr-sys", "fb-core-wwr"],
                "fb-atom": ["fb-core-log"],
                "fb-doc-check": [],
                "fb-device-check": [],
                "wwr-sys": [],
            }.items()
        ]
    },
}
_LINKING = {
    "fb-core-wwr",
    "fb-core-arena",
    "fb-core-thread-pool",
    "fb-core-planner",
    "fb-recipes-sys",
    "fb-integrals-sys",
}


@pytest.mark.parametrize(
    "files, expected",
    [
        # One crate's diff: that crate and its reverse dependents, no others.
        (["rust/core/arena/src/block.rs"], {"fb-core-arena", "fb-core-planner"}),
        (["rust/core/planner/tests/planner_device.rs"], {"fb-core-planner"}),
        (
            ["rust/core/thread_pool/Cargo.toml"],
            {"fb-core-thread-pool", "fb-core-planner"},
        ),
        # A dev-dependent counts: the -sys crates' tests use wwr.
        (["rust/core/wwr/src/runtime.rs"], _LINKING),
        (["rust/integrals/sys/src/lib.rs"], {"fb-integrals-sys"}),
        # Through a link-free crate: log has no tier 4 of its own.
        (["rust/core/log/src/lib.rs"], _LINKING),
        (["rust/atom/src/lib.rs"], set()),
        # The path rules: kernels are every kernel library's -sys crate's
        # (one library links another); docs and the bindings generator are
        # tier-3 crates'.
        (
            ["rust/recipes/kernels/merge_sort.cu"],
            {"fb-recipes-sys", "fb-integrals-sys"},
        ),
        (["rust/docs/testing.md", "rust/README.md"], set()),
        (["devtools/rust-bindings.py"], set()),
        # A file in no crate and no rule, or no diff at all: everything.
        (["rust/Cargo.lock"], _LINKING),
        (["rust/capi_dir.rs"], _LINKING),
        (None, _LINKING),
        # Nothing under rust/.
        (["src/core/log/log.cppm", "devtools/test.py"], set()),
    ],
)
def test_tier4_selects_a_crate_and_its_reverse_dependents(
    files: list[str] | None, expected: set[str]
) -> None:
    assert gates.affected_crates(files, _META) & _LINKING == expected


def test_cargo_skips_lists_each_ignored_case_with_its_reason() -> None:
    lines = [
        "     Running unittests src/lib.rs (/t/debug/deps/fb_core_wwr-0123abcd)",
        "test status::tests::names ... ok",
        "test runtime::stream::tests::syncs ... ignored, tier 4: needs a GPU",
        "test result: ok. 1 passed; 0 failed; 1 ignored; 0 measured",
        "     Running tests/arena_block.rs (/t/debug/deps/arena_block-4567)",
        "test child_entry ... ignored",
        "   Doc-tests fb_core_arena",
        "test core/arena/src/lib.rs - (line 9) ... ignored, tier 4: needs a GPU",
    ]
    assert gates.cargo_skips(lines) == [
        "fb_core_wwr unittests src/lib.rs: runtime::stream::tests::syncs"
        " -- tier 4: needs a GPU",
        "arena_block tests/arena_block.rs: child_entry -- no reason given",
        "doc-tests fb_core_arena: core/arena/src/lib.rs - (line 9)"
        " -- tier 4: needs a GPU",
    ]


class _HeldCard:
    def __init__(self, backend: str, cfg: dict[str, str]) -> None:
        _HeldCard.backend = backend

    def __enter__(self) -> _HeldCard:
        return self

    def __exit__(self, *exc: object) -> None:
        pass


def _tier4(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, devices: int
) -> tuple[int, list[str]]:
    """cargo_tier4 over the fake workspace, with a stand-in cmake, card and
    probe; (status, the cargo calls)."""
    bindir = Path(os.environ["PATH"].split(os.pathsep)[0])
    cmake = bindir / "cmake"
    cmake.write_text(f"#!{sys.executable}\nimport sys\nsys.exit(0)\n")
    cmake.chmod(0o755)
    monkeypatch.setattr(gates, "toolchain_unavailable", lambda: None)
    monkeypatch.setattr(gates, "Card", _HeldCard)
    seen = f"{devices} device(s) visible" if devices else "no device visible: why"
    monkeypatch.setattr(gates, "device_probe", lambda _build: (devices, seen))
    cfg = {"GPU_HIP_PRESETS": "hip"}
    rc = gates.cargo_tier4(
        "push", cfg, "hip", ["rust/core/wwr/src/lib.rs"], "4", tmp_path / "t4.log"
    )
    assert _HeldCard.backend == "hip"
    return rc, [c for c in Path(os.environ["FAKE_CARGO_LOG"]).read_text().splitlines()]


def test_tier4_without_a_device_lists_every_device_case_as_skipped(
    fake_cargo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc, calls = _tier4(monkeypatch, tmp_path, devices=0)
    assert rc == 0
    out = capsys.readouterr().out
    test = next(c for c in calls if c.startswith("test"))
    assert test == "test -j 4 -p fb-core-wwr -- --test-threads=1"
    assert "tier 4: 1 passed, 0 failed, 1 ignored" in out
    assert "tier 4 skipped 1 case(s) (no device visible: why):" in out
    assert (
        "skipped fb_core_wwr unittests src/lib.rs: runtime::stream::tests::syncs"
        " -- tier 4: needs a GPU" in out
    )


def test_tier4_with_a_device_runs_them_and_skips_none(
    fake_cargo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc, calls = _tier4(monkeypatch, tmp_path, devices=1)
    assert rc == 0
    out = capsys.readouterr().out
    assert next(c for c in calls if c.startswith("test")).endswith(
        "-- --test-threads=1 --include-ignored"
    )
    assert "tier 4: 2 passed, 0 failed, 0 ignored" in out
    assert "tier 4 skipped: none" in out


def test_a_failing_tier4_fails_and_is_named(
    fake_cargo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FAKE_CARGO_TIER4_RC", "101")
    rc, _ = _tier4(monkeypatch, tmp_path, devices=1)
    assert rc == 101
    assert "tier 4 FAILED: cargo test exited 101" in capsys.readouterr().err


def test_tier4_runs_nothing_for_a_link_free_diff(
    fake_cargo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(gates, "toolchain_unavailable", lambda: None)
    monkeypatch.setattr(gates, "affected_crates", lambda *_a: {"fb-core-log"})
    rc = gates.cargo_tier4(
        "push", {"GPU_HIP_PRESETS": "hip"}, "hip", [], "4", tmp_path / "l"
    )
    assert rc == 0
    assert "no affected crate links the library" in capsys.readouterr().out
    assert not any(c.startswith("test") for c in fake_cargo.read_text().splitlines())


def test_tier4_skips_without_cmake(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(gates, "toolchain_unavailable", lambda: "cmake not found")
    rc = gates.cargo_tier4(
        "push", {"GPU_HIP_PRESETS": "hip"}, "hip", None, "4", tmp_path / "l"
    )
    assert rc == 0
    assert "skipping tier 4" in capsys.readouterr().out


# --- the GPU batch split -----------------------------------------------------


def _entry(name: str, command: list[str], labels=("gpu",), serial=False) -> dict:
    props = [{"name": "LABELS", "value": list(labels)}]
    if serial:
        props.append({"name": "RUN_SERIAL", "value": True})
    return {"name": name, "command": command, "properties": props}


def test_split_batches_each_binary(tmp_path: Path) -> None:
    core, main, host = (tmp_path / n for n in ("core_gpu", "tests_gpu", "tests_host"))
    for b in (core, main, host):
        b.touch()
    flag = "--gtest_also_run_disabled_tests"
    listing = {
        "tests": [
            _entry("C.a", [str(core), "--gtest_filter=C.a", flag]),
            _entry("M.a", [str(main), "--gtest_filter=M.a", flag]),
            _entry("M.b", [str(main), "--gtest_filter=M.b", flag]),
            # Kept with ctest: RUN_SERIAL, a launcher, host, two filters.
            _entry("M.serial", [str(main), "--gtest_filter=M.serial"], serial=True),
            _entry("M.wrapped", ["compute-sanitizer", str(main), "--gtest_filter=M.w"]),
            _entry("H.a", [str(host), "--gtest_filter=H.a"], labels=("host",)),
            _entry("M.two", [str(main), "--gtest_filter=A", "--gtest_filter=B"]),
            _entry("M.extra", [str(main), "--gtest_filter=M.x", "--gtest_repeat=2"]),
        ]
    }
    # ctest chatter before the document is skipped.
    text = "Test project /x\n" + json.dumps(listing)
    excluded, filters = gates.split_gpu_entries(text, [core, tmp_path / "none", main])
    assert excluded == ["C.a", "M.a", "M.b"]
    assert filters == ["C.a", "", "M.a:M.b"]


def test_split_leaves_an_oversized_batch_to_ctest(tmp_path: Path) -> None:
    gpu = tmp_path / "gpu"
    gpu.touch()
    case = "S." + "x" * 1000
    listing = {
        "tests": [
            _entry(f"n{i}", [str(gpu), f"--gtest_filter={case}{i}"]) for i in range(200)
        ]
    }
    excluded, filters = gates.split_gpu_entries(json.dumps(listing), [gpu])
    assert excluded == [] and filters == [""]


def test_split_rejects_a_listing_without_json() -> None:
    with pytest.raises(ValueError):
        gates.split_gpu_entries("ctest: no such preset", [])


# --- pr's representative set ------------------------------------------------

# test_families.py --json, stubbed: what pr reads from it.
FAMILIES = {
    "schema": 1,
    "families": {
        "Plain": {
            "members": ["Plain.Refuses", "Plain.Validates"],
            "representative": "Plain.Validates",
            "by": "marked",
        },
        # A value param: ctest prints `/"Singlet"`, gtest `/Singlet`.
        "Spin.Matches": {
            "members": ["Cases/Spin.Matches/Singlet", "Cases/Spin.Matches/Triplet"],
            "representative": "Cases/Spin.Matches/Singlet",
            "by": "cheapest",
        },
        # A typed test: ctest prints `<double>`, gtest `/1`.
        "Typed.Works": {
            "members": ["Typed/0.Works", "Typed/1.Works"],
            "representative": "Typed/1.Works",
            "by": "cheapest",
        },
        "Timing": {
            "members": ["Timing.Big"],
            "representative": None,
            "by": "exempt",
            "reason": "a timing",
        },
        "New": {
            "members": ["New.A", "New.B"],
            "representative": None,
            "by": None,
        },
        "Host": {
            "members": ["Host.Cheap", "Host.Dear"],
            "representative": "Host.Cheap",
            "by": "cheapest",
        },
    },
    "flagged": ["New"],
    "exempt": ["Timing"],
    "not_gtest": ["example_rhf"],
    "errors": [],
    "notes": [],
}


def _listing(gpu: Path, host: Path) -> dict:
    flag = "--gtest_also_run_disabled_tests"

    def gpu_entry(ctest_name: str, case: str) -> dict:
        return _entry(ctest_name, [str(gpu), f"--gtest_filter={case}", flag])

    return {
        "tests": [
            gpu_entry("Plain.Refuses", "Plain.Refuses"),
            gpu_entry("Plain.Validates", "Plain.Validates"),
            gpu_entry('Cases/Spin.Matches/"Singlet"', "Cases/Spin.Matches/Singlet"),
            gpu_entry('Cases/Spin.Matches/"Triplet"', "Cases/Spin.Matches/Triplet"),
            gpu_entry("Typed/0.Works<float>", "Typed/0.Works"),
            gpu_entry("Typed/1.Works<double>", "Typed/1.Works"),
            gpu_entry("Timing.Big", "Timing.Big"),
            gpu_entry("New.A", "New.A"),
            gpu_entry("New.B", "New.B"),
            _entry("Host.Cheap", [str(host), "--gtest_filter=Host.Cheap"], ("host",)),
            _entry("Host.Dear", [str(host), "--gtest_filter=Host.Dear"], ("host",)),
            _entry("example_rhf", ["cmake", "-P", "run.cmake"], ("host",)),
        ]
    }


def test_the_representative_set() -> None:
    keep, reps, flagged = gates.representative_set(FAMILIES)
    assert reps == {
        "Plain.Validates",
        "Cases/Spin.Matches/Singlet",
        "Typed/1.Works",
        "Host.Cheap",
    }
    # A flagged family runs whole; an exempt one not at all.
    assert flagged == {"New"}
    assert keep == reps | {"New.A", "New.B"}


def test_select_joins_on_gtests_names(tmp_path: Path) -> None:
    gpu, host = tmp_path / "gpu", tmp_path / "host"
    keep, _, _ = gates.representative_set(FAMILIES)
    text = "ctest chatter\n" + json.dumps(_listing(gpu, host))
    cut, dropped, counts = gates.select_listing(text, keep)
    assert dropped == [
        "Plain.Refuses",
        'Cases/Spin.Matches/"Triplet"',
        "Typed/0.Works<float>",
        "Timing.Big",
        "Host.Dear",
    ]
    assert [t["name"] for t in json.loads(cut)["tests"]] == [
        "Plain.Validates",
        'Cases/Spin.Matches/"Singlet"',
        "Typed/1.Works<double>",
        "New.A",
        "New.B",
        "Host.Cheap",
        "example_rhf",
    ]
    assert counts == {"entries": 12, "selected": 6, "not_gtest": 1}


def test_gtest_name() -> None:
    assert gates.gtest_name({"command": ["b", "--gtest_filter=A/B.c/1"]}) == "A/B.c/1"
    assert gates.gtest_name({"command": ["cmake", "-P", "x"]}) is None
    two = {"command": ["b", "--gtest_filter=A", "--gtest_filter=B"]}
    assert gates.gtest_name(two) is None


def test_skipped_cases() -> None:
    log = "\n".join(
        [
            "[  SKIPPED ] Cases/Spin.Matches/Singlet (0 ms)",
            "[  SKIPPED ] 1 test, listed below:",
            "[  SKIPPED ] Cases/Spin.Matches/Singlet",
            "1/3 Test #1: Host.Cheap .......................***Skipped   0.01 sec",
            "2/3 Test #2: Host.Dear ........................   Passed    0.01 sec",
            "3/3 Test #3: Unknown ..........................***Skipped   0.01 sec",
        ]
    )
    names = {"Host.Cheap": "Host.Cheap", "Host.Dear": "Host.Dear"}
    assert gates.skipped_cases(log, names) == {
        "Cases/Spin.Matches/Singlet",
        "Host.Cheap",
    }


def _stub_lister(root: Path, doc: dict | str, code: int = 0) -> None:
    (root / "devtools").mkdir(parents=True)
    out = doc if isinstance(doc, str) else json.dumps(doc)
    (root / "devtools" / "test_families.py").write_text(
        f"import sys\nsys.stdout.write({out!r})\nsys.exit({code})\n"
    )


def test_list_families_reads_the_listers_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_lister(tmp_path, FAMILIES)
    monkeypatch.setattr(gates, "REPO_ROOT", tmp_path)
    assert gates.list_families(tmp_path / "build") == FAMILIES


@pytest.mark.parametrize(
    ("doc", "code", "message"),
    [
        (dict(FAMILIES, errors=["x.cpp:3: bad"]), 1, "x.cpp:3: bad"),
        # #1125's error: a family with no fast member is a missing marker.
        (
            dict(FAMILIES, errors=["Foo: every timed member reaches it"]),
            1,
            "a marker is wrong or missing.*Foo: every timed member",
        ),
        ("", 2, "exited 2"),
        ("not json", 0, "exited 0"),
    ],
)
def test_list_families_refuses_an_incomplete_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, doc, code: int, message: str
) -> None:
    _stub_lister(tmp_path, doc, code)
    monkeypatch.setattr(gates, "REPO_ROOT", tmp_path)
    with pytest.raises(gates.SelectionError, match=message):
        gates.list_families(tmp_path / "build")


def test_tier_tests_runs_only_the_representatives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = tmp_path / "build"
    gpu = build / gates.GPU_BINS[2]
    host = build / gates.HOST_BINS[-1]
    gpu.parent.mkdir(parents=True)
    host.parent.mkdir(parents=True, exist_ok=True)
    gpu.touch()
    host.touch()
    listing = json.dumps(_listing(gpu, host))
    seen: dict = {}

    def fake_run(cmd, **kw):
        assert "--show-only=json-v1" in cmd
        return subprocess.CompletedProcess(cmd, 0, listing, "")

    def fake_phase(name, cmd, log, env=None):
        exclude = cmd[cmd.index("--exclude-from-file") + 1]
        seen["excluded"] = Path(exclude).read_text().splitlines()
        with log.open("a") as f:
            f.write("1/1 Test #1: Host.Cheap ......***Skipped   0.01 sec\n")
        return 0

    def fake_batch(tag, binary, cfg, log, *, env, cwd, echo_log):
        seen.setdefault("batches", []).append((binary, env["GTEST_FILTER"]))
        return 0

    monkeypatch.setattr(gates.subprocess, "run", fake_run)
    monkeypatch.setattr(gates, "run_phase", fake_phase)
    monkeypatch.setattr(gates, "run_gpu_batch", fake_batch)
    log = tmp_path / "pr.log"
    cfg = {"CTEST_JOBS": "1"}
    assert gates.tier_tests(cfg, "p", build, [], log, families=FAMILIES) == 0
    assert seen["batches"] == [
        (
            gpu,
            "Plain.Validates:Cases/Spin.Matches/Singlet:Typed/1.Works:New.A:New.B",
        )
    ]
    # Every batched entry, then every unselected one, is kept from ctest.
    assert sorted(seen["excluded"]) == sorted(
        [
            "Plain.Validates",
            'Cases/Spin.Matches/"Singlet"',
            "Typed/1.Works<double>",
            "New.A",
            "New.B",
            "Plain.Refuses",
            'Cases/Spin.Matches/"Triplet"',
            "Typed/0.Works<float>",
            "Timing.Big",
            "Host.Dear",
        ]
    )
    text = log.read_text()
    assert "6 of 12 ctest entries selected, against the lister's 4 representatives" in (
        text
    )
    assert "flagged, run whole: New" in text
    assert "skipped representative: Host.Cheap" in text


def test_tier_tests_without_families_runs_the_listing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = tmp_path / "build"
    gpu = build / gates.GPU_BINS[2]
    gpu.parent.mkdir(parents=True)
    gpu.touch()
    listing = json.dumps(_listing(gpu, build / "host"))
    filters = []
    monkeypatch.setattr(
        gates.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, listing, ""),
    )
    monkeypatch.setattr(gates, "run_phase", lambda *a, **kw: 0)
    monkeypatch.setattr(
        gates,
        "run_gpu_batch",
        lambda *a, env, **kw: filters.append(env["GTEST_FILTER"]) or 0,
    )
    assert gates.tier_tests({"CTEST_JOBS": "1"}, "p", build, [], tmp_path / "l") == 0
    assert filters[0].count(":") == 8


# --- push's representative set ----------------------------------------------


def test_push_filters_select_pr_s_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build = tmp_path / "build"
    gpu = build / gates.GPU_BINS[2]
    core = build / gates.GPU_BINS[0]
    listing = _listing(gpu, build / gates.HOST_BINS[-1])
    # Behind a launcher, the binary is still found.
    listing["tests"].append(
        _entry("Core.Wrapped", ["compute-sanitizer", str(core), "--gtest_filter=C.W"])
    )
    families = json.loads(json.dumps(FAMILIES))
    families["families"]["Core"] = {
        "members": ["C.W"],
        "representative": "C.W",
        "by": "cheapest",
    }
    monkeypatch.setattr(gates, "list_families", lambda build_dir: families)
    monkeypatch.setattr(
        gates.subprocess,
        "run",
        lambda cmd, **kw: subprocess.CompletedProcess(
            cmd, 0, "chatter\n" + json.dumps(listing), ""
        ),
    )
    log = tmp_path / "push.log"
    assert gates.push_filters("push", build, log) == [
        "C.W",
        "",
        "Plain.Validates:Cases/Spin.Matches/Singlet:Typed/1.Works:New.A:New.B",
    ]
    # The host and non-gtest entries are not the GPU smoke's.
    assert "6 of 10 GPU-binary cases selected, against the lister's 5" in (
        log.read_text()
    )


@pytest.mark.parametrize("fails", ["lister", "marker", "ctest"])
def test_push_filters_fall_back_to_the_whole_fast_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fails: str
) -> None:
    doc = dict(FAMILIES, errors=["x.cpp:3: bad"]) if fails == "marker" else FAMILIES
    _stub_lister(
        tmp_path, "" if fails == "lister" else doc, 2 if fails == "lister" else 0
    )
    monkeypatch.setattr(gates, "REPO_ROOT", tmp_path)
    real_run = subprocess.run

    def fake_run(cmd, **kw):
        if cmd[0] == "ctest":
            return subprocess.CompletedProcess(cmd, 1, "", "no tests")
        return real_run(cmd, **kw)

    monkeypatch.setattr(gates.subprocess, "run", fake_run)
    log = tmp_path / "push.log"
    assert gates.push_filters("push", tmp_path / "build", log) == [None] * len(
        gates.GPU_BINS
    )
    assert "push: the GPU smoke runs the whole fast set" in log.read_text()


@pytest.mark.parametrize("run_all", [False, True])
def test_cpp_push_runs_the_filters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, run_all: bool
) -> None:
    class NoCard:
        def __init__(self, *a) -> None:
            pass

        def __enter__(self) -> NoCard:
            return self

        def __exit__(self, *exc) -> None:
            pass

    batches = []
    monkeypatch.setattr(gates, "cpp_unavailable", lambda backend: None)
    monkeypatch.setattr(gates, "cpp_build", lambda *a, **kw: 0)
    monkeypatch.setattr(gates, "tree_rev", lambda: ("abc", ""))
    monkeypatch.setattr(gates, "tree_rev_header", lambda tag, log: None)
    monkeypatch.setattr(gates, "build_dir_of", lambda preset: tmp_path)
    monkeypatch.setattr(gates, "HOST_BINS", ())
    monkeypatch.setattr(gates, "Card", NoCard)
    monkeypatch.setattr(
        gates, "push_filters", lambda tag, build_dir, log: ["C.W", "", None]
    )
    monkeypatch.setattr(
        gates,
        "run_gpu_batch",
        lambda tag, binary, cfg, log, *, env, cwd: (
            batches.append((binary.name, env.get("GTEST_FILTER"))) or 0
        ),
    )
    cfg = {"SLOW_TIER_REPORTS": str(tmp_path), "GPU_HIP_PRESETS": "hip"}
    assert gates.cpp_push(cfg, "default", "8", run_all) == (0, "default")
    names = [Path(rel).name for rel in gates.GPU_BINS]
    if run_all:
        assert batches == [(n, None) for n in names]
    else:
        # A binary with nothing selected is not run.
        assert batches == [(names[0], "C.W"), (names[2], None)]


# --- config.sh ---------------------------------------------------------------


def test_config_comes_from_config_sh_and_env_wins() -> None:
    cfg = gates.read_config(_clean_env())
    assert set(gates.CONFIG_VARS) <= cfg.keys()
    assert cfg["FAST_TEST_ARGS"] == '-m "not slow"'
    assert gates.read_config(_clean_env(CTEST_JOBS="1"))["CTEST_JOBS"] == "1"


def test_backend_of_preset() -> None:
    cfg = {"GPU_HIP_PRESETS": "hip hip-asan ci-hip"}
    assert gates.backend_of_preset("hip", cfg) == "hip"
    assert gates.backend_of_preset("ci-hip", cfg) == "hip"
    assert gates.backend_of_preset("default", cfg) == "cuda"


# --- the card lock -----------------------------------------------------------


def test_card_lock_is_the_shell_tools_lock(tmp_path: Path) -> None:
    # lib.sh's gpu_card_held probes with flock(1); it must see test.py's hold.
    # GPU_LOCK_DIR is relative to the main checkout, in both.
    root = gates.read_config(_clean_env())["PRIMARY_ROOT"] or str(_ROOT)
    env = _clean_env(GPU_LOCK_DIR=os.path.relpath(tmp_path, root))
    cfg = gates.read_config(env)
    held = f'. "{gates.LIB}"; gpu_card_held cuda'

    def shell_sees_it_held() -> bool:
        proc = subprocess.run(["bash", "-c", held], env=env, check=False)
        return proc.returncode == 0

    card = gates.Card("cuda", cfg)
    assert card._take()
    try:
        assert shell_sees_it_held()
        assert not gates.Card("cuda", cfg)._take()
    finally:
        card.release()
    assert not shell_sees_it_held()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["cuda.0.lock", "cuda.1.lock"]


def test_card_vram_reads_the_named_card(monkeypatch: pytest.MonkeyPatch) -> None:
    # The backend must reach lib.sh's gpu_card_mem as its $1.
    seen = []

    def fake_lib(script: str, *args: str, **kw):
        seen.append((script, args))
        return subprocess.CompletedProcess([], 0, "42\n", "")

    monkeypatch.setattr(gates, "lib", fake_lib)
    card = gates.Card("hip", gates.read_config(_clean_env()))
    assert card.vram() == "42"
    script, args = seen[0]
    assert args == ("hip",) and '"$1"' in script


# --- the push gate's auto preset prefers a built tree -------------------------

_TWO_CARDS = {"GPU_AUTO_PRESETS": "cuda:default hip:hip"}


def _push_choice(monkeypatch: pytest.MonkeyPatch, built: set[str], live: set[str]):
    """push_auto_preset over two cards, `built` the presets with a tree and
    `live` those lib.sh would pick among: (its choice, the auto_preset calls
    it made)."""
    calls: list[str | None] = []

    def fake_auto(among: str | None = None) -> str:
        calls.append(among)
        entries = (among or _TWO_CARDS["GPU_AUTO_PRESETS"]).split()
        pick = [p for p in (e.split(":")[1] for e in entries) if p in live]
        if pick:
            return pick[0]
        return "" if among is not None else "default"

    monkeypatch.setattr(gates, "auto_preset", fake_auto)
    monkeypatch.setattr(gates, "say", lambda *a, **k: None)
    chosen = gates.push_auto_preset(_TWO_CARDS, built=lambda p: p in built)
    return chosen, calls


def test_push_auto_takes_the_one_built_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    # #1037's fresh worktree: `pr --preset default` built build/, and the
    # freer card was the HIP one. The push stays on the built tree.
    chosen, calls = _push_choice(monkeypatch, {"default"}, {"hip", "default"})
    assert chosen == "default"
    assert calls == ["cuda:default"]


def test_push_auto_is_unchanged_with_all_or_no_trees_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for built in (set(), {"default", "hip"}):
        chosen, calls = _push_choice(monkeypatch, built, {"hip"})
        assert (chosen, calls) == ("hip", [None]), built


def test_push_auto_leaves_a_built_tree_whose_card_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Only hip is built, but its card is gone: build the other, never skip.
    chosen, calls = _push_choice(monkeypatch, {"hip"}, {"default"})
    assert chosen == "default"
    assert calls == ["hip:hip", None]


def test_auto_preset_among_narrows_lib_sh(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = []

    def fake_lib(script: str, *args: str, **kw):
        among = kw["env"]["GPU_AUTO_PRESETS"]
        seen.append(among)
        if among == "hip:hip":
            err = "gpu: auto -> default (no live card: hip=absent)\n"
            return subprocess.CompletedProcess([], 0, "default\n", err)
        return subprocess.CompletedProcess([], 0, "hip\n", "")

    monkeypatch.setattr(gates, "lib", fake_lib)
    assert gates.auto_preset("cuda:default hip:hip") == "hip"
    # No live card among them: "", not lib.sh's fallback.
    assert gates.auto_preset("hip:hip") == ""
    assert seen == ["cuda:default hip:hip", "hip:hip"]


def test_a_build_stopped_part_way_is_not_built(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The host binaries link first, so one of them is no sign of a tree.
    monkeypatch.setattr(gates, "build_dir_of", lambda preset: tmp_path)
    for binary in gates.HOST_BINS:
        (tmp_path / binary).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / binary).touch()
    assert not gates.is_built("hip")
    for binary in gates.GPU_BINS:
        (tmp_path / binary).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / binary).touch()
    assert gates.is_built("hip")


def test_gpu_auto_presets_reaches_lib_sh_from_the_environment() -> None:
    # auto_preset(among) narrows through GPU_AUTO_PRESETS in the
    # environment, so config.sh must let the environment win.
    cfg = gates.read_config(_clean_env(GPU_AUTO_PRESETS="hip:hip"))
    assert cfg["GPU_AUTO_PRESETS"] == "hip:hip"
