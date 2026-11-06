"""devtools/test.py: argument parsing, push dispatch, the GPU batch split,
config.sh reading and the card lock. The watchdog has its own file,
test_gpu_batch_watchdog.py. Nothing here builds or needs a GPU."""

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
    assert not (args.fresh or args.clean or args.no_test or args.tidy)
    assert rest == []


def test_push_flags() -> None:
    args, _ = gates.parse(["push", "--dry-run", "-j", "8", "--preset", "default"])
    assert args.dry_run and args.jobs == "8" and args.preset == "default"


def test_push_refuses_passthrough() -> None:
    with pytest.raises(SystemExit):
        gates.parse(["push", "--", "-k", "x"])


def test_a_gate_is_required() -> None:
    with pytest.raises(SystemExit):
        gates.parse([])


def test_only_is_checked() -> None:
    with pytest.raises(SystemExit):
        gates.parse(["pr", "--only", "rust"])


# --- push dispatch -----------------------------------------------------------


@pytest.mark.parametrize(
    ("files", "expected"),
    [
        (["src/core/a.cppm"], (True, False)),
        (["src/x/k.cu", "src/x/k.cuh"], (True, False)),
        (["test/gtest/CMakeLists.txt"], (True, False)),
        (["CMakePresets.json"], (True, False)),
        (["cmake/Thing.cmake"], (True, False)),
        (["test/pytest/test_x.py"], (False, True)),
        (["devtools/test.py", "src/a.cpp"], (True, True)),
        (["docs/testing.md", ".pre-commit-config.yaml"], (False, False)),
        # A submodule bump is its gitlink path, which matches nothing; files
        # inside a submodule are its own repository's business.
        (["deps/WarpWraps"], (False, False)),
        (["deps/FelizJueves/tools/emit/x.py"], (False, False)),
        ([], (False, False)),
        (None, (True, True)),
    ],
)
def test_halves(files: list[str] | None, expected: tuple[bool, bool]) -> None:
    assert gates.halves(files) == expected


def test_changed_files_reads_pre_commits_refs() -> None:
    env = _clean_env(
        PRE_COMMIT="1", PRE_COMMIT_FROM_REF="HEAD~1", PRE_COMMIT_TO_REF="HEAD"
    )
    expected = subprocess.run(
        ["git", "-C", str(_ROOT), "diff", "--name-only", "HEAD~1...HEAD"],
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    assert gates.changed_files(env) == expected


def test_a_whole_history_push_runs_both_halves() -> None:
    # pre-commit sets no refs when the push includes the root commit.
    assert gates.changed_files(_clean_env(PRE_COMMIT="1")) is None


def test_dry_run_dispatch() -> None:
    proc = subprocess.run(
        [sys.executable, str(_TEST_PY), "push", "--dry-run"],
        env=_clean_env(
            PRE_COMMIT="1", PRE_COMMIT_FROM_REF="HEAD", PRE_COMMIT_TO_REF="HEAD"
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "0 changed file(s) -> C++ half: no, Python half: no" in proc.stdout


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
