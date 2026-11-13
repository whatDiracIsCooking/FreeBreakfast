"""A build tree configured by a C++ compiler that is gone or has changed
(#1333): lib.sh's build_tree_compiler_drift, test.py's fresh reconfigure,
build.sh's, and doctor's warning. The trees are faked -- a CMakeCache.txt and
a CMakeFiles/<ver>/CMakeCXXCompiler.cmake -- and the compiler is a script that
reports a version, so nothing here needs CMake or clang.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_LIB = _ROOT / "devtools" / "lib.sh"
_spec = importlib.util.spec_from_file_location(
    "devtools_test_py", _ROOT / "devtools" / "test.py"
)
gates = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gates)


def _env(**extra: str) -> dict[str, str]:
    # Without the caller's GIT_*: lib.sh runs git at source time, and a
    # pre-push hook exports GIT_DIR (#694).
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("GIT_", "PRE_COMMIT"))
    }
    env.update(extra)
    return env


def _compiler(tmp_path: Path, version: str) -> Path:
    """A clang++ stand-in: -dumpversion prints `version`, like clang's."""
    cxx = tmp_path / "bin" / "clang++"
    cxx.parent.mkdir(exist_ok=True)
    cxx.write_text(
        f'#!/bin/sh\n[ "$1" = -dumpversion ] && {{ echo {version}; exit 0; }}\nexit 1\n'
    )
    cxx.chmod(0o755)
    return cxx


def _tree(
    tmp_path: Path, cxx: Path | str, version: str, modules_json: str = ""
) -> Path:
    tree = tmp_path / "build"
    rec = tree / "CMakeFiles" / "4.2.3"
    rec.mkdir(parents=True)
    cache = [
        "CMAKE_CACHE_MAJOR_VERSION:INTERNAL=4",
        "CMAKE_CACHE_MINOR_VERSION:INTERNAL=2",
        "CMAKE_CACHE_PATCH_VERSION:INTERNAL=3",
        "CMAKE_CXX_COMPILER:UNINITIALIZED=clang++",
    ]
    if modules_json:
        cache.append(f"CMAKE_CXX_STDLIB_MODULES_JSON:FILEPATH={modules_json}")
    (tree / "CMakeCache.txt").write_text("\n".join(cache) + "\n")
    (rec / "CMakeCXXCompiler.cmake").write_text(
        f'set(CMAKE_CXX_COMPILER "{cxx}")\n'
        'set(CMAKE_CXX_COMPILER_ARG1 "")\n'
        'set(CMAKE_CXX_COMPILER_ID "Clang")\n'
        f'set(CMAKE_CXX_COMPILER_VERSION "{version}")\n'
        'set(CMAKE_CXX_COMPILER_VERSION_INTERNAL "")\n'
    )
    return tree


def _drift(tree: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'. "{_LIB}" && build_tree_compiler_drift "$1"', "_", str(tree)],
        cwd=_ROOT,
        env=_env(),
        capture_output=True,
        text=True,
        check=False,
    )


# --- detection -----------------------------------------------------------------


def test_a_missing_compiler_is_drift(tmp_path: Path) -> None:
    gone = tmp_path / "llvm-20" / "bin" / "clang++"
    proc = _drift(_tree(tmp_path, gone, "20.1.8"))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"{gone} 20.1.8 -> gone ({gone} no longer exists)"
    assert gates.compiler_drift(tmp_path / "build") == proc.stdout.strip()


def test_the_same_path_at_another_version_is_drift(tmp_path: Path) -> None:
    # The clang 20 -> 23 case: /usr/bin/clang++ is still there.
    cxx = _compiler(tmp_path, "23.1.6")
    proc = _drift(_tree(tmp_path, cxx, "20.1.8"))
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == f"{cxx} 20.1.8 -> {cxx} 23.1.6"


def test_a_missing_module_manifest_is_drift(tmp_path: Path) -> None:
    cxx = _compiler(tmp_path, "23.1.6")
    gone = tmp_path / "llvm-20" / "lib" / "libc++.modules.json"
    proc = _drift(_tree(tmp_path, cxx, "23.1.6", str(gone)))
    assert proc.returncode == 0, proc.stderr
    assert str(gone) in proc.stdout


def test_a_matching_tree_is_not_drift(tmp_path: Path) -> None:
    cxx = _compiler(tmp_path, "23.1.6")
    manifest = tmp_path / "libc++.modules.json"
    manifest.write_text("{}")
    proc = _drift(_tree(tmp_path, cxx, "23.1.6", str(manifest)))
    assert proc.returncode == 1 and proc.stdout == ""
    assert gates.compiler_drift(tmp_path / "build") == ""


def test_an_unconfigured_tree_is_not_drift(tmp_path: Path) -> None:
    assert _drift(tmp_path / "build").returncode == 1
    (tmp_path / "build").mkdir()
    (tmp_path / "build" / "CMakeCache.txt").write_text("")
    assert _drift(tmp_path / "build").returncode == 1


# --- what the configure does with it --------------------------------------------


@pytest.mark.parametrize("stale", [True, False])
def test_test_py_reconfigures_a_drifted_tree_fresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stale: bool,
) -> None:
    tree = _tree(
        tmp_path,
        tmp_path / "gone" / "clang++" if stale else _compiler(tmp_path, "1.0"),
        "1.0",
    )
    monkeypatch.setattr(gates, "build_dir_of", lambda preset: tree)
    seen = []

    def fake_lib(script: str, *args: str, **kw):
        if script.startswith("BUILD_LOG"):
            seen.append(args)
            return subprocess.CompletedProcess([], 0, "", "")
        return gates_lib(script, *args, **kw)

    gates_lib = gates.lib
    monkeypatch.setattr(gates, "lib", fake_lib)
    log = tmp_path / "gate.log"
    assert gates.cpp_build("default", False, False, "8", log, quiet=True) == 0
    _, preset, fresh, *_ = seen[0]
    assert preset == "default"
    assert fresh == ("1" if stale else "0")
    out = capsys.readouterr().out
    if stale:
        line = next(
            ln for ln in out.splitlines() if ln.startswith("build: compiler changed")
        )
        assert line.endswith("; reconfiguring with --fresh")
        assert line in log.read_text()
    else:
        assert "compiler changed" not in out and not log.exists()


def test_build_sh_reconfigures_a_drifted_tree_fresh(tmp_path: Path) -> None:
    """build.sh run standalone: a stub cmake on PATH records each call."""
    tree = _tree(tmp_path, tmp_path / "gone" / "clang++", "20.1.8")
    bin_dir = tmp_path / "stub"
    bin_dir.mkdir()
    calls = tmp_path / "calls"
    for tool in ("cmake", "ninja"):
        (bin_dir / tool).write_text(f'#!/bin/sh\necho "$*" >> {calls}\n')
        (bin_dir / tool).chmod(0o755)
    proc = subprocess.run(
        [
            "bash",
            "-c",
            f'. "{_LIB}"; . "{_ROOT}/devtools/build.sh";'
            f' init_submodules() {{ :; }}; cpp_build_dir() {{ echo "{tree}"; }};'
            f' build_main --preset default --report "{tmp_path / "reports"}"',
        ],
        env=_env(PATH=f"{bin_dir}:{os.environ['PATH']}"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "build: compiler changed (" in proc.stdout
    assert calls.read_text().splitlines()[0] == "--preset default --fresh"


# --- doctor ----------------------------------------------------------------------


def _doctor_build_trees(tmp_path: Path, place) -> str:
    """doctor's `build trees` section over a throwaway repository, after
    `place(repo)` has laid out its trees. A stub cmake stands in for the
    toolchain."""
    repo = tmp_path / "repo"
    (repo / "devtools").mkdir(parents=True)
    for name in ("doctor.sh", "lib.sh", "config.sh"):
        (repo / "devtools" / name).write_bytes((_ROOT / "devtools" / name).read_bytes())
    env = _env(
        GIT_AUTHOR_NAME="t",
        GIT_AUTHOR_EMAIL="t@example.invalid",
        GIT_COMMITTER_NAME="t",
        GIT_COMMITTER_EMAIL="t@example.invalid",
        GIT_CONFIG_GLOBAL="/dev/null",
        GIT_CONFIG_NOSYSTEM="1",
    )
    git = ["git", "-C", str(repo)]
    subprocess.run([*git, "init", "-q", "-b", "main"], env=env, check=True)
    subprocess.run([*git, "add", "-A"], env=env, check=True)
    subprocess.run([*git, "commit", "-qm", "init"], env=env, check=True)
    place(repo)
    bin_dir = tmp_path / "stub"
    bin_dir.mkdir()
    (bin_dir / "cmake").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "cmake").chmod(0o755)
    proc = subprocess.run(
        ["bash", str(repo / "devtools" / "doctor.sh")],
        cwd=repo,
        env={**env, "PATH": f"{bin_dir}:{os.environ['PATH']}"},
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.split("\nbuild trees\n", 1)[1].split("\nnightly\n", 1)[0]


def _move_tree(tmp_path: Path, name: str, cxx_version: str, version: str, to: Path):
    """A tree recorded at `version`, by a compiler now at `cxx_version`,
    moved to `to`."""
    where = tmp_path / name
    where.mkdir()
    tree = _tree(where, _compiler(where, cxx_version), version)
    to.parent.mkdir(parents=True, exist_ok=True)
    tree.rename(to)


def test_doctor_names_a_drifted_tree(tmp_path: Path) -> None:
    """One stale build*/ tree and one sound one."""
    gone = tmp_path / "repo" / "gone" / "clang++"

    def place(repo: Path) -> None:
        _tree(repo, gone, "20.1.8")
        _move_tree(tmp_path, "sound", "23.1.6", "23.1.6", repo / "build-hip")

    section = _doctor_build_trees(tmp_path, place)
    warns = [ln for ln in section.splitlines() if "[warn]" in ln]
    assert len(warns) == 1, section
    assert f"build: configured by {gone} 20.1.8 -> gone" in warns[0]
    assert "build-hip" not in section


@pytest.mark.parametrize("stale", [True, False])
def test_doctor_scans_rust_build_trees(tmp_path: Path, stale: bool) -> None:
    """rust/build-cuda, the rust/ CMake project's tree (#1377): a compiler
    that has moved from 20.1.8 to 23.1.6 under it warns, one still at
    its recorded version does not."""
    recorded = "20.1.8" if stale else "23.1.6"

    def place(repo: Path) -> None:
        _move_tree(tmp_path, "rust", "23.1.6", recorded, repo / "rust" / "build-cuda")

    section = _doctor_build_trees(tmp_path, place)
    warns = [ln for ln in section.splitlines() if "[warn]" in ln]
    if stale:
        assert len(warns) == 1, section
        cxx = tmp_path / "rust" / "bin" / "clang++"
        assert (
            f"rust/build-cuda: configured by {cxx} 20.1.8 -> {cxx} 23.1.6" in warns[0]
        ), section
    else:
        assert warns == [], section
        assert "1 build tree(s), none configured by a changed compiler" in section
