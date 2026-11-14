"""devtools/kernel-compile-watchdog.sh: its summary survives a long list (#838),
and `--build-dir` times an already-configured build directory (#1387).

The summary used to take its slowest ten with `sort -gr | head`, and under the
script's pipefail a list longer than a pipe buffer killed sort with SIGPIPE:
a 397-kernel run exited 141 with no table. The first cases drive `--summarize`
with synthetic results files, so no toolchain and no compile is needed.

`--build-dir` is how rust/'s CMake project, which no root preset builds, is
timed. Its cases run the whole script against a fake build directory: a
`compile_commands.json` whose "compiles" are shell commands that succeed, fail
or overrun, and stub `ninja` and `cmake` first on PATH, the ninja one listing
the objects, the cmake one recording that it was called, which it never may
be: a build directory is taken as configured.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[2] / "devtools" / "kernel-compile-watchdog.sh"
)


def summarize(tmp_path: Path, lines: list[str]) -> subprocess.CompletedProcess[str]:
    results = tmp_path / "results"
    results.write_text("".join(f"{line}\n" for line in lines))
    return subprocess.run(
        ["bash", str(_SCRIPT), "--preset", "fake", "--summarize", str(results)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def many(n: int) -> list[str]:
    # Seconds in a scrambled order, so only a real sort puts the largest first.
    return [
        f"{(i * 7919) % n + 0.5:.1f} ok build/src/k/kernels/eri_{i:04d}.cu.o"
        for i in range(n)
    ]


def test_long_list_prints_the_slowest_ten_and_passes(tmp_path: Path) -> None:
    proc = summarize(tmp_path, many(1000))
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.splitlines()
    table = out[out.index("--- slowest 10 ---") + 1 :][:10]
    assert [float(row.split()[0].rstrip("s")) for row in table] == [
        999.5 - i for i in range(10)
    ]
    assert "PASS -- 1000 kernels" in proc.stdout
    assert "worst 999.5s" in proc.stdout


def test_long_list_with_a_failure_exits_one(tmp_path: Path) -> None:
    lines = many(1000)
    lines[500] = "301.0 TIMEOUT build/src/k/kernels/eri_slow.cu.o"
    lines[700] = "2.0 FAILED build/src/k/kernels/eri_broken.cu.o"
    proc = summarize(tmp_path, lines)
    assert proc.returncode == 1, proc.stderr
    assert "--- slowest 10 ---" in proc.stdout
    assert "FAIL -- 2 of 1000" in proc.stdout
    assert "TIMEOUT build/src/k/kernels/eri_slow.cu.o 301.0s" in proc.stdout
    assert "FAILED build/src/k/kernels/eri_broken.cu.o 2.0s" in proc.stdout


def test_short_list_prints_every_row(tmp_path: Path) -> None:
    proc = summarize(tmp_path, ["1.0 ok a.cu.o", "3.0 ok b.cu.o"])
    assert proc.returncode == 0, proc.stderr
    assert "PASS -- 2 kernels" in proc.stdout
    assert "worst 3.0s, b.cu.o" in proc.stdout


def test_empty_results_cannot_run(tmp_path: Path) -> None:
    proc = summarize(tmp_path, [])
    assert proc.returncode == 2
    assert "no results" in proc.stderr


def fake_build(tmp_path: Path, compiles: dict[str, str]) -> Path:
    """A configured build directory under `tmp_path/build` whose objects are
    `compiles`' keys, each "compiled" by its value, and stub `ninja` and
    `cmake` in `tmp_path/bin`."""
    build = tmp_path / "build"
    build.mkdir()
    (build / "build.ninja").write_text("# fake\n")
    entries = [
        {"directory": str(build), "command": cmd, "file": f"{obj}.cu", "output": obj}
        for obj, cmd in compiles.items()
    ]
    (build / "compile_commands.json").write_text(json.dumps(entries))
    targets = "".join(f"{obj}: CUDA_COMPILER__k_\n" for obj in compiles)
    (build / "targets").write_text(targets)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    ninja = bin_dir / "ninja"
    # `-C DIR -t targets all` lists the objects; `-t query` names no
    # prerequisite; any build succeeds.
    ninja.write_text(
        "#!/usr/bin/env bash\n"
        'if [ "$3" = -t ] && [ "$4" = targets ]; then cat "$2/targets"; fi\n'
        "exit 0\n"
    )
    cmake = bin_dir / "cmake"
    cmake.write_text(f'#!/usr/bin/env bash\necho "$@" >>{tmp_path}/cmake-called\n')
    for stub in (ninja, cmake):
        stub.chmod(0o755)
    return build


def watchdog(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """The script with `args`, run from `tmp_path` with the stubs first on
    PATH, so a relative `--build-dir build` is the fake one."""
    env = dict(os.environ, PATH=f"{tmp_path / 'bin'}:{os.environ['PATH']}")
    return subprocess.run(
        ["bash", str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        cwd=tmp_path,
        env=env,
    )


def test_build_dir_times_the_matching_kernels_and_passes(tmp_path: Path) -> None:
    fake_build(
        tmp_path,
        {
            "integrals/kernels/CMakeFiles/k.dir/s_00.cu.o": "true",
            "integrals/kernels/CMakeFiles/k.dir/s_11.cu.o": "true",
            # Not matched by the pattern: never run, or the run would fail.
            "recipes/kernels/CMakeFiles/r.dir/sort.cu.o": "exit 3",
        },
    )
    proc = watchdog(tmp_path, "--build-dir", "build", "integrals/kernels/")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "kernel-compile-watchdog: 2 kernels" in proc.stdout
    assert "PASS -- 2 kernels, all under 300s (build-dir=build," in proc.stdout
    assert "sort.cu.o" not in proc.stdout
    assert not (tmp_path / "cmake-called").exists(), "a build dir is not configured"


def test_build_dir_reports_a_failure_and_a_timeout(tmp_path: Path) -> None:
    fake_build(
        tmp_path,
        {
            "integrals/kernels/CMakeFiles/k.dir/ok.cu.o": "true",
            "integrals/kernels/CMakeFiles/k.dir/broken.cu.o": "echo nope; exit 3",
            "integrals/kernels/CMakeFiles/k.dir/slow.cu.o": "sleep 30",
        },
    )
    proc = watchdog(tmp_path, "--build-dir", str(tmp_path / "build"), "--limit", "1")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "FAIL -- 2 of 3 over 1s or broken (build-dir=" in proc.stdout
    assert "FAILED integrals/kernels/CMakeFiles/k.dir/broken.cu.o" in proc.stdout
    assert "TIMEOUT integrals/kernels/CMakeFiles/k.dir/slow.cu.o" in proc.stdout
    assert "nope" in proc.stdout, "a failure's output is shown"


def test_an_unconfigured_build_dir_cannot_run(tmp_path: Path) -> None:
    (tmp_path / "build").mkdir()
    (tmp_path / "bin").mkdir()
    proc = watchdog(tmp_path, "--build-dir", "build")
    assert proc.returncode == 2
    assert "has no build.ninja" in proc.stderr
    assert "cmake --preset cuda" in proc.stderr


def test_a_build_dir_and_a_preset_are_refused(tmp_path: Path) -> None:
    (tmp_path / "bin").mkdir()
    proc = watchdog(tmp_path, "--build-dir", "build", "--preset", "default")
    assert proc.returncode == 2
    assert "give one" in proc.stderr


def test_a_flag_without_its_value_cannot_run(tmp_path: Path) -> None:
    (tmp_path / "bin").mkdir()
    for flag in ("--build-dir", "--preset", "--limit"):
        proc = watchdog(tmp_path, flag)
        assert proc.returncode == 2, flag
        assert f"{flag} needs a value" in proc.stderr, flag
