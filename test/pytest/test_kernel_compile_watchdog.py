"""devtools/kernel-compile-watchdog.sh: its summary survives a long list (#838).

The summary used to take its slowest ten with `sort -gr | head`, and under the
script's pipefail a list longer than a pipe buffer killed sort with SIGPIPE:
a 397-kernel run exited 141 with no table. These cases drive `--summarize`
with synthetic results files, so no toolchain and no compile is needed.
"""

from __future__ import annotations

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
