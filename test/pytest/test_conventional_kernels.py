"""tools/emit/conventional_kernels.py: the conventional ERI's units, which
the C++ tree does not have, emitted again under another root and byte for
byte what rust/integrals/kernels/eri/conventional/kernels/ (packed) and
eri/conventional/dense/kernels/ (dense) carry. With no
C++ twin for rust/'s device_check to compare against, this is the check that
they are the generator's (rust/docs/rules.md).

Slow: the emit is ~30 s of FelizJueves on one core. It needs only the
FelizJueves submodule, which imports with the standard library alone.
"""

from __future__ import annotations

import filecmp
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _ROOT / "tools" / "emit" / "conventional_kernels.py"
# Each directory, and the units the tool writes there: the packed 21
# classes through (dd|dd), their header, eri_kperm and eri_task_threads.h
# (both forms' threads per task); the dense 81 ordered
# classes in each of two layouts and each layout's header.
_DIRS = {
    Path("eri/conventional/kernels"): 21 + 1 + 1 + 1,
    Path("eri/conventional/dense/kernels"): 2 * (81 + 1),
}


@pytest.mark.slow
@pytest.mark.skipif(
    not (_ROOT / "deps/FelizJueves/src/felizjueves").is_dir(),
    reason="needs the deps/FelizJueves submodule checked out",
)
def test_out_root_matches_rust(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(_TOOL), "--out-root", str(tmp_path)],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    for rel, count in _DIRS.items():
        emitted = sorted(p for p in (tmp_path / rel).iterdir() if p.is_file())
        assert len(emitted) == count, rel
        committed = _ROOT / "rust/integrals/kernels" / rel
        carried = sorted(
            p.name
            for p in committed.iterdir()
            if p.is_file() and p.name != ".clang-format"
        )
        assert carried == [p.name for p in emitted], (
            f"{rel}: a unit added or left behind"
        )
        for path in emitted:
            twin = committed / path.name
            assert filecmp.cmp(path, twin, shallow=False), f"{twin} differs"
