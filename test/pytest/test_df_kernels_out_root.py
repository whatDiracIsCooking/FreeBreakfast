"""tools/emit/df_kernels.py --out-root: the same tree under another root, byte
for byte what src/integrals/ and rust/integrals/kernels/ carry (#1524).

Slow: the emit is ~12 s of FelizJueves on one core. It needs only the
FelizJueves submodule, which imports with the standard library alone.
"""

from __future__ import annotations

import filecmp
import subprocess
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_TOOL = _ROOT / "tools" / "emit" / "df_kernels.py"
_DF = Path("eri/density_fitting")


@pytest.mark.slow
@pytest.mark.skipif(
    not (_ROOT / "deps/FelizJueves/src/felizjueves").is_dir(),
    reason="needs the deps/FelizJueves submodule checked out",
)
def test_out_root_matches_both_trees(tmp_path: Path) -> None:
    proc = subprocess.run(
        [sys.executable, str(_TOOL), "--out-root", str(tmp_path)],
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    emitted = sorted(p for p in (tmp_path / _DF).rglob("*") if p.is_file())
    # metric's 28 classes and eri3's 70, each with its family header.
    assert len(emitted) == 28 + 1 + 70 + 1
    # Nothing written outside the family directories.
    assert {p.relative_to(tmp_path).parts[:4] for p in emitted} == {
        ("eri", "density_fitting", "metric", "kernels"),
        ("eri", "density_fitting", "three_center", "kernels"),
    }
    for tree in (_ROOT / "src/integrals", _ROOT / "rust/integrals/kernels"):
        for path in emitted:
            twin = tree / path.relative_to(tmp_path)
            assert filecmp.cmp(path, twin, shallow=False), f"{twin} differs"
