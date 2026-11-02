"""devtools/lib.sh's CUDA version helpers, which gate a cloud VM's bring-up.

gpu-vm-setup.sh and doctor.sh compare the driver's CUDA ceiling against the
image's CUDA, and the card's compute capability against sm_80, through these.
A fake nvidia-smi stands in for the driver -- no GPU needed.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[2]
_LIB = _ROOT / "devtools" / "lib.sh"

# Without the caller's GIT_*: lib.sh runs git at source time, and a pre-push
# hook exports GIT_DIR.
_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def lib(snippet: str, **env: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", f'. "{_LIB}" && {snippet}'],
        cwd=_ROOT,
        env={**_ENV, **env},
        capture_output=True,
        text=True,
        check=False,
    )


@pytest.mark.parametrize(
    ("have", "want", "ok"),
    [
        ("13.0", "13.0", True),
        ("13.1", "13.0", True),
        ("12.8", "13.0", False),
        ("13.10", "13.2", True),  # numeric, not lexical
        ("9.0", "8.0", True),
        ("8.6", "8.0", True),
        ("7.5", "8.0", False),
        ("7.0", "8.0", False),
    ],
)
def test_version_ge(have: str, want: str, ok: bool) -> None:
    assert (lib(f"version_ge {have} {want}").returncode == 0) is ok


def test_cuda_image_version_reads_dockerfile() -> None:
    arg = re.search(
        r"^ARG CUDA_VERSION=(\S+)$",
        (_ROOT / "docker" / "Dockerfile.cuda").read_text(),
        re.MULTILINE,
    )
    assert arg is not None
    out = lib("cuda_image_version", CUDA_VERSION="").stdout.strip()
    assert out == arg.group(1).replace("-", ".")


def test_cuda_image_version_honours_build_arg() -> None:
    assert lib("cuda_image_version", CUDA_VERSION="12-9").stdout.strip() == "12.9"


def test_cuda_driver_ceiling(tmp_path: Path) -> None:
    smi = tmp_path / "nvidia-smi"
    smi.write_text(
        "#!/bin/sh\n"
        "echo '| NVIDIA-SMI 580.95.05   Driver Version: 580.95.05"
        "   CUDA Version: 13.0     |'\n"
    )
    smi.chmod(0o755)
    path = f"{tmp_path}:{os.environ['PATH']}"
    assert lib("cuda_driver_ceiling", PATH=path).stdout.strip() == "13.0"


def test_cuda_driver_ceiling_empty_without_driver(tmp_path: Path) -> None:
    smi = tmp_path / "nvidia-smi"
    smi.write_text("#!/bin/sh\necho 'NVIDIA-SMI has failed' >&2\nexit 9\n")
    smi.chmod(0o755)
    path = f"{tmp_path}:{os.environ['PATH']}"
    out = lib("cuda_driver_ceiling", PATH=path)
    assert out.returncode == 0
    assert out.stdout.strip() == ""
