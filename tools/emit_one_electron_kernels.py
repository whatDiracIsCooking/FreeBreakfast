#!/usr/bin/env python3
"""Regenerate the one-electron kernels under src/integrals/one_electron/.

The kernels are FelizJueves's (deps/FelizJueves), emitted here and committed,
so a build never runs Python. Run this after bumping that submodule:

    tools/emit_one_electron_kernels.py

Only the CUDA spelling is written, as a `.cu` that both backends compile: nvcc
under CUDA and clang `-x hip` under HIP. FelizJueves's HIP spelling differs
from it only by `#include <hip/hip_runtime.h>`, so instead every file gets
wwr's backend-neutral `#include <runtime.h>`, which pulls in whichever vendor
runtime header the compiling front end needs.

Each `v_*.cu` defines its own non-static `__device__ fj_boys` (FelizJueves's
wide Boys function, #266), so the six cannot share a translation unit as
emitted: the launcher that includes them gives each a namespace of its own
(src/integrals/one_electron/nuclear_launch.cu).

Only the classes `la <= lb` are written. The host pair builder
(free_breakfast.integrals.mmd) swaps every shell pair into that order, so a
`(1, 0)` kernel would be `(0, 1)`'s mirror and nothing would launch it.

FelizJueves imports nothing outside the standard library, so this needs no
virtualenv.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deps/FelizJueves/src"))

from felizjueves.device import by_name  # noqa: E402
from felizjueves.emit import cuda  # noqa: E402
from felizjueves.kernel_s import build_s_kernel  # noqa: E402
from felizjueves.kernel_t import build_t_kernel  # noqa: E402
from felizjueves.kernel_v import build_v_kernel  # noqa: E402

# The kernels' own L limit: d, as basis's kMaxAngularMomentum.
MAX_L = 2
# FelizJueves declares one NVIDIA arch. Its CUDA text does not depend on the
# arch beyond the banner, and the build compiles it for CMAKE_CUDA_ARCHITECTURES.
ARCH = "sm_86"

# One directory per family: the overlap S, the kinetic energy T and the nuclear
# attraction V. T + V is the core Hamiltonian.
FAMILIES = {
    "s_overlap": build_s_kernel,
    "t_kinetic": build_t_kernel,
    "v_nuclear": build_v_kernel,
}


def with_runtime_include(text: str) -> str:
    """Insert wwr's runtime.h below the generated banner."""
    banner, body = text.split("\n\n", 1)
    return (
        f"{banner}\n"
        f"// {Path(__file__).relative_to(ROOT)} added the include below.\n\n"
        f"#include <runtime.h>\n\n{body}"
    )


def main() -> int:
    arch = by_name(ARCH)
    for family, build in FAMILIES.items():
        out = ROOT / "src/integrals/one_electron" / family
        out.mkdir(parents=True, exist_ok=True)
        for la in range(MAX_L + 1):
            for lb in range(la, MAX_L + 1):
                kernel = build((la, lb))
                path = out / f"{kernel.name}.cu"
                path.write_text(with_runtime_include(cuda.emit(kernel, arch)))
                print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
