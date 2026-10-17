#!/usr/bin/env python3
"""Regenerate the one-electron kernels under src/integrals/one_electron/.

The kernels are FelizJueves's (deps/FelizJueves), emitted here and committed,
so a build never runs Python. Run this after bumping that submodule:

    tools/emit_one_electron_kernels.py

They are FelizJueves's `gpu` spelling, written as a `.cu` that both backends
compile as emitted: nvcc under CUDA and clang `-x hip` under HIP. Each file
includes FelizJueves's own `felizjueves/runtime.h` (deps/FelizJueves/include,
the `felizjueves::headers` target), which pulls in whichever vendor runtime
the compiling front end needs. Nothing is added to the text here.

Each `v_*.cu` defines its own non-static `__device__ fj_boys` (FelizJueves's
wide Boys function, #266), so the six cannot share a translation unit as
emitted: the launcher that includes them gives each a namespace of its own
(src/integrals/one_electron/potential_en/potential_en_launch.cu).

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
from felizjueves.emit import gpu  # noqa: E402
from felizjueves.kernel_s import build_s_kernel  # noqa: E402
from felizjueves.kernel_t import build_t_kernel  # noqa: E402
from felizjueves.kernel_v import build_v_kernel  # noqa: E402

# The kernels' L limit: d. It must equal kMaxAngularMomentum
# (src/basis/angular_momentum.h), the repo's one limit: every family's launcher
# sizes its kernel table by it and names one kernel per la <= lb <= it.
MAX_L = 2
# The gpu backend reads only the arch's wavefront, and these kernels have no
# shuffle, so their text names neither: any wave32 arch writes the same files.
ARCH = "sm_86"

# One directory per family: the overlap S, the kinetic energy T and the
# electron-nuclear potential V (potential_en). T + V is the core Hamiltonian.
# The kernels go in the family's kernels/ subdirectory, whose .clang-format
# turns formatting off without reaching the hand-written code beside it.
FAMILIES = {
    "overlap": build_s_kernel,
    "kinetic": build_t_kernel,
    "potential_en": build_v_kernel,
}


def main() -> int:
    arch = by_name(ARCH)
    for family, build in FAMILIES.items():
        out = ROOT / "src/integrals/one_electron" / family / "kernels"
        out.mkdir(parents=True, exist_ok=True)
        for la in range(MAX_L + 1):
            for lb in range(la, MAX_L + 1):
                kernel = build((la, lb))
                path = out / f"{kernel.name}.cu"
                path.write_text(gpu.emit(kernel, arch))
                print(path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
