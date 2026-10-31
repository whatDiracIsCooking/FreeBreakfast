#!/usr/bin/env python3
"""Regenerate the one-electron Hessian kernels under
src/integrals/one_electron/hessian/kernels/.

The kernels are FelizJueves's (deps/FelizJueves), emitted here and committed,
so a build never runs Python. Run this after bumping that submodule:

    tools/emit/one_electron_hessian_kernels.py

They are FelizJueves's `--hess-one-e-l 3` ladder (`felizjueves.emit`'s
`hess_one_e_classes(3)`, every pair class la <= lb <= 3): the three d2
one-electron families of FelizJueves's `hessian/kernel_h1e.py`, `d2s_`, `d2t_`
and `d2v_{la}{lb}`, each contracting its second-derivative integrals with a
matrix M into the 3N x 3N Hessian, H[A, k, B, l] += SUM_mn M_mn
d2X_mn/dR_{A,k}dR_{B,l}, for the overlap S, the kinetic energy T and the
nuclear attraction V (its operator's centre-nucleus and nucleus-nucleus terms
included). The `d2v` form is FelizJueves's own choice per class: low-stack
from la + lb = 4 (`kernel_h1e.h1e_lowstack`), unrolled below.

As tools/emit/one_electron_gradient_kernels.py does, each class is one
explicit specialization of its family's kernel template
(`overlap_hessian<la, lb>`, `kinetic_hessian<la, lb>`,
`potential_hessian<la, lb>`), and beside the units each family gets one
header, `<family>.h`, declaring the primary, every specialization and the
`[la][lb]` table the launcher indexes, null where `la > lb`. Only la <= lb is
written: free_breakfast.integrals.mmd swaps every shell pair of a basis into
that order.

The three families share one kernels/ directory, whose .clang-format turns
formatting off. `d2v`'s units include the shared `eval_boys.h`, which
tools/emit/one_electron_kernels.py and tools/emit/coulomb_kernels.py write to
src/integrals/shared/; this tool does not write it.

FelizJueves imports nothing outside the standard library, so this needs no
virtualenv.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deps/FelizJueves/src"))

from felizjueves.emit import gpu  # noqa: E402
from felizjueves.emit.__main__ import (  # noqa: E402
    HESS_ONE_E_L_MAX,
    hess_one_e_classes,
)
from felizjueves.emit.gpu import Specialization  # noqa: E402
from felizjueves.hessian.kernel_h1e import build_h1e_kernel  # noqa: E402
from felizjueves.performance.device import by_name  # noqa: E402
from felizjueves.spec import Target  # noqa: E402

# The kernels' L limit: f, FelizJueves's HESS_ONE_E_L_MAX. The launcher
# (hessian_launch.cu) reads each family's limit from its own table and the
# module refuses a class above through check_one_electron_build's
# kOneElectronMaxL, which must equal it.
MAX_L = 3
assert MAX_L == HESS_ONE_E_L_MAX, "FelizJueves's hessian ladder moved: revisit MAX_L"
# The gpu backend reads only the arch's wavefront, and these kernels have no
# shuffle, so their text names neither: any wave32 arch writes the same files.
ARCH = "sm_86"

# The family's kernel template, by FelizJueves's target.
FAMILIES = {
    "overlap_hessian": Target.S,
    "kinetic_hessian": Target.T,
    "potential_hessian": Target.V,
}

OUT = ROOT / "src/integrals/one_electron/hessian/kernels"


def main() -> int:
    arch = by_name(ARCH)
    OUT.mkdir(parents=True, exist_ok=True)
    classes = hess_one_e_classes(MAX_L)
    for family, target in FAMILIES.items():
        for cls in classes:
            kernel = build_h1e_kernel((target, *cls))
            spec = Specialization(family, cls)
            write(OUT / f"{kernel.name}.cu", gpu.emit(kernel, arch, spec))
        # Any class's kernel spells the primary: one family, one parameter block.
        write(OUT / spec.header, gpu.emit_template(kernel, arch, spec, classes))
    return 0


def write(path: Path, text: str) -> None:
    path.write_text(text)
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    sys.exit(main())
