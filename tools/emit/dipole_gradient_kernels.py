#!/usr/bin/env python3
"""Regenerate the dipole gradient kernels under
src/integrals/one_electron/dipole_gradient/kernels/.

The kernels are FelizJueves's (deps/FelizJueves), emitted here and committed,
so a build never runs Python. Run this after bumping that submodule:

    tools/emit/dipole_gradient_kernels.py

They are the dipole half of FelizJueves's `--grad-mu-l 3` ladder
(`felizjueves.emit`'s `grad_mu_classes(3)`, every pair class la <= lb <= 3),
`gradient/kernel_gmu.py`'s `gmu_{la}{lb}`: the dipole integrals'
nuclear derivative contracted with a matrix M into a per-component,
per-atom gradient,

    G[c, A, k] += q SUM_mn M_mn d<m| (r - C)_c |n> / dR_{A,k},

c = x, y, z, about a fixed origin C (no operator term), with C and the charge
q runtime scalars. The quadrupole's `gquad_` units of the same ladder are not
written: nothing in the tree reads them.

As tools/emit/one_electron_gradient_kernels.py does, each class is one
explicit specialization of the kernel template `dipole_gradient<la, lb>`,
and beside the units one header, `dipole_gradient.h`, declares the primary,
every specialization and the `[la][lb]` table the launcher indexes, null where
`la > lb`. Only la <= lb is written: free_breakfast.integrals.mmd swaps every
shell pair of a basis into that order, and the kernels' weight
M[a, b] + M[b, a] is the same either way round.

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
    GRAD_MU_L_MAX,
    grad_mu_classes,
)
from felizjueves.emit.gpu import Specialization  # noqa: E402
from felizjueves.gradient.kernel_gmu import build_gmu_kernel  # noqa: E402
from felizjueves.performance.device import by_name  # noqa: E402
from felizjueves.spec import Target  # noqa: E402

# The kernels' L limit: f, FelizJueves's GRAD_MU_L_MAX. The launcher
# (dipole_gradient_launch.cu) reads it from the table and the module refuses a
# class above through check_one_electron_build's kOneElectronMaxL, which must
# equal it.
MAX_L = 3
assert MAX_L == GRAD_MU_L_MAX, (
    "FelizJueves's moment-gradient ladder moved: revisit MAX_L"
)
# The gpu backend reads only the arch's wavefront, and these kernels have no
# shuffle, so their text names neither: any wave32 arch writes the same files.
ARCH = "sm_86"
FAMILY = "dipole_gradient"

OUT = ROOT / "src/integrals/one_electron/dipole_gradient/kernels"


def main() -> int:
    arch = by_name(ARCH)
    OUT.mkdir(parents=True, exist_ok=True)
    classes = grad_mu_classes(MAX_L)
    for cls in classes:
        kernel = build_gmu_kernel((Target.DIPOLE, *cls))
        spec = Specialization(FAMILY, cls)
        write(OUT / f"{kernel.name}.cu", gpu.emit(kernel, arch, spec))
    # Any class's kernel spells the primary: one family, one parameter block.
    write(OUT / spec.header, gpu.emit_template(kernel, arch, spec, classes))
    return 0


def write(path: Path, text: str) -> None:
    path.write_text(text)
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    sys.exit(main())
