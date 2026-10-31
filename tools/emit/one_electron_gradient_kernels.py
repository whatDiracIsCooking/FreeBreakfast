#!/usr/bin/env python3
"""Regenerate the one-electron gradient kernels under
src/integrals/one_electron/gradient/kernels/.

The kernels are FelizJueves's (deps/FelizJueves), emitted here and committed,
so a build never runs Python. Run this after bumping that submodule:

    tools/emit/one_electron_gradient_kernels.py

They are FelizJueves's `--grad-l 3` ladder (`felizjueves.emit`'s
`grad_classes(3)`, every pair class la <= lb <= 3), less its `gjab_` units:
those are the Coulomb gradient's, emitted with the Coulomb kernels by
tools/emit/coulomb_kernels.py. What is left is the three d1 one-electron
families of FelizJueves's `gradient/kernel_g1e.py`, `gs_`, `gt_` and
`gv_{la}{lb}`, each contracting its derivative integrals with a matrix M
into a per-atom gradient, G[A, k] += SUM_mn M_mn dX_mn/dR_{A,k}, for the
overlap S, the kinetic energy T and the nuclear attraction V (its
Hellmann-Feynman term on every nucleus included). The `gv` form is
FelizJueves's own choice per class: low-stack from la + lb = 5
(`kernel_g1e.g1e_lowstack`), unrolled below.

As tools/emit/one_electron_kernels.py does, each class is one explicit
specialization of its family's kernel template (`overlap_gradient<la, lb>`,
`kinetic_gradient<la, lb>`, `potential_gradient<la, lb>`), and beside the
units each family gets one header, `<family>.h`, declaring the primary,
every specialization and the `[la][lb]` table the launcher indexes, null
where `la > lb`. Only la <= lb is written: free_breakfast.integrals.mmd
swaps every shell pair of a basis into that order, and the kernels' weight
M[a, b] + M[b, a] is the same either way round.

The three families share one kernels/ directory, whose .clang-format turns
formatting off. `gv`'s units include the shared `eval_boys.h`, which
tools/emit/one_electron_kernels.py and tools/emit/coulomb_kernels.py write
to src/integrals/shared/; this tool does not write it.

Beside them, the derivative MATRICES of FelizJueves's
`hessian/kernel_ip1e.py` (its `--ip-one-e-l 3` ladder), the inputs an
analytic Hessian's CPHF right-hand sides need as matrices rather than a
contraction: `ips_`, `ipt_` and `ipv_{la}{lb}`, X1[k, m, n] += dX_mn/dA_k
with A the centre of m, all atoms in one launch, for S, T and V (every
nucleus's operator held fixed); and `iprv_{la}{lb}`, V's operator term at
the one nucleus `nuc`, XC[k, m, n] += dV^C_mn/dC_k. Their templates are
`overlap_ip1`, `kinetic_ip1`, `potential_ip1` and `potential_rinv`, the
names of FelizJueves's host references (`gradient.host`).

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
    GRAD_L_MAX,
    IP_ONE_E_L_MAX,
    grad_classes,
)
from felizjueves.emit.gpu import Specialization  # noqa: E402
from felizjueves.gradient.kernel_g1e import build_g1e_kernel  # noqa: E402
from felizjueves.hessian.kernel_ip1e import build_ip1e_kernel  # noqa: E402
from felizjueves.performance.device import by_name  # noqa: E402
from felizjueves.spec import Target  # noqa: E402

# The kernels' L limit: f, FelizJueves's GRAD_L_MAX. kGradientMaxL is not
# written separately: the launcher (gradient_launch.cu) reads each family's
# limit from its own table and the module refuses a class above through
# check_one_electron_build's kOneElectronMaxL, which must equal it.
MAX_L = 3
assert MAX_L == GRAD_L_MAX, "FelizJueves's gradient ladder moved: revisit MAX_L"
assert MAX_L == IP_ONE_E_L_MAX, "FelizJueves's ip1 ladder moved: revisit MAX_L"
# The gpu backend reads only the arch's wavefront, and these kernels have no
# shuffle, so their text names neither: any wave32 arch writes the same files.
ARCH = "sm_86"

# The family's kernel template, by FelizJueves's target. `gjab_` is not here.
FAMILIES = {
    "overlap_gradient": Target.S,
    "kinetic_gradient": Target.T,
    "potential_gradient": Target.V,
}

# The derivative-matrix family's kernel template, by FelizJueves's target and
# matrix form (kernel_ip1e.IP1E).
MATRIX_FAMILIES = {
    "overlap_ip1": (Target.S, "ip1"),
    "kinetic_ip1": (Target.T, "ip1"),
    "potential_ip1": (Target.V, "ip1"),
    "potential_rinv": (Target.V, "rinv"),
}

OUT = ROOT / "src/integrals/one_electron/gradient/kernels"


def main() -> int:
    arch = by_name(ARCH)
    OUT.mkdir(parents=True, exist_ok=True)
    classes = grad_classes(MAX_L)
    for family, target in FAMILIES.items():
        for cls in classes:
            kernel = build_g1e_kernel((target, *cls))
            spec = Specialization(family, cls)
            write(OUT / f"{kernel.name}.cu", gpu.emit(kernel, arch, spec))
        # Any class's kernel spells the primary: one family, one parameter block.
        write(OUT / spec.header, gpu.emit_template(kernel, arch, spec, classes))
    for family, (target, form) in MATRIX_FAMILIES.items():
        for cls in classes:
            kernel = build_ip1e_kernel((target, *cls, form))
            spec = Specialization(family, cls)
            write(OUT / f"{kernel.name}.cu", gpu.emit(kernel, arch, spec))
        write(OUT / spec.header, gpu.emit_template(kernel, arch, spec, classes))
    return 0


def write(path: Path, text: str) -> None:
    path.write_text(text)
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    sys.exit(main())
