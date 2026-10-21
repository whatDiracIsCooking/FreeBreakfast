#!/usr/bin/env python3
"""Regenerate the one-electron kernels under src/integrals/one_electron/,
and the mmd record bound under src/integrals/mmd/kernels/.

The kernels are FelizJueves's (deps/FelizJueves), emitted here and committed,
so a build never runs Python. Run this after bumping that submodule:

    tools/emit_one_electron_kernels.py

They are FelizJueves's `gpu` spelling, written as a `.cu` that both backends
compile as emitted: nvcc under CUDA and clang `-x hip` under HIP. Each file
includes FelizJueves's own `felizjueves/runtime.h` (deps/FelizJueves/include,
the `felizjueves::headers` target), which pulls in whichever vendor runtime
the compiling front end needs. Nothing is added to the text here.

Each class is written as one explicit specialization of the family's kernel
template, `overlap<la, lb>` (FelizJueves's `gpu.Specialization`), never as a
C name per class. Every unit's banner names its specialization. Beside the
units, each family's `kernels/` gets one header, `<family>.h`
(`gpu.emit_template`), which every unit includes by that name: the template's
primary (`template <uint8_t La, uint8_t Lb>`, defined nowhere), each class's
specialization declared, and `<family>_kernels`, the `[la][lb]` table the
launcher indexes, null where `la > lb`. Each unit is a source of the family's
device library, compiled on its own; the launcher includes only the header.

Only the classes `la <= lb` are written. The host pair builder
(free_breakfast.integrals.mmd) swaps every shell pair into that order, so a
`(1, 0)` kernel would be `(0, 1)`'s mirror and nothing would launch it.

One more family is not a one-electron integral but rides along because it
is per pair class, keyed and laid out the same way: `pair_bound<la, lb>`
(FelizJueves's Target.PAIR_BOUND), each record's pair-factor-free Schwarz
bound I = ao.pairs.pair_bound(zeta, la + lb). It is a field of every
free_breakfast.integrals.mmd record, computed once per basis after the zeta
sort, so it is written to src/integrals/mmd/kernels/pair_bound/; the J and K
screens read it as Ip (a bra) and Iq (a ket).

The device Boys routine the units with a fundamental include, `eval_boys.h`,
and the per-order `eval_boys_{M}.h` it includes for M in 0..20 (all of
`gpu.emit_boys()`), are written once, to src/integrals/shared/, by this tool
and tools/emit_coulomb_kernels.py alike: they are one text, so either leaves
them as the other would.

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
from felizjueves.emit.gpu import Specialization  # noqa: E402
from felizjueves.kernel_pair_bound import build_pair_bound_kernel  # noqa: E402
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
# turns formatting off without reaching the hand-written code beside it. The
# family's name is its kernel template's name, and its class list's prefix.
FAMILIES = {
    "overlap": build_s_kernel,
    "kinetic": build_t_kernel,
    "potential_en": build_v_kernel,
}

# The mmd section: the record bound, one directory of its own under mmd's
# kernels/ (whose .clang-format is the same DisableFormat). One record vector,
# the exponent, so no [n, 3] array to split.
MMD_FAMILIES = {"pair_bound": build_pair_bound_kernel}


def main() -> int:
    arch = by_name(ARCH)
    for family, build in FAMILIES.items():
        out = ROOT / "src/integrals/one_electron" / family / "kernels"
        emit_family(arch, family, build, out)
    for family, build in MMD_FAMILIES.items():
        emit_family(arch, family, build, ROOT / "src/integrals/mmd/kernels" / family)
    for name, text in gpu.emit_boys().items():
        write(ROOT / "src/integrals/shared" / name, text)
    return 0


def emit_family(arch, family: str, build, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    classes = [(la, lb) for la in range(MAX_L + 1) for lb in range(la, MAX_L + 1)]
    for cls in classes:
        kernel = build(cls)
        spec = Specialization(family, cls)
        write(out / f"{kernel.name}.cu", gpu.emit(kernel, arch, spec))
    # Any class's kernel spells the primary: one family, one parameter block.
    write(out / spec.header, gpu.emit_template(kernel, arch, spec, classes))


def write(path: Path, text: str) -> None:
    path.write_text(text)
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    sys.exit(main())
