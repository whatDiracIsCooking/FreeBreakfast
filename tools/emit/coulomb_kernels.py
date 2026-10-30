#!/usr/bin/env python3
"""Regenerate the Coulomb-build kernels under
src/integrals/eri/integral_direct/coulomb/.

The kernels are FelizJueves's (deps/FelizJueves), emitted here and committed,
so a build never runs Python. Run this after bumping that submodule:

    tools/emit/coulomb_kernels.py

They are the three generated steps of FelizJueves's McMurchie-Davidson J
build (its docs/implementation/j_engine.md, "The J chain as device code"),
one kernel template per family. The record bound its screen reads,
`pair_bound<la, lb>`, is not here: it is a field of every mmd record, computed
once per basis, so tools/emit/one_electron_kernels.py writes it, to
src/integrals/mmd/kernels/pair_bound/.

- `dq<lc, ld>`, the ket pre-contraction D_q = w SUM_cd D_cd s E^cd_q and
  Dmax, reading the AO density in place (`density="ao"`: Dao[off_c + i,
  off_d + j], so no gather and no transposed copy for a swapped pair);
- `e_hermite<lp, lq>`, the [p|q] kernel J_p += SUM_q [p|q] D_q, one per
  Hermite class, its write to J_p atomic (`ir.passes.atomize`) because every
  ket class of a bra class adds into the same J_p;
- `jab<la, lb>`, the bra back-transform and scatter of J_p into J_mu_nu.

And the two its nuclear gradient adds (FelizJueves's docs/implementation/
gradient.md, "The kernels"; the `gjab_` half of its `--grad-l 3` ladder, whose
`gs_`, `gt_` and `gv_` are tools/emit/one_electron_gradient_kernels.py's):

- the raised Hermite row, `e_hermite<2 MAX_L + 1, lq>` for every lq <= 2
  MAX_L: the d0 [p|q] of `kernel_gjab.gjab_pq_spec`'s class, the bra raised
  by one, whose J_p at order la + lb + 1 the gradient reads. The same
  builder and passes as every other `e_hermite` class, so the same template
  and table, one row longer;
- `gjab<la, lb>`, the bra derivative of E_J contracted with the density into
  a per-atom G (`kernel_gjab`), in FelizJueves's default form per class
  (low-stack from la + lb = 5, `kernel_gjab.gjab_lowstack`), written to
  coulomb/gradient/.

Every [n, 3] record vector is split into three flat arrays (`ir.passes.
split_axes`) named as free_breakfast.integrals.mmd names its fields: pa_*,
pb_*, and p_* / q_* for the [p|q] bra and ket centres. D_q is stored q-major,
D[n_q, n_rec] (`ir.passes.transpose`), in both kernels that touch it: dq writes
it and e_hermite reads it, its lanes consecutive kets, so a warp's read is
contiguous.

They are FelizJueves's `gpu` spelling, compiled as emitted by both backends:
nvcc under CUDA and clang `-x hip` under HIP. `e_hermite` is emitted
`any_wavefront`: its shuffles and its block-sized shared J_p accumulator read
FJ_WAVEFRONT, which deps/CMakeLists.txt sets from WWR_WARP_SIZE, so one text
serves a 32- and a 64-lane build; its header's `e_hermite_block_dim` is the
block to launch it with, {1, FJ_WAVEFRONT}. `dq` and `jab`
have no shuffle, so their text has no width at all. Nothing is added to the text here.

As tools/emit/one_electron_kernels.py does, each class is one explicit
specialization of its family's template, written to the `kernels/` of the
family's module (coulomb/dq/, coulomb/e_hermite/, coulomb/j_coulomb/ for
jab, and coulomb/gradient/ for gjab), which also gets one header,
`<family>.h`, every unit includes: the primary, each class's declaration and
`<family>_kernels`, the table a launcher indexes. Pair
classes are written `la <= lb` only, for the reason that tool gives; the
Hermite classes are every (lp, lq) with both at most 2 MAX_L, and the raised
row lp = 2 MAX_L + 1 (the lower raised classes are energy classes already), so
the table is [2 MAX_L + 2][2 MAX_L + 1].

The device Boys routine the units with a fundamental include, `eval_boys.h`,
and the per-order `eval_boys_{M}.h` it includes for M in 0..20 (all of
`gpu.emit_boys()`), are written once, to src/integrals/shared/, by this tool
and tools/emit/one_electron_kernels.py alike: they are one text, so either leaves
them as the other would.

FelizJueves needs only the standard library to import, so this needs no
virtualenv.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deps/FelizJueves/src"))

from felizjueves.emit import gpu  # noqa: E402
from felizjueves.emit.gpu import WAVEFRONT, Specialization  # noqa: E402
from felizjueves.eri.j_coulomb.kernel_dq import build_dq_kernel  # noqa: E402
from felizjueves.eri.j_coulomb.kernel_jab import build_jab_kernel  # noqa: E402
from felizjueves.gradient.kernel_gjab import build_gjab_kernel  # noqa: E402
from felizjueves.ir.passes import atomize, split_axes, transpose  # noqa: E402
from felizjueves.mmd.kernel import build_pq_kernel  # noqa: E402
from felizjueves.performance.device import by_name  # noqa: E402
from felizjueves.spec import Operator, Spec, Target  # noqa: E402

# The kernels' L limit: f. It must equal kCoulombMaxMomentum
# (coulomb/pair_class/pair_class_bridge.h), not the tree-wide
# kMaxAngularMomentum; each family's launcher asserts its table against it. At
# (ff) FelizJueves's build_dq_kernel and build_jab_kernel pick their low-stack
# text on their own (la + lb >= 6).
MAX_L = 3
# With any_wavefront and no shuffle elsewhere, the arch fixes nothing in the
# text; the gpu backend still takes one.
ARCH = "sm_86"

OUT = ROOT / "src/integrals/eri/integral_direct/coulomb"
SHARED = ROOT / "src/integrals/shared"

# The record vectors, as mmd names them.
PAIR_AXES = {"XPA": ("pa_x", "pa_y", "pa_z"), "XPB": ("pb_x", "pb_y", "pb_z")}
PQ_AXES = {"P": ("p_x", "p_y", "p_z"), "Q": ("q_x", "q_y", "q_z")}
# D_q, q-major in dq and e_hermite alike.
Q_MAJOR = ("D",)


def pair_classes() -> list[tuple[int, int]]:
    return [(la, lb) for la in range(MAX_L + 1) for lb in range(la, MAX_L + 1)]


def hermite_classes() -> list[tuple[int, int]]:
    # Every energy class, then the gradient's raised row: an (ff) bra raised
    # by one is lp = 2 MAX_L + 1, against every ket.
    top = 2 * MAX_L
    energy = [(lp, lq) for lp in range(top + 1) for lq in range(top + 1)]
    return energy + [(top + 1, lq) for lq in range(top + 1)]


def dq(cls: tuple[int, int]):
    lc, ld = cls
    k = build_dq_kernel(cls, density="ao", name=f"dq_{lc}{ld}")
    return transpose(split_axes(k, PAIR_AXES), Q_MAJOR)


def jab(cls: tuple[int, int]):
    la, lb = cls
    return split_axes(build_jab_kernel(cls, name=f"jab_{la}{lb}"), PAIR_AXES)


def gjab(cls: tuple[int, int]):
    return split_axes(build_gjab_kernel(cls), PAIR_AXES)


def e_hermite(cls: tuple[int, int]):
    # Any shell quartet with these Hermite orders writes the same text; this
    # one splits each order as FelizJueves's own pq_spec does.
    lp, lq = cls
    am = ((lp + 1) // 2, lp // 2, (lq + 1) // 2, lq // 2)
    k = build_pq_kernel(
        Spec(Operator.COULOMB, 0, am, Target.J), name=f"e_hermite_{lp}{lq}"
    )
    return transpose(split_axes(atomize(k), PQ_AXES), Q_MAJOR)


# family -> (its module's directory, its classes, its builder, the emitter's
# options); the kernels go in that directory's kernels/.
FAMILIES = {
    "dq": ("dq", pair_classes, dq, {}),
    "e_hermite": (
        "e_hermite",
        hermite_classes,
        e_hermite,
        {"block_dim": (1, WAVEFRONT), "any_wavefront": True},
    ),
    "jab": ("j_coulomb", pair_classes, jab, {}),
    "gjab": ("gradient", pair_classes, gjab, {}),
}


def main() -> int:
    arch = by_name(ARCH)
    for family, (directory, classes_of, build, options) in FAMILIES.items():
        out = OUT / directory / "kernels"
        out.mkdir(parents=True, exist_ok=True)
        classes = classes_of()
        for cls in classes:
            kernel = build(cls)
            spec = Specialization(family, cls)
            write(out / f"{kernel.name}.cu", gpu.emit(kernel, arch, spec, **options))
        # Any class's kernel spells the primary: one family, one parameter block.
        header = gpu.emit_template(kernel, arch, spec, classes, **options)
        write(out / spec.header, header)
    for name, text in gpu.emit_boys().items():
        write(SHARED / name, text)
    return 0


def write(path: Path, text: str) -> None:
    path.write_text(text)
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    sys.exit(main())
