#!/usr/bin/env python3
"""Regenerate the density-fitting kernels under
src/integrals/eri/density_fitting/{metric,three_center}/kernels/.

The kernels are FelizJueves's (deps/FelizJueves), emitted here and committed,
so a build never runs Python. Run this after bumping that submodule:

    tools/emit/df_kernels.py

Two families, one per FelizJueves target, each its module's own directory:

- `metric<lP, lQ>`, FelizJueves's `Target.METRIC`
  (`eri/density_fit/kernel_metric.py`): the Coulomb metric (P|Q) of an
  auxiliary basis, one thread per aux shell pair, written to metric/kernels/.
  Every `lP <= lQ` with both at most AUX_MAX_L (i) is written, 28 classes, the
  top one (i|i) at L = 12.
- `eri3<la, lb, lP>`, FelizJueves's `Target.ERI3`
  (`eri/density_fit/kernel_eri3.py`): the three-center (mn|P), one thread per
  (AO shell pair, aux shell), written to three_center/kernels/. Every
  `la <= lb` with both at most MAX_L (f), against every `lP` at most AUX_MAX_L
  (i), 70 classes, the top one (ff|i) at L = 12.

Pair classes are written `la <= lb` (and `lP <= lQ`) only, for the reason
tools/emit/one_electron_kernels.py gives: each kernel writes every element and
its mirror, so a pair staged swapped needs no class of its own.

Every [n, 3] record vector is split into three flat arrays
(`ir.passes.split_axes`): the bra's as free_breakfast.integrals.mmd names its
fields, `p_*` for the centre P, `pa_*` = P - A and `pb_*` = P - B; the aux
record's centre `A` (one Gaussian, so its own centre) as `a_*`, in both
families alike.

The limits are each family's own, as J's and K's are: MAX_L must equal
`kThreeCenterMaxMomentum` (three_center/three_center_bridge.h), AUX_MAX_L
`kThreeCenterAuxMaxMomentum` there and `kMetricMaxMomentum`
(metric/metric_bridge.h). This tool reads the three from those headers and
refuses to write a ladder that disagrees with them; each family's
`<family>_table.cu` checks its emitted table against them at compile time, and
three_center_bridge.h its orbital limit against the tree-wide
kMaxAngularMomentum.

They are FelizJueves's `gpu` spelling, compiled as emitted by both backends:
nvcc under CUDA and clang `-x hip` under HIP. Neither family has a shuffle or
a shared array, so they are emitted with the default options and their text
has no width. Nothing is added to the text here.

As tools/emit/coulomb_kernels.py does, each class is one explicit
specialization of its family's template, and each directory gets one header,
`metric.h` or `eri3.h`, every unit includes: the primary, each class's
declaration and `metric_kernels` / `eri3_kernels`, the table a launcher
indexes. The device Boys routine the units include is
src/integrals/shared/eval_boys.h, written by the one-electron and Coulomb
tools; this one writes nothing there.

FelizJueves needs only the standard library to import, so this needs no
virtualenv.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deps/FelizJueves/src"))

from felizjueves.emit import gpu  # noqa: E402
from felizjueves.emit.gpu import Specialization  # noqa: E402
from felizjueves.eri.density_fit.kernel_eri3 import build_eri3_kernel  # noqa: E402
from felizjueves.eri.density_fit.kernel_metric import (  # noqa: E402
    build_metric_kernel,
)
from felizjueves.ir.passes import split_axes  # noqa: E402
from felizjueves.performance.device import by_name  # noqa: E402

# The orbital limit, f: kThreeCenterMaxMomentum (module docstring).
MAX_L = 3
# The aux limit, i: kThreeCenterAuxMaxMomentum and kMetricMaxMomentum. It is
# deliberately above the orbital MAX_L: a JKFIT or RI set carries shells two l
# above the orbital set it fits, so the two ladders are separate knobs.
AUX_MAX_L = 6
# With no shuffle the arch fixes nothing in the text; the gpu backend still
# takes one.
ARCH = "sm_86"

OUT = ROOT / "src/integrals/eri/density_fitting"
METRIC_BRIDGE = OUT / "metric/metric_bridge.h"
THREE_CENTER_BRIDGE = OUT / "three_center/three_center_bridge.h"

# The record vectors (module docstring).
AUX_AXES = {"A": ("a_x", "a_y", "a_z")}
BRA_AXES = {
    "P": ("p_x", "p_y", "p_z"),
    "XPA": ("pa_x", "pa_y", "pa_z"),
    "XPB": ("pb_x", "pb_y", "pb_z"),
}


def metric_classes() -> list[tuple[int, int]]:
    ls = range(AUX_MAX_L + 1)
    return [(lp, lq) for lp in ls for lq in ls if lp <= lq]


def eri3_classes() -> list[tuple[int, int, int]]:
    ls = range(MAX_L + 1)
    return [
        (la, lb, lp)
        for la in ls
        for lb in ls
        if la <= lb
        for lp in range(AUX_MAX_L + 1)
    ]


def metric(cls: tuple[int, int]):
    return split_axes(build_metric_kernel(cls), AUX_AXES)


def eri3(cls: tuple[int, int, int]):
    return split_axes(build_eri3_kernel(cls), {**BRA_AXES, **AUX_AXES})


# family -> (its module's directory, its classes, its builder); the kernels go
# in that directory's kernels/.
FAMILIES = {
    "metric": ("metric", metric_classes, metric),
    "eri3": ("three_center", eri3_classes, eri3),
}


def main() -> int:
    check_limits()
    arch = by_name(ARCH)
    for family, (directory, classes_of, build) in FAMILIES.items():
        out = OUT / directory / "kernels"
        out.mkdir(parents=True, exist_ok=True)
        classes = classes_of()
        for cls in classes:
            kernel = build(cls)
            spec = Specialization(family, cls)
            write(out / f"{kernel.name}.cu", gpu.emit(kernel, arch, spec))
        # Any class's kernel spells the primary: one family, one parameter block.
        write(out / spec.header, gpu.emit_template(kernel, arch, spec, classes))
    return 0


def check_limits() -> None:
    """Refuse a ladder the bridge headers' limits disagree with."""
    wanted = {
        (METRIC_BRIDGE, "kMetricMaxMomentum"): AUX_MAX_L,
        (THREE_CENTER_BRIDGE, "kThreeCenterMaxMomentum"): MAX_L,
        (THREE_CENTER_BRIDGE, "kThreeCenterAuxMaxMomentum"): AUX_MAX_L,
    }
    for (path, name), value in wanted.items():
        match = re.search(
            rf"inline constexpr int32_t {name} = (\d+);", path.read_text()
        )
        if match is None:
            sys.exit(f"{path.relative_to(ROOT)}: no {name}")
        if int(match.group(1)) != value:
            sys.exit(
                f"{path.relative_to(ROOT)}: {name} = {match.group(1)}, but this "
                f"tool writes its ladder through {value}"
            )


def write(path: Path, text: str) -> None:
    path.write_text(text)
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    sys.exit(main())
