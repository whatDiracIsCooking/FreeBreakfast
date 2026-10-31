#!/usr/bin/env python3
"""Regenerate the density-fitting gradient kernels under
src/integrals/eri/density_fitting/gradient/kernels/.

The kernels are FelizJueves's (deps/FelizJueves), emitted here and committed,
so a build never runs Python. Run this after bumping that submodule:

    tools/emit/df_gradient_kernels.py

It writes the ladder FelizJueves's `--grad-df-l 3 --grad-df-aux-l 6` does
(`emit.__main__.grad_df_classes`, called here, so the two cannot disagree), as
specializations of two templates rather than as free kernels:

- `gmetric<lP, lQ>`, FelizJueves's `Target.METRIC` at derivative order 1
  (`gradient/kernel_gmetric.py`): SUM_PQ W_PQ d(P|Q)/dR, one thread per aux
  shell pair, staged as the metric's energy kernel is. Every `lP <= lQ` with
  both at most AUX_MAX_L (i), 28 classes, the top one (i|i).
- `geri3<la, lb, lP>`, `Target.ERI3` at order 1
  (`gradient/kernel_geri3.py`): SUM_mnP Gam_mnP d(mn|P)/dR, one thread per
  (AO shell pair, aux shell), staged as the three-center energy kernel is.
  Every `la <= lb` with both at most MAX_L (f), against every `lP` at most
  AUX_MAX_L (i), 70 classes, the top one (ff|i).

Each builder picks its own form per class: from FelizJueves's measured `L`
floors (`kernel_gmetric.LOWSTACK_MIN_L`, `kernel_geri3.LOWSTACK_MIN_L`) the
low-stack one, which holds no density and keeps (ff|i) under its stack budget
(FelizJueves's docs/implementation/gradient.md, "The low-stack form"). Nothing
here forces either.

Every [n, 3] record vector is split into three flat arrays
(`ir.passes.split_axes`), named as tools/emit/df_kernels.py names the energy
kernels': the bra's `p_*`, `pa_*`, `pb_*`, the aux record's centre `a_*`.

The limits are the energy kernels': MAX_L must equal `kThreeCenterMaxMomentum`
(three_center/three_center_bridge.h), AUX_MAX_L `kThreeCenterAuxMaxMomentum`
there and `kMetricMaxMomentum` (metric/metric_bridge.h). This tool reads the
three from those headers and refuses to write a ladder that disagrees with
them; gradient/gradient_launch.cu checks the emitted tables against them at
compile time.

The kernels are FelizJueves's `gpu` spelling, compiled as emitted by both
backends. Neither family has a shuffle or a shared array (each ends in plain
`atomicAdd`s into G), so they are emitted with the default options. One header
per family, `gmetric.h` and `geri3.h`, declares the primary, each class's
specialization and the table a launcher indexes, as df_kernels.py's do. The
device Boys routine the units include is src/integrals/shared/eval_boys.h,
written by the one-electron and Coulomb tools; this one writes nothing there.

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
from felizjueves.emit.__main__ import grad_df_classes  # noqa: E402
from felizjueves.emit.gpu import Specialization  # noqa: E402
from felizjueves.gradient.kernel_geri3 import build_geri3_kernel  # noqa: E402
from felizjueves.gradient.kernel_gmetric import (  # noqa: E402
    build_gmetric_kernel,
)
from felizjueves.gradient.kernel_ipmetric import (  # noqa: E402
    build_ipmetric_kernel,
)
from felizjueves.gradient.kernel_xeri3 import build_xeri3_kernel  # noqa: E402
from felizjueves.ir.passes import split_axes  # noqa: E402
from felizjueves.performance.device import by_name  # noqa: E402

# The orbital limit, f: kThreeCenterMaxMomentum (module docstring).
MAX_L = 3
# The aux limit, i: kThreeCenterAuxMaxMomentum and kMetricMaxMomentum, above
# the orbital one on purpose (tools/emit/df_kernels.py).
AUX_MAX_L = 6
# With no shuffle the arch fixes nothing in the text; the gpu backend still
# takes one.
ARCH = "sm_86"

DF = ROOT / "src/integrals/eri/density_fitting"
OUT = DF / "gradient/kernels"
METRIC_BRIDGE = DF / "metric/metric_bridge.h"
THREE_CENTER_BRIDGE = DF / "three_center/three_center_bridge.h"

# The record vectors (module docstring).
AUX_AXES = {"A": ("a_x", "a_y", "a_z")}
BRA_AXES = {
    "P": ("p_x", "p_y", "p_z"),
    "XPA": ("pa_x", "pa_y", "pa_z"),
    "XPB": ("pb_x", "pb_y", "pb_z"),
}


def gmetric(cls: tuple[int, int]):
    return split_axes(build_gmetric_kernel(cls), AUX_AXES)


def geri3(cls: tuple[int, int, int]):
    return split_axes(build_geri3_kernel(cls), {**BRA_AXES, **AUX_AXES})


def ipmetric(cls: tuple[int, int]):
    return split_axes(build_ipmetric_kernel(cls), AUX_AXES)


def xeri3(cls: tuple[int, int, int]):
    return split_axes(build_xeri3_kernel(cls), {**BRA_AXES, **AUX_AXES})


def main() -> int:
    check_limits()
    arch = by_name(ARCH)
    metric_classes, eri3_classes = grad_df_classes(MAX_L, AUX_MAX_L)
    OUT.mkdir(parents=True, exist_ok=True)
    for family, classes, build in (
        ("gmetric", metric_classes, gmetric),
        ("geri3", eri3_classes, geri3),
        ("ipmetric", metric_classes, ipmetric),
        ("xeri3", eri3_classes, xeri3),
    ):
        for cls in classes:
            kernel = build(cls)
            spec = Specialization(family, cls)
            write(OUT / f"{kernel.name}.cu", gpu.emit(kernel, arch, spec))
        # Any class's kernel spells the primary: one family, one parameter block.
        write(OUT / spec.header, gpu.emit_template(kernel, arch, spec, classes))
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
