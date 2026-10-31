#!/usr/bin/env python3
"""Regenerate the density-fitting Hessian kernels under
src/integrals/eri/density_fitting/hessian/kernels/.

The kernels are FelizJueves's (deps/FelizJueves), emitted here and committed,
so a build never runs Python. Run this after bumping that submodule:

    tools/emit/df_hessian_kernels.py

It writes the ladders FelizJueves's `--hess-metric-l 6` and `--hess-eri3-l 3
--hess-eri3-aux-l 6` do (`emit.__main__.hess_metric_classes` and
`hess_eri3_classes`, called here, so the two cannot disagree), as
specializations of two templates rather than as free kernels:

- `d2metric<lP, lQ>`, FelizJueves's `Target.METRIC` at derivative order 2
  (`hessian/kernel_d2metric.py`): SUM_PQ W_PQ d2(P|Q)/dR dR, one thread per
  aux shell pair, staged as the metric's energy kernel is. Every `lP <= lQ`
  with both at most AUX_MAX_L (i), 28 classes, the top one (i|i).
- `d2eri3<la, lb, lP>`, `Target.ERI3` at order 2
  (`hessian/kernel_d2eri3.py`): SUM_mnP Gam_mnP d2(mn|P)/dR dR, one thread
  per (AO shell pair, aux shell), staged as the three-center energy kernel
  is. Every `la <= lb` with both at most MAX_L (f), against every `lP` at most
  AUX_MAX_L (i), 70 classes, the top one (ff|i).

Each kernel's parameter block is its first-derivative twin's
(tools/emit/df_gradient_kernels.py), the n_atom x 3 G replaced by the
n_atom x 3 x n_atom x 3 H. The aux centre is differentiated by the Hermite
shift, not by the Cartesian rule: its single-center E does not depend on it,
so d/dC moves only [r]^(0), and the tower is read two indices up (L + 2, 14
at (ff|i)) instead of the aux E table raised to l = 8. No centre is inferred
from the others by translational invariance, so the sum rule is a check, not
an assumption (FelizJueves's docs/implementation/hessian.md).

Each builder picks its own form per class: from FelizJueves's measured `L`
floors (`kernel_d2metric.LOWSTACK_MIN_L`, `kernel_d2eri3.LOWSTACK_MIN_L`) the
low-stack one. Nothing here forces either.

Every [n, 3] record vector is split into three flat arrays
(`ir.passes.split_axes`), named as df_gradient_kernels.py names them. The
limits are the energy kernels', read from the bridge headers as that tool
reads them; hessian/hessian_launch.cu checks the emitted tables against them
at compile time. The device Boys routine the units include is
src/integrals/shared/eval_boys.h, written by the one-electron and Coulomb
tools; this one writes nothing there.

FelizJueves needs only the standard library to import, so this needs no
virtualenv.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deps/FelizJueves/src"))
sys.path.insert(0, str(ROOT / "tools/emit"))

from df_gradient_kernels import (  # noqa: E402
    ARCH,
    AUX_AXES,
    AUX_MAX_L,
    BRA_AXES,
    DF,
    MAX_L,
    check_limits,
    write,
)
from felizjueves.emit import gpu  # noqa: E402
from felizjueves.emit.__main__ import (  # noqa: E402
    hess_eri3_classes,
    hess_metric_classes,
)
from felizjueves.emit.gpu import Specialization  # noqa: E402
from felizjueves.hessian.kernel_d2eri3 import build_d2eri3_kernel  # noqa: E402
from felizjueves.hessian.kernel_d2metric import (  # noqa: E402
    build_d2metric_kernel,
)
from felizjueves.ir.passes import split_axes  # noqa: E402
from felizjueves.performance.device import by_name  # noqa: E402

OUT = DF / "hessian/kernels"


def d2metric(cls: tuple[int, int]):
    return split_axes(build_d2metric_kernel(cls), AUX_AXES)


def d2eri3(cls: tuple[int, int, int]):
    return split_axes(build_d2eri3_kernel(cls), {**BRA_AXES, **AUX_AXES})


def main() -> int:
    check_limits()
    arch = by_name(ARCH)
    OUT.mkdir(parents=True, exist_ok=True)
    for family, classes, build in (
        ("d2metric", hess_metric_classes(AUX_MAX_L), d2metric),
        ("d2eri3", hess_eri3_classes(MAX_L, AUX_MAX_L), d2eri3),
    ):
        for cls in classes:
            kernel = build(cls)
            spec = Specialization(family, cls)
            write(OUT / f"{kernel.name}.cu", gpu.emit(kernel, arch, spec))
        # Any class's kernel spells the primary: one family, one parameter block.
        write(OUT / spec.header, gpu.emit_template(kernel, arch, spec, classes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
