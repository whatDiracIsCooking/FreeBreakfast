#!/usr/bin/env python3
"""Regenerate rust/'s packed three-center kernels under
rust/integrals/kernels/eri/density_fitting/three_center/packed/kernels/.

The kernels are FelizJueves's (deps/FelizJueves), emitted here and committed,
so a build never runs Python. Run this after bumping that submodule:

    tools/emit/packed_three_center_kernels.py

`--out-root DIR` writes the same tree under DIR instead of
rust/integrals/kernels/, each file at its path relative to it. The C++ tree
has no such kernels: this tool writes rust/'s alone, and its default root is
rust/'s (rust/README.md, "Build and test").

One family, FelizJueves's `Target.ERI3` built `packed=True`
(`eri/density_fit/kernel_eri3.py`, "THE PACKED OUTPUT"): `peri3<la, lb, lP>`,
the three-center (mn|P) with the square `eri3<la, lb, lP>`'s records, cells
and arithmetic, each cell's whole staged Cartesian block stored at its cell
table's destination offset `dst`, element `(a, b, q)` at
`dst + n_aux (a nB + b) + q`, the aux index fastest, a diagonal shell pair
(`diag = 1`) storing its full square. Every class tools/emit/df_kernels.py
emits for `eri3`: `la <= lb` with both at most MAX_L (f), against every `lP`
at most AUX_MAX_L (i), 70 classes, each in the same form as its square twin
(FelizJueves's default, but for tools/emit/lowstack.py's forced classes), so
a packed block is bitwise the square kernel's.

As tools/emit/df_kernels.py does, each class is one explicit specialization
of the template `peri3`, and `peri3.h` declares the primary, every class and
`peri3_kernels`, the [la][lb][lP] table a launcher indexes. Every [n, 3]
record vector is split into three flat arrays (`ir.passes.split_axes`),
spelled as there, so a launcher hands the bra a class's record rows as they
are. The units include the device Boys routine by bare name, the
`eval_boys.h` rust/integrals/kernels/shared/ carries; this tool writes
nothing there.

FelizJueves needs only the standard library to import, so this needs no
virtualenv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deps/FelizJueves/src"))

from df_kernels import AUX_AXES, BRA_AXES, eri3_classes  # noqa: E402
from felizjueves.emit import gpu  # noqa: E402
from felizjueves.emit.gpu import Specialization  # noqa: E402
from felizjueves.eri.density_fit.kernel_eri3 import build_eri3_kernel  # noqa: E402
from felizjueves.ir.passes import split_axes  # noqa: E402
from felizjueves.performance.device import by_name  # noqa: E402
from lowstack import forced  # noqa: E402

# With no shuffle the arch fixes nothing in the text; the gpu backend still
# takes one.
ARCH = "sm_86"

# Relative to the output root.
OUT = Path("eri/density_fitting/three_center/packed/kernels")


def peri3(cls: tuple[int, int, int]):
    # The square twin's form (tools/emit/df_kernels.py's `eri3`).
    return split_axes(
        build_eri3_kernel(cls, packed=True, **forced("eri3", cls)),
        {**BRA_AXES, **AUX_AXES},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "rust/integrals/kernels",
        help="the directory written in place of rust/integrals/kernels/ (default: it)",
    )
    root = parser.parse_args().out_root.resolve()
    out = root / OUT
    out.mkdir(parents=True, exist_ok=True)
    arch = by_name(ARCH)
    classes = eri3_classes()
    for cls in classes:
        kernel = peri3(cls)
        spec = Specialization("peri3", cls)
        write(out / f"{kernel.name}.cu", gpu.emit(kernel, arch, spec))
    # Any class's kernel spells the primary: one family, one parameter block.
    write(out / spec.header, gpu.emit_template(kernel, arch, spec, classes))
    return 0


def write(path: Path, text: str) -> None:
    path.write_text(text)
    print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    sys.exit(main())
