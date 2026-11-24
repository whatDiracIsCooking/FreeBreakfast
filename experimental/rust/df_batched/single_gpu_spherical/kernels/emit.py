#!/usr/bin/env python3
"""Emit FelizJueves's fully spherical packed three-center kernels,
`ssperi3<la, lb, lP>` (FelizJueves#745), into a directory.

    emit.py OUT_DIR

The family is rust/'s `speri3` (tools/emit/packed_three_center_kernels.py)
with the bra transformed in the kernel too: `build_eri3_kernel(cls,
packed=True, spherical_aux=True, spherical_bra=True)`, `(mu nu|Q)` with all
three indices real solid harmonics, `(2 la + 1)(2 lb + 1)(2 lP + 1)` elements
a cell, element `(mu, nu, q)` at `dst + n_aux (mu (2 lb + 1) + nu) + q`.
Same classes (`la <= lb <= f`, `lP <= i`, 70 of them), same parameter list
and the same split record axes as `speri3`, so a launcher is the same but for
the table it indexes. Low-stack only, as FelizJueves emits it.

CMakeLists.txt runs this at build time, so the generated units are never
committed; they follow deps/FelizJueves, which must be at or past
FelizJueves 6c9b85e (#745). FelizJueves needs only the standard library to
import, so this needs no virtualenv.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(ROOT / "deps/FelizJueves/src"), str(ROOT / "tools/emit")]

from df_kernels import AUX_AXES, BRA_AXES, eri3_classes  # noqa: E402
from felizjueves.emit import gpu  # noqa: E402
from felizjueves.emit.gpu import Specialization  # noqa: E402
from felizjueves.eri.density_fit.kernel_eri3 import build_eri3_kernel  # noqa: E402
from felizjueves.ir.passes import split_axes  # noqa: E402
from felizjueves.performance.device import by_name  # noqa: E402

# With no shuffle the arch fixes nothing in the text, as for speri3.
ARCH = "sm_86"


def ssperi3(cls: tuple[int, int, int]):
    return split_axes(
        build_eri3_kernel(cls, packed=True, spherical_aux=True, spherical_bra=True),
        {**BRA_AXES, **AUX_AXES},
    )


def write(path: Path, text: str) -> None:
    # Unchanged text keeps its mtime, so a rerun rebuilds nothing.
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    out = Path(sys.argv[1]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    arch = by_name(ARCH)
    classes = eri3_classes()
    for cls in classes:
        kernel = ssperi3(cls)
        spec = Specialization("ssperi3", cls)
        write(out / f"{kernel.name}.cu", gpu.emit(kernel, arch, spec))
    write(out / spec.header, gpu.emit_template(kernel, arch, spec, classes))
    return 0


if __name__ == "__main__":
    sys.exit(main())
