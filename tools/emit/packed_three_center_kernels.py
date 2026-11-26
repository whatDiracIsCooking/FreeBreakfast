#!/usr/bin/env python3
"""Regenerate rust/'s packed three-center kernels under
rust/integrals/kernels/eri/density_fitting/three_center/packed/kernels/ and
.../three_center/spherical/kernels/.

The kernels are FelizJueves's (deps/FelizJueves), emitted here and committed,
so a build never runs Python. Run this after bumping that submodule:

    tools/emit/packed_three_center_kernels.py

`--out-root DIR` writes the same tree under DIR instead of
rust/integrals/kernels/, each file at its path relative to it. The C++ tree
has no such kernels: this tool writes rust/'s alone, and its default root is
rust/'s (rust/README.md, "Build and test").

Two families. The first is FelizJueves's `Target.ERI3` built `packed=True,
spherical_aux=True` (`eri/density_fit/kernel_eri3.py`, "THE PACKED OUTPUT"
and "THE SPHERICAL-AUX OUTPUT"; `docs/implementation/three_center.md`, "The
spherical-aux packed output (#743)"): `speri3<la, lb, lP>`, the
three-center (ab|Q) with the square `eri3<la, lb, lP>`'s records and cells,
`a`, `b` Cartesian and `Q` the aux shell's real solid harmonics, `2 lP + 1`
of them, so the aux half of the spherical transform is the kernel's. Each
cell's whole block is stored at its cell table's destination offset `dst`,
element `(a, b, q)` at `dst + n_aux (a nB + b) + q`, `n_aux` the aux
basis's spherical functions and the aux index fastest, a diagonal shell
pair (`diag = 1`) storing its full square. Every class
tools/emit/df_kernels.py emits for `eri3`: `la <= lb` with both at most
MAX_L (f), against every `lP` at most AUX_MAX_L (i), 70 classes, each in
FelizJueves's low-stack form, the only one the mode has, so
tools/emit/lowstack.py's choices do not apply.

The second, under spherical/kernels/, is the same build with the bra
transformed in the kernel too, `spherical_bra=True` (FelizJueves#745):
`ssperi3<la, lb, lP>`, `(mu nu|Q)` with all three indices real solid
harmonics, `(2 la + 1)(2 lb + 1)(2 lP + 1)` elements a cell, element
`(mu, nu, q)` at `dst + n_aux (mu (2 lb + 1) + nu) + q`. The same classes,
parameter list and split record axes as `speri3`, so its launcher is the
same but for the table it indexes; `ssperi3.h` declares its table,
`ssperi3_kernels`.

As tools/emit/df_kernels.py does, each class is one explicit specialization
of the template `speri3`, and `speri3.h` declares the primary, every class
and `speri3_kernels`, the [la][lb][lP] table a launcher indexes. Every
[n, 3] record vector is split into three flat arrays
(`ir.passes.split_axes`), spelled as there, so a launcher hands the bra a
class's record rows as they are. The units include the device Boys routine
by bare name, the `eval_boys.h` rust/integrals/kernels/shared/ carries;
this tool writes nothing there.

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

# With no shuffle the arch fixes nothing in the text; the gpu backend still
# takes one.
ARCH = "sm_86"

# Each family's name, its directory relative to the output root, and whether
# the kernel transforms the bra to real solid harmonics too.
FAMILIES = (
    ("speri3", Path("eri/density_fitting/three_center/packed/kernels"), False),
    ("ssperi3", Path("eri/density_fitting/three_center/spherical/kernels"), True),
)


def build(cls: tuple[int, int, int], spherical_bra: bool):
    return split_axes(
        build_eri3_kernel(
            cls, packed=True, spherical_aux=True, spherical_bra=spherical_bra
        ),
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
    arch = by_name(ARCH)
    classes = eri3_classes()
    for name, rel, spherical_bra in FAMILIES:
        out = root / rel
        out.mkdir(parents=True, exist_ok=True)
        for cls in classes:
            kernel = build(cls, spherical_bra)
            spec = Specialization(name, cls)
            write(out / f"{kernel.name}.cu", gpu.emit(kernel, arch, spec))
        # Any class's kernel spells the primary: one family, one parameter
        # block.
        write(out / spec.header, gpu.emit_template(kernel, arch, spec, classes))
    return 0


def write(path: Path, text: str) -> None:
    path.write_text(text)
    print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    sys.exit(main())
