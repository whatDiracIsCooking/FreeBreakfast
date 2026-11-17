#!/usr/bin/env python3
"""Regenerate rust/'s conventional-ERI kernels under
rust/integrals/kernels/eri/conventional/kernels/.

The kernels are FelizJueves's packed full ERI (deps/FelizJueves,
`eri/full_eri/`), emitted here and committed, so a build never runs Python.
Run this after bumping that submodule:

    tools/emit/conventional_kernels.py

`--out-root DIR` writes the same tree under DIR instead of
rust/integrals/kernels/, each file at its path relative to it. The C++ tree
has no such kernels: this tool writes rust/'s alone, and its default root is
rust/'s (rust/README.md, "Build and test").

Two kernels, each FelizJueves's `--eri-l 2` ladder's, emitted with the
default options (neither has a shuffle or a shared array, so the text has no
width):

- `eri<la, lb, lc, ld>`, `Target.ERI` (`eri/full_eri/kernel_eri.py`): every
  (mn|ls) of a class, packed by AO pair into one symmetric n_pp x n_pp matrix
  (`n_pp = n_ao (n_ao + 1) / 2`), one thread per (bra shell pair, ket shell
  pair) task, plain stores. Every quartet of pair classes `la <= lb`,
  `lc <= ld`, both at most MAX_L (d), the bra pair class not after the ket's
  in `(la, lb)` order: 21 classes, the top one (dd|dd) at L = 8. Each class is
  one explicit specialization of the template `eri`, and `eri.h` declares
  the primary, every class and `eri_kernels`, the [la][lb][lc][ld] table a
  launcher indexes (null where a class is not emitted).
- `eri_kperm` (`eri/full_eri/kernel_kperm.py`): the exchange-ordered
  `X_pp[pidx(l, s), pidx(m, n)] = ((ml|ns) + (ms|nl)) / 2`, gathered from
  the packed ERI once, one thread per packed column. No class, so one plain
  kernel; its launcher declares it.

FelizJueves's third, `eri_jgemv`, is not emitted: J and K are the BLAS's
`gemv` over the two symmetric matrices (`fb_core_wwr::blas::gemv`), what the
tree already wraps.

Every [n, 3] record vector is split into three flat arrays
(`ir.passes.split_axes`), the bra's as free_breakfast.integrals.mmd names its
fields, `p_*`, `pa_*`, `pb_*`, and the ket's the same with a `q` for the `p`:
`q_*` for Q, `qc_*` = Q - C and `qd_*` = Q - D. So a launcher hands each side
a class's record rows as they are.

FelizJueves needs only the standard library to import, so this needs no
virtualenv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "deps/FelizJueves/src"))

from felizjueves.emit import gpu  # noqa: E402
from felizjueves.emit.gpu import Specialization  # noqa: E402
from felizjueves.eri.full_eri.kernel_eri import build_eri_kernel  # noqa: E402
from felizjueves.eri.full_eri.kernel_kperm import (  # noqa: E402
    build_kperm_kernel,
)
from felizjueves.ir.passes import split_axes  # noqa: E402
from felizjueves.performance.device import by_name  # noqa: E402

# The orbital limit, d: FelizJueves's ERI_L_MAX, and FB_CONVENTIONAL_MAX_MOMENTUM
# in rust/'s conventional.h, which conventional_launch.cu static_asserts
# against the emitted table.
MAX_L = 2
# With no shuffle the arch fixes nothing in the text; the gpu backend still
# takes one.
ARCH = "sm_86"

# Relative to the output root.
OUT = Path("eri/conventional/kernels")

# The record vectors (module docstring).
AXES = {
    "P": ("p_x", "p_y", "p_z"),
    "XPA": ("pa_x", "pa_y", "pa_z"),
    "XPB": ("pb_x", "pb_y", "pb_z"),
    "Q": ("q_x", "q_y", "q_z"),
    "XQC": ("qc_x", "qc_y", "qc_z"),
    "XQD": ("qd_x", "qd_y", "qd_z"),
}


def eri_classes() -> list[tuple[int, int, int, int]]:
    """FelizJueves's `emit.__main__.eri_classes(MAX_L)`, spelled here."""
    pcs = [(la, lb) for la in range(MAX_L + 1) for lb in range(la, MAX_L + 1)]
    return [(*x, *y) for i, x in enumerate(pcs) for y in pcs[i:]]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out-root",
        type=Path,
        default=ROOT / "rust/integrals/kernels",
        help="the directory written in place of rust/integrals/kernels/ (default: it)",
    )
    out = parser.parse_args().out_root.resolve() / OUT
    out.mkdir(parents=True, exist_ok=True)
    arch = by_name(ARCH)
    classes = eri_classes()
    for cls in classes:
        kernel = split_axes(build_eri_kernel(cls), AXES)
        spec = Specialization("eri", cls)
        write(out / f"{kernel.name}.cu", gpu.emit(kernel, arch, spec))
    # Any class's kernel spells the primary: one family, one parameter block.
    write(out / spec.header, gpu.emit_template(kernel, arch, spec, classes))
    kperm = build_kperm_kernel()
    write(out / f"{kperm.name}.cu", gpu.emit(kperm, arch))
    return 0


def write(path: Path, text: str) -> None:
    path.write_text(text)
    print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    sys.exit(main())
