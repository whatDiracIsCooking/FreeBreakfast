#!/usr/bin/env python3
"""Regenerate the exchange-build kernels under src/integrals/exchange/kernels/.

The kernels are FelizJueves's (deps/FelizJueves), emitted here and committed,
so a build never runs Python. Run this after bumping that submodule:

    tools/emit_exchange_kernels.py

There is one family, `kb<la, lc, lb, ld>`: FelizJueves's `Target.K` on
opt-UM's mapping (`kernel_kblock.py`), K_ab += SUM_cd D_cd (ac|bd) on the J
engine's recurrence, one warp per contracted shell block K_AB, its lanes over
the block's kets, every lane walking its bras, and lane 0 adding the block
straight into K. The class is the Hermite pairing, bra `(a, c)`
first and ket `(b, d)` second, not the AO quartet order; `(lp, lq) =
(la + lc, lb + ld)`. Every ordered `(la, lc)` x ordered `(lb, ld)` with each
l at most 2 (kMaxAngularMomentum) is written: 81 classes. K has no symmetry
inside a pair, so unlike the J and one-electron tools nothing is written
`la <= lb` only.

Each class is `build_kblock_kernel(am)`, whose write into K is already an
atomic add (several classes add into one K), then:

- `passes.split_axes`: each [n, 3] record vector becomes three flat arrays,
  named as free_breakfast.integrals.mmd names its fields: `p_*` for the bra
  centre P and `q_*` for the ket centre Q (as the J kernel's), and the two
  displacements of each side by the letter of the shell they are from,
  `pa_*` = P - A and `pc_*` = P - C on the bra, `qb_*` = Q - B and
  `qd_*` = Q - D on the ket. In an ordered pair record's terms (the first
  shell the output index) `pa_*`/`pc_*` are the bra record's first and
  second displacements, `qb_*`/`qd_*` the ket's.

They are FelizJueves's `gpu` spelling, compiled as emitted by both backends:
nvcc under CUDA and clang `-x hip` under HIP. Each is emitted `any_wavefront`
with `block_dim=(1, WAVEFRONT)`: its shuffles and its shared `Ksh[n_p +
n_a n_b, FJ_WAVEFRONT]` read FJ_WAVEFRONT, which deps/CMakeLists.txt sets
from WWR_WARP_SIZE, so one text serves a 32- and a 64-lane build; the
header's `kb_block_dim` is the block to launch with, {1, FJ_WAVEFRONT}.
Nothing is added to the text here.

As tools/emit_coulomb_kernels.py does, each class is one explicit
specialization of the template, and the directory gets one header, `kb.h`,
every unit includes: the primary, each class's declaration and `kb_kernels`,
the table a launcher indexes. The device Boys routine the units include is
src/integrals/shared/eval_boys.h, written by the other two tools; this one
writes nothing there.

FelizJueves needs only the standard library to import, so this needs no
virtualenv.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deps/FelizJueves/src"))

from felizjueves.device import by_name  # noqa: E402
from felizjueves.emit import gpu  # noqa: E402
from felizjueves.emit.gpu import WAVEFRONT, Specialization  # noqa: E402
from felizjueves.kernel_kblock import build_kblock_kernel  # noqa: E402
from felizjueves.passes import split_axes  # noqa: E402

# The kernels' L limit: d. It must equal kMaxAngularMomentum
# (src/basis/angular_momentum.h).
MAX_L = 2
# With any_wavefront the arch fixes nothing in the text; the gpu backend
# still takes one.
ARCH = "sm_86"

OUT = ROOT / "src/integrals/exchange/kernels"
FAMILY = "kb"

# The record vectors, as mmd names them (module docstring).
AXES = {
    "P": ("p_x", "p_y", "p_z"),
    "XPA": ("pa_x", "pa_y", "pa_z"),
    "XPC": ("pc_x", "pc_y", "pc_z"),
    "Q": ("q_x", "q_y", "q_z"),
    "XQB": ("qb_x", "qb_y", "qb_z"),
    "XQD": ("qd_x", "qd_y", "qd_z"),
}
OPTIONS = {"block_dim": (1, WAVEFRONT), "any_wavefront": True}


def classes() -> list[tuple[int, int, int, int]]:
    ls = range(MAX_L + 1)
    return [(la, lc, lb, ld) for la in ls for lc in ls for lb in ls for ld in ls]


def kb(cls: tuple[int, int, int, int]):
    return split_axes(build_kblock_kernel(cls), AXES)


def main() -> int:
    arch = by_name(ARCH)
    out = OUT / FAMILY
    out.mkdir(parents=True, exist_ok=True)
    every = classes()
    for cls in every:
        kernel = kb(cls)
        spec = Specialization(FAMILY, cls)
        write(out / f"{kernel.name}.cu", gpu.emit(kernel, arch, spec, **OPTIONS))
    # Any class's kernel spells the primary: one family, one parameter block.
    write(out / spec.header, gpu.emit_template(kernel, arch, spec, every, **OPTIONS))
    return 0


def write(path: Path, text: str) -> None:
    path.write_text(text)
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    sys.exit(main())
