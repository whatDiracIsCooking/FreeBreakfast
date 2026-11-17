#!/usr/bin/env python3
"""Regenerate rust/'s conventional-ERI kernels under
rust/integrals/kernels/eri/conventional/kernels/ (packed) and
rust/integrals/kernels/eri/conventional/dense/kernels/ (dense).

The kernels are FelizJueves's full ERI (deps/FelizJueves, `eri/full_eri/`),
packed and dense, emitted here and committed, so a build never runs Python.
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
  pair) task through (pp|pp) and one per bra row of a task from a d shell
  (the low-stack form), plain stores. Every quartet of pair classes `la <= lb`,
  `lc <= ld`, both at most MAX_L (d), the bra pair class not after the ket's
  in `(la, lb)` order: 21 classes, the top one (dd|dd) at L = 8. Each class is
  one explicit specialization of the template `eri`, and `eri.h` declares
  the primary, every class and `eri_kernels`, the [la][lb][lc][ld] table a
  launcher indexes (null where a class is not emitted).
- `eri_kperm` (`eri/full_eri/kernel_kperm.py`): the exchange-ordered
  `X_pp[pidx(l, s), pidx(m, n)] = ((ml|ns) + (ms|nl)) / 2`, gathered from
  the packed ERI once, one thread per packed column. No class, so one plain
  kernel; its launcher declares it.

And the dense ERI, into `eri/conventional/dense/kernels/`
(`eri/full_eri/kernel_dense.py`, `Target.ERI_DENSE`, FelizJueves's
`--dense-eri-l 2`): a row block of `M_J[(m,n), (l,s)] = (mn|ls)` or of
`M_K[(m,n), (l,s)] = (ml|ns)`, each dense `n_ao^2 x n_ao^2` with
`(m, n) -> m n_ao + n`, one thread per ORDERED shell quartet task through
(pp|pp) and one per bra Cartesian of a task from a d shell, plain
stores, no symmetry. Every ordered class `(la lb|lc ld)`, each `l` at most
MAX_L: 81 classes in each of the two layouts, `eri_dj<la, lb, lc, ld>` and
`eri_dk<la, lb, lc, ld>`, each layout a template of its own whose header
(`eri_dj.h`, `eri_dk.h`) declares the primary, every class and the
[la][lb][lc][ld] table. The record and pair parameters are the packed
kernel's; `m0 n_row n_col` (the local row block) replace `n_pp` and `pidx`.

Each class's threads per task, FelizJueves's `task_threads` for the form
it emits (1 in the one-thread-per-task form through (pp|pp); from a d shell
the low-stack form's, `kernel_eri.task_threads` = `n_a n_b` packed, one bra
row a thread, and `kernel_dense.task_threads` = `n_a` dense, the `n_b` rows
of one bra Cartesian a thread), into `eri/conventional/kernels/` as
`eri_task_threads.h`: the two [la][lb][lc][ld] tables `eri_task_threads`
and `eri_dense_task_threads`, which the launchers multiply `n_task` by
(FelizJueves's `launch_shape(n_task, threads, rows=...)`).

FelizJueves's third packed kernel, `eri_jgemv`, is not emitted: J and K are the BLAS's
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
from felizjueves.eri.full_eri.kernel_dense import (  # noqa: E402
    LAYOUTS as DENSE_LAYOUTS,
)
from felizjueves.eri.full_eri.kernel_dense import (  # noqa: E402
    build_dense_kernel,
)
from felizjueves.eri.full_eri.kernel_dense import (  # noqa: E402
    task_threads as dense_task_threads,
)
from felizjueves.eri.full_eri.kernel_eri import (  # noqa: E402
    build_eri_kernel,
    task_threads,
)
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

# Relative to the output root: the packed units, and the dense ones.
OUT = Path("eri/conventional/kernels")
DENSE_OUT = Path("eri/conventional/dense/kernels")

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


def dense_classes() -> list[tuple[int, int, int, int]]:
    """FelizJueves's `emit.__main__.dense_eri_classes(MAX_L)`, spelled here:
    every ordered quartet class."""
    ls = range(MAX_L + 1)
    return [(la, lb, lc, ld) for la in ls for lb in ls for lc in ls for ld in ls]


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
    classes = eri_classes()
    for cls in classes:
        kernel = split_axes(build_eri_kernel(cls), AXES)
        spec = Specialization("eri", cls)
        write(out / f"{kernel.name}.cu", gpu.emit(kernel, arch, spec))
    # Any class's kernel spells the primary: one family, one parameter block.
    write(out / spec.header, gpu.emit_template(kernel, arch, spec, classes))
    kperm = build_kperm_kernel()
    write(out / f"{kperm.name}.cu", gpu.emit(kperm, arch))

    dense_out = root / DENSE_OUT
    dense_out.mkdir(parents=True, exist_ok=True)
    dense = dense_classes()
    for layout in DENSE_LAYOUTS:
        for cls in dense:
            kernel = split_axes(build_dense_kernel(cls, layout), AXES)
            spec = Specialization(f"eri_d{layout}", cls)
            write(dense_out / f"{kernel.name}.cu", gpu.emit(kernel, arch, spec))
        write(dense_out / spec.header, gpu.emit_template(kernel, arch, spec, dense))

    write(out / "eri_task_threads.h", threads_header(classes, dense))
    return 0


def threads_header(
    packed: list[tuple[int, int, int, int]], dense: list[tuple[int, int, int, int]]
) -> str:
    """`eri_task_threads.h`: each emitted class's threads per task, packed
    and dense, as two [la][lb][lc][ld] tables (0 where a class is not
    emitted)."""
    n = MAX_L + 1

    def table(name: str, classes: list[tuple[int, int, int, int]], rule) -> str:
        values = {c: rule(c) for c in classes}

        def cell(*idx: int) -> str:
            if len(idx) == 4:
                return str(values.get(idx, 0))
            return "{" + ", ".join(cell(*idx, i) for i in range(n)) + "}"

        rows = ",\n".join(f"    {cell(la)}" for la in range(n))
        return (
            f"inline constexpr int32_t {name}[{n}][{n}][{n}][{n}] = {{\n{rows},\n}};\n"
        )

    return (
        "// eri_task_threads: generated by tools/emit/conventional_kernels.py\n"
        "// from FelizJueves's kernel_eri.task_threads and\n"
        "// kernel_dense.task_threads -- do not edit. Threads per task of each\n"
        "// emitted class, [la][lb][lc][ld], 0 where none is emitted: a launch\n"
        "// covers n_task times it.\n"
        "#pragma once\n"
        "#include <cstdint>\n"
        "\n"
        + table("eri_task_threads", packed, task_threads)
        + "\n"
        + table("eri_dense_task_threads", dense, dense_task_threads)
    )


def write(path: Path, text: str) -> None:
    path.write_text(text)
    print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)


if __name__ == "__main__":
    sys.exit(main())
