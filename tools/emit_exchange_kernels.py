#!/usr/bin/env python3
"""Regenerate the exchange-build kernels under
src/integrals/eri/integral_direct/exchange/kernels/, and the dressed record
bound under src/integrals/mmd/kernels/.

The kernels are FelizJueves's (deps/FelizJueves), emitted here and committed,
so a build never runs Python. Run this after bumping that submodule:

    tools/emit_exchange_kernels.py

The K build's family is `kb<la, lc, lb, ld>`: FelizJueves's `Target.K` on
opt-UM's mapping (`eri/k_exchange/kernel_kblock.py`), K_ab += SUM_cd D_cd
(ac|bd) on the J engine's recurrence, one warp per contracted shell block
K_AB, its lanes over the block's kets, every lane walking its bras, and
lane 0 adding the block straight into K. The class is the Hermite pairing, bra `(a, c)`
first and ket `(b, d)` second, not the AO quartet order; `(lp, lq) =
(la + lc, lb + ld)`. Every ordered `(la, lc)` x ordered `(lb, ld)` with each
l at most 3 (MAX_L, kExchangeMaxMomentum) is written, 256 classes, in one of
two families by FelizJueves's rule "through d kb, at f kbr" (#580): `kb` for
the 81 with every l at most 2 (KB_MAX_L, kExchangeKbMaxMomentum), and `kbr`
for the 175 with an f shell. A `kb` at f is never written: `kb_33_33` is 22 MB
of source and never left cicc (#569). K has no symmetry inside a pair, so
unlike the J and one-electron tools nothing is written `la <= lb` only.

Each `kb` class is `build_kblock_kernel(am)`, whose write into K is already an
atomic add (several classes add into one K), then:

- `ir.passes.split_axes`: each [n, 3] record vector becomes three flat arrays,
  named as free_breakfast.integrals.mmd names its fields: `p_*` for the bra
  centre P and `q_*` for the ket centre Q (as the J kernel's), and the two
  displacements of each side by the letter of the shell they are from,
  `pa_*` = P - A and `pc_*` = P - C on the bra, `qb_*` = Q - B and
  `qd_*` = Q - D on the ket. In an ordered pair record's terms (the first
  shell the output index) `pa_*`/`pc_*` are the bra record's first and
  second displacements, `qb_*`/`qd_*` the ket's.

They are FelizJueves's `gpu` spelling, compiled as emitted by both backends:
nvcc under CUDA and clang `-x hip` under HIP. Each `kb` and `kbr` unit is
emitted `any_wavefront`
with `block_dim=(1, WAVEFRONT)`: its shuffles and its shared `Ksh[n_p +
n_a n_b, FJ_WAVEFRONT]` read FJ_WAVEFRONT, which deps/CMakeLists.txt sets
from WWR_WARP_SIZE, so one text serves a 32- and a 64-lane build; the
header's `kb_block_dim` is the block to launch with, {1, FJ_WAVEFRONT}.
Nothing is added to the text here.

Each `kbr` class is `build_kblock_rolled_kernel(am)` (record-major, not the
transposed `kbrt`), `kernel_kblock_rolled.py`: `kb`'s mapping, screen and
output, with E an input (`Eac`, `Ebd`, the records' undressed nonzero E,
which the launcher makes once per basis) and the contraction rolled into loops
over five class tables, so its text stays small at (ff|ff). Only its two
centres are split (`p_*`, `q_*`); it reads no displacement. Its tables
(`kernel_kblock_rolled.tables`) are constants of the class, so they are written
here too, once each, as device arrays in kernels/kbr/kbr_tables.h: E^{l1 l2}'s
CSR pattern per pair (`kbr_e_ptr`, `kbr_e_col`; a class's `ac_*` is its bra
pair's, `bd_*` its ket pair's) and `hpq` per `(lp, lq)` (`kbr_hpq`), each with
a host table of offsets. Every class's five tables are checked against those
slices before anything is written.

As tools/emit_coulomb_kernels.py does, each class is one explicit
specialization of its family's template, and each directory gets one header,
`kb.h` or `kbr.h`, every unit includes: the primary, each class's declaration
and `kb_kernels` / `kbr_kernels`, the table a launcher indexes (`nullptr`
where the other family has the class). The device Boys routine the units include is
src/integrals/shared/eval_boys.h, written by the other two tools; this one
writes nothing there.

One more family rides along because K's screen is its first reader:
`record_bound<la, lb>`, FelizJueves's `Target.RECORD_BOUND`
(`eri/k_exchange/kernel_record_bound.py`), K's DRESSED per-record Schwarz bound

    I = max_{i in a} SUM_{k in b} SUM_p |s E^{a_i b_k}_p| I_p(zeta)

(FelizJueves's host `eri.k_exchange.k.record_bounds`), one thread per
record, the max over the FIRST shell's Cartesians. mmd forms
it once per basis over its own records, as stored and swapped (its
bound_ab and bound_ba), so it is written to src/integrals/mmd/kernels/
record_bound/, beside mmd's `pair_bound`. The class is an ORDERED pair, so
unlike `pair_bound` every `(la, lb)` with each l at most 3 is written: 16
classes. FelizJueves picks a low-stack text at f itself (#586), so nothing here
chooses. Its two record vectors are split as mmd names them, `pa_*` = P - A
(A the first shell) and `pb_*` = P - B. It has no shuffle, so it is emitted
with the default options and its text has no width.

FelizJueves needs only the standard library to import, so this needs no
virtualenv.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "deps/FelizJueves/src"))

from felizjueves.emit import gpu  # noqa: E402
from felizjueves.emit.gpu import WAVEFRONT, Specialization  # noqa: E402
from felizjueves.eri.k_exchange.kernel_kblock import build_kblock_kernel  # noqa: E402
from felizjueves.eri.k_exchange.kernel_kblock_rolled import (  # noqa: E402
    build_kblock_rolled_kernel,
    tables,
)
from felizjueves.eri.k_exchange.kernel_record_bound import (  # noqa: E402
    build_record_bound_kernel,
)
from felizjueves.ir.passes import split_axes  # noqa: E402
from felizjueves.performance.device import by_name  # noqa: E402

# The kernels' L limit: f. It must equal kExchangeMaxMomentum
# (exchange_helpers_bridge.h), which exchange_launch.cu and mmd_launch.cu check
# against the tables written here.
MAX_L = 3
# The largest momentum `kb` is written at: through d kb, at f kbr (FelizJueves
# #580). A `kb` class with an f shell must never be written: `kb_33_33` is 22 MB
# of source and was still in cicc after 2.7 h (FelizJueves #569).
KB_MAX_L = 2
# With any_wavefront the arch fixes nothing in the text; the gpu backend
# still takes one.
ARCH = "sm_86"

OUT = ROOT / "src/integrals/eri/integral_direct/exchange/kernels"
FAMILY = "kb"
ROLLED_FAMILY = "kbr"
# kbr's class tables, one header beside its units (module docstring).
TABLES_HEADER = "kbr_tables.h"

# The record vectors, as mmd names them (module docstring).
AXES = {
    "P": ("p_x", "p_y", "p_z"),
    "XPA": ("pa_x", "pa_y", "pa_z"),
    "XPC": ("pc_x", "pc_y", "pc_z"),
    "Q": ("q_x", "q_y", "q_z"),
    "XQB": ("qb_x", "qb_y", "qb_z"),
    "XQD": ("qd_x", "qd_y", "qd_z"),
}
# kbr forms no E, so only its two centres are split.
ROLLED_AXES = {key: AXES[key] for key in ("P", "Q")}
OPTIONS = {"block_dim": (1, WAVEFRONT), "any_wavefront": True}

# The record bound: its own directory under mmd/kernels/ (whose
# .clang-format is the same DisableFormat), its two record vectors split.
PAIRS_OUT = ROOT / "src/integrals/mmd/kernels"
BOUND_FAMILY = "record_bound"
BOUND_AXES = {"XPA": ("pa_x", "pa_y", "pa_z"), "XPB": ("pb_x", "pb_y", "pb_z")}


def classes() -> list[tuple[int, int, int, int]]:
    ls = range(MAX_L + 1)
    return [(la, lc, lb, ld) for la in ls for lc in ls for lb in ls for ld in ls]


def kb_classes() -> list[tuple[int, int, int, int]]:
    return [cls for cls in classes() if max(cls) <= KB_MAX_L]


def kbr_classes() -> list[tuple[int, int, int, int]]:
    return [cls for cls in classes() if max(cls) > KB_MAX_L]


def bound_classes() -> list[tuple[int, int]]:
    ls = range(MAX_L + 1)
    return [(la, lb) for la in ls for lb in ls]


def kb(cls: tuple[int, int, int, int]):
    return split_axes(build_kblock_kernel(cls), AXES)


def kbr(cls: tuple[int, int, int, int]):
    return split_axes(build_kblock_rolled_kernel(cls), ROLLED_AXES)


def record_bound(cls: tuple[int, int]):
    return split_axes(build_record_bound_kernel(cls), BOUND_AXES)


def main() -> int:
    arch = by_name(ARCH)
    emit_family(arch, OUT / FAMILY, FAMILY, kb_classes(), kb, OPTIONS)
    emit_family(arch, OUT / ROLLED_FAMILY, ROLLED_FAMILY, kbr_classes(), kbr, OPTIONS)
    write(OUT / ROLLED_FAMILY / TABLES_HEADER, kbr_tables(kbr_classes()))
    emit_family(
        arch, PAIRS_OUT / BOUND_FAMILY, BOUND_FAMILY, bound_classes(), record_bound, {}
    )
    return 0


def emit_family(arch, out: Path, family: str, every, build, options) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for cls in every:
        kernel = build(cls)
        spec = Specialization(family, cls)
        write(out / f"{kernel.name}.cu", gpu.emit(kernel, arch, spec, **options))
    # Any class's kernel spells the primary: one family, one parameter block.
    write(out / spec.header, gpu.emit_template(kernel, arch, spec, every, **options))


def kbr_tables(every) -> str:
    """kbr's five class tables (FelizJueves's `kernel_kblock_rolled.tables`)
    as device arrays: the CSR pattern of E^{l1 l2} once per pair, as
    `kbr_e_ptr` / `kbr_e_col` (a class's `ac_*` from its bra pair, `bd_*`
    from its ket pair), and `hpq` once per `(lp, lq)`, each array with a
    host-side table of where each block starts in it."""
    pairs = sorted(
        {(la, lc) for la, lc, _, _ in every} | {(lb, ld) for _, _, lb, ld in every}
    )
    sums = sorted({(la + lc, lb + ld) for la, lc, lb, ld in every})
    e_ptr, e_col, hpq = [], [], []
    ptr_at, col_at, hpq_at = {}, {}, {}
    for l1, l2 in pairs:
        t = tables((l1, l2, 0, 0))
        ptr_at[(l1, l2)], col_at[(l1, l2)] = len(e_ptr), len(e_col)
        e_ptr += t["ac_ptr"]
        e_col += t["ac_p"]
    for lp, lq in sums:
        # hpq depends only on la + lc and lb + ld.
        hpq_at[(lp, lq)] = len(hpq)
        hpq += tables((lp, 0, lq, 0))["hpq"]
    for cls in every:
        la, lc, lb, ld = cls
        t = tables(cls)
        assert t["ac_ptr"] == tuple(e_ptr[ptr_at[(la, lc)] :][: len(t["ac_ptr"])])
        assert t["ac_p"] == tuple(e_col[col_at[(la, lc)] :][: len(t["ac_p"])])
        assert t["bd_ptr"] == tuple(e_ptr[ptr_at[(lb, ld)] :][: len(t["bd_ptr"])])
        assert t["bd_q"] == tuple(e_col[col_at[(lb, ld)] :][: len(t["bd_q"])])
        assert t["hpq"] == tuple(hpq[hpq_at[(la + lc, lb + ld)] :][: len(t["hpq"])])
    n_l, n_s = MAX_L + 1, 2 * MAX_L + 1

    def at(table: dict, n: int) -> str:
        rows = ",\n".join(
            "    {" + ", ".join(str(table.get((i, j), -1)) for j in range(n)) + "}"
            for i in range(n)
        )
        return "{\n" + rows + "\n}"

    def values(v: list[int]) -> str:
        return "{" + ", ".join(map(str, v)) + "}"

    return f"""\
// kbr's class tables: written by tools/emit_exchange_kernels.py from
// FelizJueves's kernel_kblock_rolled.tables. Do not edit.
// Included by exchange_launch.cu alone: it defines the arrays.
//
// The three device arrays are not const. The host only takes their address
// (wwrGetSymbolAddress) and no device code in this TU names them, and clang's
// HIP device compile drops a const __device__ variable no device code uses
// while the host side still registers it: the runtime then aborts with
// "Cannot create GlobalVar Obj for symbol". __constant__ would also keep them,
// but they are ~190 KiB, over CUDA's 64 KiB constant bank.
#pragma once

#include <stdint.h>

// E^{{l1 l2}}'s CSR pattern, every pair a kbr class reads: its rows' ptr at
// kbr_e_ptr + kbr_e_ptr_at[l1][l2], their Hermite columns at
// kbr_e_col + kbr_e_col_at[l1][l2] (-1: no class reads the pair).
__device__ int32_t kbr_e_ptr[] = {values(e_ptr)};
__device__ int32_t kbr_e_col[] = {values(e_col)};
constexpr int32_t kbr_e_ptr_at[{n_l}][{n_l}] = {at(ptr_at, n_l)};
constexpr int32_t kbr_e_col_at[{n_l}][{n_l}] = {at(col_at, n_l)};

// hpq of lp = la + lc and lq = lb + ld, row-major, at
// kbr_hpq + kbr_hpq_at[lp][lq] (-1: no class has the pair).
__device__ int32_t kbr_hpq[] = {values(hpq)};
constexpr int32_t kbr_hpq_at[{n_s}][{n_s}] = {at(hpq_at, n_s)};
"""


def write(path: Path, text: str) -> None:
    path.write_text(text)
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    sys.exit(main())
