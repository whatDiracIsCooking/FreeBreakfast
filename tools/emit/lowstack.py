"""The kernel classes the emitters take in a lower-stack form than
FelizJueves's default (#1263).

CUDA reserves a kernel's per-thread stack for every resident thread from
its first launch on (104,448 threads on the RTX 3080), outside the pool,
and `DeviceResources::kKernelStackBytes` keeps room for the binary's
largest stack in every run (src/core/wwr/README.md, "Kernel stacks outside
the pool"). The target is STACK_TARGET bytes a thread. Every class listed
here needed more than that in FelizJueves's default form, read off the
built binary (`cuobjdump --dump-resource-usage`, sm_86) at FelizJueves
2b3c019, and is emitted in the lowest-stack form FelizJueves offers:

- LOWSTACK: `lowstack=True`, the family's low-stack form, which the
  default picks only from a higher `L` floor (`potential_hessian<1,2>`
  17,784 -> 5,296 B; every class here lands under the target), and the
  low-stack `d2s` / `d2t` (unrolled text over `t = 0` tables before,
  `kinetic_hessian<3,3>` 23,192 -> 1,952 B).
- COMPACT: `compact=True`, the low-stack form with its tables holding only
  the reachable `t <= i + j`: `d2eri3`'s bra tables (#1263, `d2eri3<3,3,6>`
  22,560 -> 18,912 B), and the low-stack `d2v`'s and `geri3`'s and every
  `xeri3`'s (#1265). Bitwise the same values. Timed on the RX 9060 XT
  (CH2O cc-pVTZ / cc-pVTZ-acCD-6-6): `geri3` / `xeri3` 0.87-1.01x the dense
  form's time, so every f class over the target takes it; the compact
  `d2v` was 0.98x at `<1,3>`, 1.01x at `<2,3>`, but 1.05x at `<2,2>` and
  1.08x at `<3,3>`, which stay dense (`potential_hessian<3,3>`, 16,912 B,
  is then the largest stack in the binary).
- SHARED: `shared=True`, `d2eri3`'s warp-shared tower (#1265): the bra
  tables and the Hermite tower `R` in shared memory, formed once per bra
  record / record pair instead of once per item, bitwise the per-item form
  (`d2eri3<3,3,6>` 18,912 B before). Its speed depends on the class, so it
  is taken only where it was measured as fast or faster: the RX 9060 XT,
  CH2O cc-pVDZ / cc-pVDZ-JKFIT and cc-pVTZ / cc-pVTZ-acCD-6-6, at most
  1.01x the per-item form's time in every system that launches the class.
  The classes over the target it was slower at, 1.06-2.1x (`<2,2,2>`,
  `<2,2,6>`, `<1,3,6>`, `<2,3,0..2>`, `<3,3,0>`, `<3,3,3>`), stayed
  per-item. SPAN below has since taken ten of its classes and four of
  those.
- SPAN: `span=True`, `d2eri3`'s spanning windows (#1346): the warp-shared
  tower's split with its phases in item space -- windows of 32
  consecutive items that span record pairs and bra records, each spanned
  bra record's tables formed an axis a lane -- bitwise the per-item form.
  It replaces the warp-shared tower, or the per-item form, wherever it was
  the fastest of the three on the RX 9060 XT, the same two systems, the
  median over 9 alternated rounds of paired ratios: 0.75-0.99x the
  per-item form at the 15 classes here, and faster than the warp-shared
  tower at each of them that had it. It was 1.09-1.65x the per-item form
  at `<1,2,4>`, `<1,3,0..3>`, `<2,2,0..2>`, `<2,3,0..2>` on cc-pVTZ /
  acCD (cc-pVDZ / JKFIT's `<2,2,0..3>` were 0.80-0.89x), and slower than
  the warp-shared tower at `<1,2,5>`, `<1,3,3..5>`, `<2,2,3..5>`,
  `<2,3,5>`, which keep their forms. The RTX 3080 is not timed (#1336).

- FUSED: the `fused` lever of FelizJueves's `[p|q]` kernel (its #344),
  `e_hermite`'s low-stack form (#1337): each `[r]^(0)` is contracted into
  `J_p` the moment the recurrence forms it, so one target is live at a
  time instead of all `n_t(L)` of them (`e_hermite<7,6>` 15,808 -> 2,920 B
  stand-alone). It adds a slot's terms one at a time rather than as one
  balanced tree a ket, so it is NOT bitwise the literal kernel; it agrees
  with it to the summation bound (FelizJueves's test_kernel_fused.py).
  It is a speed lever as well, so it is taken at every class where it is
  at most 1.01x the literal kernel's time (#1345), not only over the
  target. Timed on the RX 9060 XT (hipcc 7.2.4, synthetic records, 1024
  bras x 1024 kets to L = 9 and 256 x 256 above, `tau = 0`; the median of
  5 and of 11 alternated runs): 0.29-0.41x at the four over the target and
  at (4,6), 0.67-0.95x from L = 5 up, and level (0.98-1.00x) at the low
  classes and wherever `lq = 0`, where one term reaches each slot and the
  two forms agree bitwise. It is slower at (2,2), 1.06x, and over 1.01x in
  a run at (2,1) and (7,0), which keep the literal kernel. The RTX 3080 is
  not timed (#1336).

The classes over the target that no form brings under it (the per-item
`d2eri3`s above, the low-stack `geri3` / `xeri3` / `gk` /
`d2v` at f, `kb<2,2,2,*>`, `d2metric<6,6>`, `eri3<3,3,6>`) keep their form;
the largest of them all sets `kKernelStackBytes`, which ctest's
`core_kernel_stack_bound` holds to the built binary.

Every other class is emitted exactly as FelizJueves's default builds it.
"""

from __future__ import annotations

STACK_TARGET = 8192

# family -> the classes emitted with lowstack=True.
LOWSTACK: dict[str, frozenset[tuple[int, ...]]] = {
    "overlap_hessian": frozenset({(3, 3)}),
    "kinetic_hessian": frozenset({(2, 2), (2, 3), (3, 3)}),
    "potential_hessian": frozenset({(0, 3), (1, 2)}),
    "potential_gradient": frozenset({(2, 2)}),
    "gjab": frozenset({(2, 2)}),
    "gk": frozenset({(2, 2, 0, 0)}),
    "eri3": frozenset({(1, 2, 5), (1, 3, 3), (1, 3, 4), (2, 2, 3)}),
    "gmetric": frozenset({(3, 6), (4, 5)}),
    "geri3": frozenset({(1, 2, 1), (2, 2, 0)}),
    "d2metric": frozenset({(4, 4)}),
}

# family -> the classes emitted with compact=True (low-stack by default).
_F_ERI3 = frozenset({(3, 3, lP) for lP in range(7)} | {(2, 3, 5), (2, 3, 6)})
COMPACT: dict[str, frozenset[tuple[int, ...]]] = {
    "potential_hessian": frozenset({(1, 3), (2, 3)}),
    "geri3": _F_ERI3,
    "xeri3": _F_ERI3,
    "d2eri3": frozenset(
        {(0, 3, lP) for lP in (5, 6)}
        | {(1, 2, lP) for lP in (4, 5, 6)}
        | {(1, 3, lP) for lP in range(7)}
        | {(2, 2, lP) for lP in range(7)}
        | {(2, 3, lP) for lP in range(7)}
        | {(3, 3, lP) for lP in range(7)}
    ),
}


# family -> the classes emitted with shared=True (over compact tables).
SHARED: dict[str, frozenset[tuple[int, ...]]] = {
    "d2eri3": frozenset(
        {(1, 3, lP) for lP in (3, 4, 5)}
        | {(2, 2, lP) for lP in (3, 4, 5)}
        | {(2, 3, 5)}
    ),
}


# family -> the classes emitted with span=True (over compact tables).
SPAN: dict[str, frozenset[tuple[int, ...]]] = {
    "d2eri3": frozenset(
        {(0, 3, 5), (0, 3, 6), (1, 2, 6), (1, 3, 6), (2, 2, 6)}
        | {(2, 3, lP) for lP in (3, 4, 6)}
        | {(3, 3, lP) for lP in range(7)}
    ),
}


# family -> the classes built with the `fused` lever in their Spec.
FUSED: dict[str, frozenset[tuple[int, ...]]] = {
    "e_hermite": frozenset(
        {(lp, lq) for lp in range(8) for lq in range(7)} - {(2, 1), (2, 2), (7, 0)}
    ),
}


def levers(family: str, cls: tuple[int, ...]) -> tuple[str, ...]:
    """The Spec levers that pick `cls`'s form in `family`: empty for
    FelizJueves's default."""
    return ("fused",) if tuple(cls) in FUSED.get(family, ()) else ()


def forced(family: str, cls: tuple[int, ...]) -> dict[str, bool]:
    """The keyword arguments that pick `cls`'s form in `family`'s builder:
    empty for FelizJueves's default."""
    out: dict[str, bool] = {}
    if tuple(cls) in LOWSTACK.get(family, ()):
        out["lowstack"] = True
    if tuple(cls) in COMPACT.get(family, ()):
        out["compact"] = True
    if tuple(cls) in SHARED.get(family, ()):
        out["shared"] = True
    if tuple(cls) in SPAN.get(family, ()):
        out["span"] = True
    return out
