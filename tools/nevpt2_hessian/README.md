# `tools/nevpt2_hessian/`

The SC- and PC-NEVPT2 nuclear Hessian references (#803):
`test/gtest/data/nevpt2_hessian/`, whose README has the cases, the numbers
and the keys.

- `nevpt2_hessian.py` writes them: a fourth-order central stencil at h, h/2
  and h/4 over the fourth-order central gradient of re-converged PySCF
  energies, guarded, ratio-checked against 16, symmetry- and
  bilinearity-checked. Its docstring is the method. Needs the `tools`
  dependency group (PySCF, block2), and reuses `tools/emit`'s
  `casscf_reference`, `nevpt2_reference`, `nevpt2_gradient_reference` and
  `qd_nevpt2_reference` (PC by the C++ step's cut).
- `stencil.py` is its arithmetic, standard library only:
  `test/pytest/test_nevpt2_hessian.py` checks it and recomputes every
  fixture from its recorded gradients on the push gate.

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    tools/nevpt2_hessian/nevpt2_hessian.py [--only CASE ...] [--jobs N]
```

`--jobs` is worker processes of one thread each, 8 by default and at most:
the box's per-agent budget. About 15 minutes for all four cases.

## The QD Hessian references (#953)

`test/gtest/data/qd_nevpt2_hessian/`, whose README has the cases, the
numbers and the keys.

- `qd_nevpt2_hessian.py` writes them: the same stencil, ladder, floor and
  symmetry check, over this tree's analytic SC- and PC-QD-NEVPT2 gradient
  G(W) at each tracked state's W^K = u_K u_K^T and each pair's W^P + W^Q.
  Its docstring is the method. Standard library only: it writes a request,
  runs the C++ driver over it, and judges the points.
- The driver is `free-breakfast-qd-nevpt2-hessian`
  (`tools/qd_nevpt2_hessian/main.cpp`, built with the tree): at every
  requested point the chain and guard of `tools/qd_nevpt2_gradient`'s
  `ChainSampler`, then `nevpt2::qd_sc_gradient` / `qd_pc_gradient` at
  every W. Needs a GPU.

```sh
tools/nevpt2_hessian/qd_nevpt2_hessian.py [--only CASE ...] \
    [--binary build/tools/qd_nevpt2_hessian/free-breakfast-qd-nevpt2-hessian] \
    [--points DIR [--keep]]
```

`--points DIR --keep` leaves the driver's raw points in `DIR`; `--points DIR`
alone judges them again without the GPU. About 6 minutes per ladder (106
points) on the RTX 3080, two ladders per case.

## SC's unequal-weight case (#1060)

`qd_weight_search.py` screens the skew case at sixteen weightings, at
their reference points alone, over `tools/emit/qd_nevpt2_reference.py`'s
SC-QD-NEVPT2: the inactive gaps, the smallest kept tuple norms per root and
class, the root gaps and SC's spectrum. It writes
`test/gtest/data/qd_nevpt2_hessian/qd_weight_search.json`, which also holds
the reference point that the laddered case
(`h2co_skew_ccpvdz_sa3_w424`, 0.4/0.2/0.4) is held to, and so runs first.
Needs the `tools` group (PySCF). A few minutes, no GPU.

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    tools/nevpt2_hessian/qd_weight_search.py
tools/nevpt2_hessian/qd_nevpt2_hessian.py --only h2co_skew_ccpvdz_sa3_w424
```

## The C2v A1/B2 coupling case (#1058)

`test/gtest/data/qd_nevpt2_hessian/h2nb_sto3g_sa2.json` and its
`.analysis.json`, whose README section has the numbers.

- `qd_coupling_analysis.py` writes the analysis first. Over
  `tools/emit/qd_nevpt2_reference.py`'s SC- and PC-QD-NEVPT2 at R0 and at
  two steps along each b2 coordinate, it records the tuples the guard drops
  for one ket and the other keeps, with their transition tangents, and PC's
  cut metric modes that the coordinate lifts with an overlap to the other
  state. Needs the `tools` group (PySCF). About a minute, no GPU.
- `qd_nevpt2_hessian.py` (`COUPLING_CASES`) then ladders SC's C_MN through
  the driver's `"gradients": false` mode. It judges each ladder and the pair's
  bilinearity, probes PC at one displaced point, and writes the record with
  both variants' expected refusals. Seconds on the RTX 3080.

```sh
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \
    tools/nevpt2_hessian/qd_coupling_analysis.py
tools/nevpt2_hessian/qd_nevpt2_hessian.py --only h2nb_sto3g_sa2
```

## The C2v A1/B2 whole Hessians (#1093)

`test/gtest/data/qd_nevpt2_hessian/h2nb_sto3g_sa2_hessian.json`, whose
README section has the numbers. `qd_nevpt2_hessian.py` (`SYMMETRIC_CASES`)
ladders the coupling case's state Hessians in its C2v-adapted coordinates,
each scheme skipping the coordinates, states and pairs its `not_laddered`
names: PC (2.5e-3) the two b2 vibrations, SC (3.125e-4) the three
symmetry-lowering vibrations and state 1. Its noise floor also reads σ_g
off the elements C2v forbids (`symmetry_sigma`). The coupling case's record
must exist first. About 2 minutes on the gfx1200.

```sh
tools/nevpt2_hessian/qd_nevpt2_hessian.py --only h2nb_sto3g_sa2_hessian
```

## The SA3 crossing (#1059)

`test/gtest/data/qd_nevpt2_hessian/ethylene_ccpvdz_sa3_{sc,pc}_meci.json`
and `ethylene_meci.json`, whose README section has the numbers.

- `qd_meci.py` writes the MECIs first: ethylene's SC- and PC-QD-NEVPT2
  states 0 and 1 over its DF-SA3-CASSCF(2,2), each converged by the tree's
  `[optimize]` MECI mode through the `free-breakfast` binary
  (`examples/qd_meci.toml`'s config), its end-point criteria and the third
  state's margin checked and recorded. Standard library only; about a
  minute per variant on the RTX 3080.
- `qd_nevpt2_hessian.py` (`CROSSING_CASES`) then ladders, at each MECI, the
  pair sum G(W^0 + W^1), the fixed seam W and the third state. The driver's
  `crossing` request tracks the pair by its two-state eigenspace
  (`src/pipeline/hessian`'s `tracked_qd_pair`, which the driver calls)
  rather than by u_K, and takes no W^0 or W^1. The seam's W is the
  eigenspace's own, not the eigensolver's R_0, R_1 (which a rerun returns
  turned by ~1e-7 rad and signed differently, #1098): ½ s (V_0 V_1ᵀ + V_1
  V_0ᵀ), V the Löwdin-orthonormalized projections of two CASSCF roots onto
  the eigenspace (`hessian::df_qd_sc_nevpt2_crossing`'s basis) and s the
  sign of H_eff between those roots; each seam entry records it (`w`,
  `roots`, `sign`). About 2.3 minutes per case.

```sh
tools/nevpt2_hessian/qd_meci.py [--binary build/src/free-breakfast]
tools/nevpt2_hessian/qd_nevpt2_hessian.py \
    --only ethylene_ccpvdz_sa3_sc_meci ethylene_ccpvdz_sa3_pc_meci
```
