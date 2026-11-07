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
