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
the box's per-agent budget. About 12 minutes for all three cases.
