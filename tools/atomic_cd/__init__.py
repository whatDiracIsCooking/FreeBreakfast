"""aCD and acCD atomic Cholesky decomposition, in Python.

The full pipeline: pivoted/incomplete Cholesky of an atom's own two-electron
integrals over its uncontracted primitives (aCD), whose pivots pick the
primitive pairs worth a candidate; a second pivoted Cholesky over those
candidates per angular momentum, whose pivots are the auxiliary basis (acCD,
the "compact" pass; Lehtola, JCTC 17, 6886 (2021)); and an SVD contraction of
them against the atom's own orbital products (Lehtola, JCTC 19, 6242
(2023)). The raw
two-electron integrals come from PySCF (``mol.intor``) rather than a
hand-rolled GPU kernel, which lets this package work at arbitrary angular
momentum -- past the f ceiling (``kMaxAngularMomentum``) of the GPU engines.
There is no C++ counterpart to aCD/acCD in this tree. The C++ density fitting
(``src/integrals/eri/density_fitting``) fits over a ready-made auxiliary
basis (``data/aux_basis/``) and does not generate one.

Modules, in dependency order:

- ``cholesky``   -- the pivoted Cholesky both stages run: aCD on the
                    primitive ERIs, acCD on each l's candidate metric.
- ``eri_quartet``-- the orbital primitives, their ERIs (eight-fold packed)
                    and aCD's screening of primitive pairs; Mole and BSE
                    input helpers.
- ``compact``    -- acCD: the candidate-pool Cholesky at a fixed target angular
                    momentum, and its pivots as uncontracted auxiliary shells.
- ``contract``   -- SVD contraction of those pivots against the atom's own
                    orbital products (Lehtola 2023), one general contraction
                    per l.
- ``df_check``   -- how well a generated auxiliary basis fits: one-centre ERI
                    error and DF-RHF energy error for small molecules.
- ``cli``        -- ``python3 -m atomic_cd.cli`` end-to-end driver, one
                    element.
- ``aux_set``    -- every element of an orbital BSE document through ``cli``,
                    as one named set for ``data/aux_basis/``.

Needs PySCF and numpy, which live in the opt-in ``tools`` dependency group of
``pyproject.toml`` (not ``dev``, so the image and the push gate do not carry
them). From the repo root (inside the devcontainer, prefix
``UV_PROJECT_ENVIRONMENT=$PWD/.venv`` so the group does not go into the
image's ``/opt/venv``; ``tools/README.md``)::

    uv sync --group tools
    uv run --group tools python -m tools.atomic_cd.cli C sto-3g
"""

from __future__ import annotations
