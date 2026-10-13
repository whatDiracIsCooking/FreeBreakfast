"""aCD and acCD atomic Cholesky decomposition, in Python.

The full pipeline: pivoted/incomplete Cholesky of an atom's own two-electron
integrals (aCD), then a second pivoted Cholesky over uncontracted primitives
graded by real solid harmonic content (acCD, the "compact" pass). The raw
two-electron integrals come from PySCF (``mol.intor``) rather than a
hand-rolled GPU kernel, which lets this package work at arbitrary angular
momentum and with native real solid harmonics -- past the L<=2, Cartesian-only
shape of ``src/basis``. There is no C++ counterpart in this tree.

Modules, in dependency order:

- ``angular``    -- small Cartesian/normalization helpers shared below.
- ``harmonics``  -- real solid harmonic graded transform, at arbitrary combined
                    angular momentum (memoized per combined_l).
- ``eri_quartet``-- one atom's raw AO-basis (ab|cd) tensor, in this repo's own
                    AO convention.
- ``cholesky``   -- aCD: ``atomic_cholesky`` over the dense AO-pair space.
- ``compact``    -- acCD: the candidate-pool Cholesky at a fixed target angular
                    momentum.
- ``cli``        -- ``python3 -m atomic_cd.cli`` end-to-end driver.

Needs PySCF and numpy, which live in the opt-in ``tools`` dependency group of
``pyproject.toml`` (not ``dev``, so the image and the push gate do not carry
them). From the repo root::

    uv sync --group tools
    uv run --group tools python -m tools.atomic_cd.cli C sto-3g
"""

from __future__ import annotations
