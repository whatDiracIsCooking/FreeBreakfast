"""acCD: a SECOND, independent pivoted Cholesky over UNCONTRACTED
PRIMITIVE-pair candidates at one fixed target angular momentum `l`, removing
linear dependence among the primitives to produce far fewer, genuinely
portable auxiliary shells. Built on ``harmonics.py``.

WHERE A CANDIDATE COMES FROM. A shell pair (sa, sb) with angular momenta
(la, lb) has combined momentum L = la+lb; ``harmonics.graded_transform(L)``
splits its ``ncart(L)`` Cartesian monomials into blocks of pure angular
momentum L, L-2, .... For a target l, every shell pair whose combined L is
>= l with matching parity contributes one candidate per UNCONTRACTED
primitive pair (a_i, b_j): exponent zeta = a_i+b_j, angular content the
l-block's representative column of that shell pair's L-degree transform (any
one of its 2l+1 columns -- they are radially equivalent). The original
shells' contraction coefficients play NO role -- acCD rediscovers its own
weights entirely from the pivoted Cholesky below.

RAW SAME-CENTER INTEGRAL. The bilinear form needs the raw 2-center Coulomb
value (A|C) between two same-center, single-primitive, unnormalized monomial
Gaussians. ``_same_center_raw_block`` below gets it from PySCF's own
``int2c2e`` (the 2-center Coulomb-metric integral ``pyscf.df`` uses for the
density-fitting (P|Q) metric), evaluated directly between two single-primitive
shells, then divided by PySCF's own per-l primitive normalization
(``angular.primitive_norm``) to undo the normalization PySCF always applies.
(The obvious alternative -- a 4-center ERI with an angular-momentum-0,
exponent-0 "identity" filler shell in the B/D slots -- is unavailable here:
PySCF always normalizes a declared shell to unit self-overlap, which diverges
for a constant function.) Checked against the closed-form same-center s-s
Coulomb integral ``2*pi^(5/2) / (z1*z2*sqrt(z1+z2))`` (exact to float64
precision), and against the physical requirement that, for two same-center
Gaussians of definite angular momentum l, `radial_eri` agree across every one
of the 2l+1 harmonic components of that l and vanish between different l's
(also exact, to within Gram-Schmidt conditioning).

PYSCF REORDERS SHELLS. ``_same_center_raw_block`` builds its throwaway `Mole`
from a basis STRING with shell_a (ell1) listed before shell_c (ell2), but
PySCF sorts a single atom's shells by ANGULAR MOMENTUM ascending regardless of
input order, so when ell1 > ell2 the lower-momentum shell lands at bas index
0. Slicing at a fixed offset would then read the wrong sub-block. That case is
reached whenever a target l draws candidates from more than one combined_l
(every l with `harmonics` blocks below it; l=0 always), so each shell's block
is located from the ``ell1 <= ell2`` ordering PySCF actually produced. With
that, carbon/cc-pVDZ l=0 reconstructs to ~5e-9, in line with l=1..4, and the
matrix `compact_cholesky_for_l` decomposes is numerically PSD (min eigenvalue
~1e-12, pure roundoff).

WHAT THIS DOES NOT DO. ``AuxShell`` is spherical, not Cartesian, and this
module does not evaluate density-fitting integrals from these shells back onto
an orbital basis. There is no L ceiling (harmonics.py has none); cli.py adds
the file I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from pyscf import gto

from .angular import ncart, primitive_norm
from .harmonics import graded_transform

# spectroscopic notation, ell=0..10 -- far past anything this reaches
_ANGULAR_LETTERS = "SPDFGHIKLMN"


@dataclass(frozen=True)
class AuxPrimitive:
    """One primitive of a portable acCD auxiliary shell: exponent and its
    Cholesky-derived contraction coefficient, as given (no normalization)."""

    exponent: float
    coefficient: float


@dataclass(frozen=True)
class AuxShell:
    """One acCD auxiliary shell: angular momentum ell plus its primitive
    contraction, applying identically to each of the 2*ell+1 real solid
    harmonic components of that ell -- NOT a Cartesian src/basis Shell (see
    module docstring)."""

    ell: int
    primitives: list[AuxPrimitive]


@dataclass(frozen=True)
class ShellSpec:
    """A center-independent shell: angular momentum and primitive exponents
    only -- acCD's candidates need nothing else (see module docstring: the
    original contraction coefficients play no role)."""

    ell: int
    exponents: list[float]


@dataclass(frozen=True)
class _Candidate:
    exponent: float
    combined_l: int
    embedding: np.ndarray  # length ncart(combined_l)


def build_candidates(shells: list[ShellSpec], target_l: int) -> list[_Candidate]:
    """Every uncontracted-primitive candidate at `target_l`, from every shell
    pair (sa <= sb) of `shells` whose combined momentum reaches `target_l`
    with the right parity. sa <= sb only: (sb, sa) would recompute the
    identical candidate set (the underlying matrix is symmetric under
    swapping bra/ket shells), so including both would just double-count."""
    candidates: list[_Candidate] = []
    n = len(shells)
    for sa in range(n):
        for sb in range(sa, n):
            combined_l = shells[sa].ell + shells[sb].ell
            if combined_l < target_l or (combined_l - target_l) % 2 != 0:
                continue
            embedding = graded_transform(combined_l).block_columns(target_l)[:, 0]
            for ea in shells[sa].exponents:
                for eb in shells[sb].exponents:
                    candidates.append(_Candidate(ea + eb, combined_l, embedding))
    return candidates


def _same_center_raw_block(
    ell1: int, zeta1: float, ell2: int, zeta2: float
) -> np.ndarray:
    """The raw (ncart(ell1), ncart(ell2)) two-electron block between two
    same-center, single-primitive, coefficient-1 Cartesian monomial Gaussians
    -- see module docstring for the int2c2e/primitive_norm derivation, and its
    "PYSCF REORDERS SHELLS" paragraph for why the two shells are located by
    angular momentum rather than by their position in the basis string."""
    mol = gto.Mole()
    mol.atom = [["H", (0.0, 0.0, 0.0)]]
    shell_a = f"H {_ANGULAR_LETTERS[ell1]}\n   {zeta1:.16f} 1.0\n"
    shell_c = f"H {_ANGULAR_LETTERS[ell2]}\n   {zeta2:.16f} 1.0\n"
    mol.basis = {"H": gto.basis.parse(shell_a + shell_c)}
    mol.cart = True
    mol.spin = 1  # keeps mol.build() happy; intor() ignores occupation entirely
    mol.verbose = 0
    mol.build()
    block = mol.intor("int2c2e")
    # PySCF sorts one atom's shells by angular momentum ascending, ignoring
    # the order given above: shell_a (ell1) is only at bas index 0 when
    # ell1 <= ell2.
    if ell1 <= ell2:
        n_first = ncart(ell1)
        raw = block[:n_first, n_first:]
    else:
        n_first = ncart(ell2)
        raw = block[n_first:, :n_first]
    return raw / (primitive_norm(zeta1, ell1) * primitive_norm(zeta2, ell2))


def radial_eri(k1: _Candidate, k2: _Candidate) -> float:
    """The two-electron radial matrix element between two candidates -- the
    bilinear form through the raw monomial-pair block (module docstring)."""
    m_raw = _same_center_raw_block(
        k1.combined_l, k1.exponent, k2.combined_l, k2.exponent
    )
    return float(k1.embedding @ m_raw @ k2.embedding)


def _dense_matrix(candidates: list[_Candidate]) -> np.ndarray:
    n = len(candidates)
    m = np.empty((n, n))
    for i in range(n):
        for j in range(i, n):
            v = radial_eri(candidates[i], candidates[j])
            m[i, j] = v
            m[j, i] = v
    return m


@dataclass
class AccdResult:
    """`vectors[J][k]` is Cholesky vector J's coefficient on candidate k
    (`exponents[k]` its exponent) -- directly the primitives of the portable
    shell `to_aux_shells` builds from vector J."""

    ell: int
    threshold: float
    exponents: list[float] = field(default_factory=list)
    vectors: list[np.ndarray] = field(default_factory=list)


def compact_cholesky_for_l(
    shells: list[ShellSpec], target_l: int, threshold: float
) -> AccdResult:
    """Run acCD's radial pivoted Cholesky at `target_l`: build every
    uncontracted-primitive candidate, form their dense two-electron matrix,
    and decompose it with the same pivot/project/normalize loop as
    `cholesky.atomic_cholesky`, stopping once every residual diagonal is
    below `threshold`."""
    candidates = build_candidates(shells, target_l)
    result = AccdResult(target_l, threshold, [c.exponent for c in candidates])
    if not candidates:
        return result

    m = _dense_matrix(candidates)
    d = np.diagonal(m).copy()
    n = len(candidates)
    for _ in range(n):
        pivot = int(np.argmax(d))
        if d[pivot] < threshold:
            break
        col = m[:, pivot].copy()
        for prev in result.vectors:
            coeff = prev[pivot]
            if coeff != 0.0:
                col -= coeff * prev
        col *= 1.0 / math.sqrt(d[pivot])
        d -= col * col
        result.vectors.append(col)
    return result


def reconstruct(result: AccdResult) -> np.ndarray:
    """sum_J v_J[k1] v_J[k2], for direct comparison against the matrix
    `compact_cholesky_for_l` decomposed."""
    n = len(result.exponents)
    out = np.zeros((n, n))
    for v in result.vectors:
        out += np.outer(v, v)
    return out


def reconstruction_error(shells: list[ShellSpec], result: AccdResult) -> float:
    """max |reconstruct(result) - M|, M rebuilt independently (bypassing the
    Cholesky loop entirely)."""
    candidates = build_candidates(shells, result.ell)
    if not candidates:
        return 0.0
    return float(np.abs(reconstruct(result) - _dense_matrix(candidates)).max())


def to_aux_shells(result: AccdResult) -> list[AuxShell]:
    """Package `result`'s accepted vectors into portable AuxShell objects, one
    per vector, dropping exact numerical zeros."""
    shells: list[AuxShell] = []
    for v in result.vectors:
        primitives = [
            AuxPrimitive(result.exponents[k], float(v[k]))
            for k in range(len(v))
            if v[k] != 0.0
        ]
        shells.append(AuxShell(result.ell, primitives))
    return shells
