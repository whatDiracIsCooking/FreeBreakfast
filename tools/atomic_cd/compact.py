"""acCD: a SECOND, independent pivoted Cholesky over UNCONTRACTED
PRIMITIVE-pair candidates at one fixed target angular momentum `l`, whose
pivots are the auxiliary basis -- the procedure of Lehtola, J. Chem. Theory
Comput. 17, 6886 (2021), arXiv:2106.11081, Sec. 3.

WHERE A CANDIDATE COMES FROM. The orbital basis is fully uncontracted into
its distinct (ell, exponent) primitives (``eri_quartet.primitives``); the
original contraction coefficients play no role. The pool is a list of
primitive pairs: every pair (``eri_quartet.all_pairs``), or only those aCD's
pivots land on (``eri_quartet.screen_pairs``, the paper's default "reduced"
pool). A same-center primitive pair (la, a), (lb, b) is
r^(la+lb) exp(-(a+b) r^2) times a product of two real spherical harmonics,
which couples to every l with |la-lb| <= l <= la+lb and la+lb+l even (the
Gaunt selection rule); each such l gets one candidate from the pair. At
l < la+lb that candidate's true radial part is r^(la+lb-l) * r^l exp(-(a+b)
r^2) -- not a Gaussian of angular momentum l (l=0 from p x p is
r^2 exp(-zeta r^2)). Lehtola's Eq. 16 replaces it with the pure spherical
GTO r^l exp(-zeta_eff r^2) of the same <r> (``effective_exponent``), so every
candidate, and every shell this module emits, is a plain single-primitive
spherical Gaussian (l, zeta_eff). At l = la+lb, zeta_eff = a+b exactly.

THE METRIC. Candidates are radial functions only: two same-center pure-l
Gaussians have a Coulomb integral that is identical across the 2l+1
components of l and zero between different m, so one component's (P|Q)
stands for the shell, and it has a closed form. The Fourier transform of
r^l exp(-a r^2) Y_lm is k^l exp(-k^2/4a) Y_lm up to a^-(l+3/2), so the
unnormalized (a|b) goes as 1/(a b (a+b)^(l+1/2)); normalized to (A|A) = 1
it is (2 sqrt(ab)/(a+b))^(l+1/2) = sech(ln(a/b)/2)^(l+1/2) -- a function of
the log-exponent spacing alone (compare the overlap's exponent l+3/2).
``coulomb_metric`` evaluates that, matching PySCF's ``int2c2e`` (the
density-fitting metric ``pyscf.df`` uses) to 1e-15 at l=0 and 3e-13 at
l=10, where the closed form is the more accurate of the two. Identical
(l, zeta_eff) candidates -- distinct pairs whose exponents sum to the same
zeta -- are dropped before the metric is built.

THE AUXILIARY BASIS. The pivots of a pivoted Cholesky of that metric, taken
as-is: one uncontracted shell (l, zeta_eff) per pivot (``to_aux_shells``).
Every non-pivot candidate is then fit by the pivots to within the threshold
(its Cholesky residual), which is the property an auxiliary basis needs; the
Cholesky VECTORS are Coulomb overlaps (phi_k|psi_J), not expansion
coefficients, and are kept only for ``reconstruction_error``.

WHAT THIS DOES NOT DO. This module does not evaluate density-fitting
integrals from these shells back onto an orbital basis; that is
``src/integrals/eri/density_fitting``'s job, over an aux basis it is handed
(``df_check`` measures it with PySCF). There is no l ceiling here, so a
target l can exceed the g limit of the C++ fitting kernels; cli.py adds the
file I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from pyscf import gto

from . import cholesky, eri_quartet

# two candidate exponents closer than this (relative) are one candidate
_DUPLICATE_RTOL = 1e-12


@dataclass(frozen=True)
class AuxShell:
    """One auxiliary shell of angular momentum ell: a general contraction,
    ``coefficients[f][k]`` being function f's coefficient on the normalized
    primitive spherical Gaussian r^ell exp(-exponents[k] r^2), each function
    for every one of the 2*ell+1 real solid harmonic components. acCD's own
    pivots are the one-primitive, one-function case (``to_aux_shells``);
    ``contract`` makes the rest."""

    ell: int
    exponents: tuple[float, ...]
    coefficients: tuple[tuple[float, ...], ...]


def pruned_max_l(element: str, l_obs: int, l_inc: int) -> int:
    """Lehtola's high-angular-momentum pruning, J. Chem. Theory Comput. 19,
    6242 (2023), arXiv:2306.11039, Eq. 9: keep target l up to
    max(2 l_occ, l_occ + l_obs + l_inc), where l_obs is the orbital basis's
    highest ell for `element` and l_occ the highest occupied ell of the
    neutral atom (Yang et al.'s ranges, which already give K/Ca d and Cs/Ba
    f). The first term fits occupied x occupied products, the second
    occupied x virtual ones; virtual x virtual products, the only source of
    the l above that, never enter a global-fit HF or MP2 energy. Not a
    bound on integral error -- the threshold's guarantee does not cover the
    dropped l."""
    z = gto.charge(element)
    l_occ = 0 if z <= 2 else 1 if z <= 18 else 2 if z <= 54 else 3
    return max(2 * l_occ, l_occ + l_obs + l_inc)


def effective_exponent(target_l: int, combined_l: int, zeta: float) -> float:
    """Lehtola's Eq. 16: the exponent of the pure GTO r^target_l
    exp(-zeta_eff r^2) whose <r> (Eq. 14, <r> = Gamma(l+2) /
    (Gamma(l+3/2) sqrt(2 zeta)) for r^l exp(-zeta r^2)) matches that of the
    primitive-pair product r^combined_l exp(-zeta r^2) -- Eq. 15 solved for
    zeta_eff. The scale is 1 at target_l = combined_l and falls monotonically
    below it (about 0.1375 for g x g -> s).

    Measured against plain zeta_eff = a+b on C/H/O (CO and CH4 DF-RHF,
    contracted, unscreened): as good or better at cc-pVDZ..cc-pVQZ
    (cc-pVTZ CO 1e-7 Eh against 5e-7), a tie at cc-pV5Z (both within
    6e-8, each ahead on one molecule); the one-centre ERI error favours
    neither consistently."""
    log_scale = (
        math.lgamma(target_l + 2)
        + math.lgamma(combined_l + 1.5)
        - math.lgamma(combined_l + 2)
        - math.lgamma(target_l + 1.5)
    )
    return math.exp(2.0 * log_scale) * zeta


def build_candidates(
    pairs: list[eri_quartet.PrimitivePair], target_l: int
) -> np.ndarray:
    """Every distinct candidate exponent at `target_l`, ascending: one
    zeta_eff per primitive pair of the pool `pairs` that couples to
    `target_l` (module docstring)."""
    exponents: list[float] = []
    for (la, a), (lb, b) in pairs:
        if not abs(la - lb) <= target_l <= la + lb:
            continue
        if (la + lb - target_l) % 2 != 0:
            continue
        exponents.append(effective_exponent(target_l, la + lb, a + b))
    exponents.sort()
    distinct: list[float] = []
    for e in exponents:
        if not distinct or e - distinct[-1] > _DUPLICATE_RTOL * e:
            distinct.append(e)
    return np.asarray(distinct)


def coulomb_metric(ell: int, exponents: np.ndarray) -> np.ndarray:
    """The (n, n) Coulomb metric (P|Q) between same-center single-primitive
    spherical Gaussians of angular momentum `ell`, one per exponent, in the
    given order, each normalized to (P|P) = 1: sech(ln(a/b)/2)^(ell+1/2)
    (module docstring). Through the log-exponents, so the diagonal is
    exactly 1 and nothing overflows however far apart the exponents are."""
    t = np.log(exponents)
    return np.cosh(0.5 * (t[:, None] - t[None, :])) ** -(ell + 0.5)


@dataclass
class AccdResult:
    """`exponents[k]` is candidate k's zeta_eff; `pivots` the candidates the
    Cholesky accepted, in pivot order -- the auxiliary shells
    (``to_aux_shells``). `vectors[J, k]` is Cholesky vector J's value at
    candidate k, (phi_k|psi_J), a diagnostic (``reconstruction_error``)."""

    ell: int
    threshold: float
    exponents: np.ndarray = field(default_factory=lambda: np.zeros(0))
    pivots: list[int] = field(default_factory=list)
    vectors: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))


def compact_cholesky_for_l(
    pairs: list[eri_quartet.PrimitivePair], target_l: int, threshold: float
) -> AccdResult:
    """Run acCD's pivoted Cholesky at `target_l`: build every candidate of
    the pool `pairs`, form
    their unit-diagonal Coulomb metric, and decompose it with
    ``cholesky.pivoted_cholesky``, stopping once every residual is below
    `threshold`.

    Unit-diagonal -- candidates normalized to (A|A) = 1, as Lehtola's are --
    because the self-Coulomb of an overlap-normalized primitive still scales
    as 1/zeta (its charge as zeta^-3/4, its extent as zeta^-1/2), spanning
    ~5 orders of magnitude across one cc-pVDZ pool, so an
    absolute cutoff on the raw metric would act partly as an exponent
    ceiling rather than an accuracy. On this metric `threshold` is a
    relative residual: each candidate's unexplained fraction of its own
    self-Coulomb."""
    exponents = build_candidates(pairs, target_l)
    if exponents.size == 0:
        return AccdResult(target_l, threshold, exponents)
    pivots, vectors = cholesky.pivoted_cholesky(
        coulomb_metric(target_l, exponents), threshold
    )
    return AccdResult(target_l, threshold, exponents, pivots, vectors)


def reconstruction_error(result: AccdResult) -> float:
    """max |V^T V - M| against the metric M rebuilt independently of the
    Cholesky loop; M's unit diagonal makes this relative, like the
    threshold, so Cauchy-Schwarz bounds it by `result.threshold`."""
    if result.exponents.size == 0:
        return 0.0
    return cholesky.reconstruction_error(
        coulomb_metric(result.ell, result.exponents), result.vectors
    )


def to_aux_shells(result: AccdResult) -> list[AuxShell]:
    """One uncontracted AuxShell per pivot, tightest first (the usual
    basis-set order)."""
    return [
        AuxShell(result.ell, (float(e),), ((1.0,),))
        for e in sorted(result.exponents[result.pivots], reverse=True)
    ]


def to_pyscf_basis(aux_shells: dict[int, list[AuxShell]]) -> list:
    """`aux_shells` in PySCF's own internal basis format, ``[[ell, [exponent,
    c_0, c_1, ...], ...], ...]`` (one coefficient column per contracted
    function) -- usable directly as a Mole's basis or a density-fitting
    ``auxbasis`` (``df_check``)."""
    return [
        [
            s.ell,
            *(
                [e, *(row[k] for row in s.coefficients)]
                for k, e in enumerate(s.exponents)
            ),
        ]
        for shells in aux_shells.values()
        for s in shells
    ]
