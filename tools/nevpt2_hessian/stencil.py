"""The host-side arithmetic of nevpt2_hessian.py: the stencils, the step
ladder and its judgement, the symmetry and bilinearity checks.

Standard library only, so test/pytest/test_nevpt2_hessian.py checks it, and
recomputes a fixture from its recorded gradients, on the push gate. Matrices
are lists of rows; H[i][j] = dg_i / dR_j, column j the derivative of the
gradient along coordinate (or direction) j.
"""

from __future__ import annotations

import math

# The fourth-order central first derivative, f'(0) ~ sum_k w_k f(o_k s) / s:
# (f(-2s) - 8 f(-s) + 8 f(s) - f(2s)) / 12s, error -s^4 f^(5)(0) / 30 + O(s^6).
OFFSETS = (-2.0, -1.0, 1.0, 2.0)
WEIGHTS = (1.0 / 12.0, -8.0 / 12.0, 8.0 / 12.0, -1.0 / 12.0)
# The ladder halves the step twice: h, h/2, h/4. A fourth-order error falls
# 2^4 = 16-fold per halving, so the ratio of successive differences
# (D(h) - D(h/2)) / (D(h/2) - D(h/4)) is 16 in the asymptotic range.
HALVINGS = 2
ORDER = 4
TARGET = 2.0**ORDER


def steps(h: float) -> list[float]:
    """The ladder's steps, coarsest first: h, h/2, h/4."""
    return [h / 2.0**k for k in range(HALVINGS + 1)]


def displacements(h: float) -> list[float]:
    """Every signed displacement the ladder at h reads, ascending. The
    finer steps share points with the coarser (h/2's 2s is h's s), so the
    three stencils read 8 points, not 12."""
    out = {o * s for s in steps(h) for o in OFFSETS}
    return sorted(out)


def derivative(values: dict[float, float], s: float) -> float:
    """The fourth-order central derivative at step s from a map of
    displacement to value (the map may hold other displacements)."""
    return sum(w * values[o * s] for o, w in zip(OFFSETS, WEIGHTS, strict=True)) / s


def noise_gain(s: float) -> float:
    """sqrt(sum w_k^2) / s: the factor by which one stencil at step s
    multiplies independent noise of unit size in each value it reads."""
    return math.sqrt(sum(w * w for w in WEIGHTS)) / s


def column_ladder(points: dict[float, list[float]], h: float) -> list[list[float]]:
    """The stencil at each of the ladder's steps over vectors (gradients)
    keyed by displacement: one derivative vector per step, coarsest first."""
    n = len(next(iter(points.values())))
    return [
        [derivative({d: v[i] for d, v in points.items()}, s) for i in range(n)]
        for s in steps(h)
    ]


def transpose(m: list[list[float]]) -> list[list[float]]:
    return [list(row) for row in zip(*m, strict=True)]


def judge(ladder: list[float], floor: float, window: tuple[float, float]) -> dict:
    """One element's ladder D(h), D(h/2), D(h/4): its two differences, their
    ratio, and the verdict. Below `floor` the finer difference is noise, not
    truncation, so the ratio says nothing and is not judged (`ratio` None);
    the bar is then the floor, and the coarser difference must still be one
    a settled ladder could have above it (|upper| under window[1] times the
    floor), or `settled` is False: a ladder whose coarse rung moved far more
    than its fine one is not converging, it has a kink the fine rung steps
    over. Above the floor the ratio must lie inside `window` (around TARGET)
    and the bar is the finer difference."""
    coarse, middle, fine = ladder
    upper, lower = coarse - middle, middle - fine
    if abs(lower) <= floor:
        return {
            "upper": upper,
            "lower": lower,
            "ratio": None,
            "settled": None if abs(upper) < window[1] * floor else False,
            "bar": floor,
        }
    ratio = upper / lower
    return {
        "upper": upper,
        "lower": lower,
        "ratio": ratio,
        "settled": window[0] < ratio < window[1],
        "bar": abs(lower),
    }


def asymmetry(m: list[list[float]]) -> list[list[float]]:
    """|M - M^T|, element by element."""
    n = len(m)
    return [[abs(m[i][j] - m[j][i]) for j in range(n)] for i in range(n)]


def symmetrized(m: list[list[float]]) -> list[list[float]]:
    n = len(m)
    return [[0.5 * (m[i][j] + m[j][i]) for j in range(n)] for i in range(n)]


def symmetric_bar(bar: list[list[float]]) -> list[list[float]]:
    """The bar of the symmetrized Hessian: the larger of the two elements'."""
    n = len(bar)
    return [[max(bar[i][j], bar[j][i]) for j in range(n)] for i in range(n)]


def symmetry_misses(
    m: list[list[float]], bar: list[list[float]]
) -> list[tuple[int, int, float, float]]:
    """The pairs i < j whose asymmetry |M_ij - M_ji| exceeds the larger of
    their bars: (i, j, asymmetry, bar)."""
    n = len(m)
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            miss = abs(m[i][j] - m[j][i])
            allowed = max(bar[i][j], bar[j][i])
            if miss > allowed:
                out.append((i, j, miss, allowed))
    return out


def bilinearity(
    m: list[list[float]],
    bar: list[list[float]],
    i: int,
    j: int,
    qq: float,
    qq_bar: float,
) -> dict:
    """Eq. (bilinear) on the pair (e_i, e_j): the directional second
    derivative qq along the unit q = (e_i + e_j) / sqrt 2 against
    (M_ii + M_jj + M_ij + M_ji) / 2, the columns' own (unsymmetrized)
    elements. A failure is a residual above qq's bar plus half the four
    elements' bars."""
    predicted = 0.5 * (m[i][i] + m[j][j] + m[i][j] + m[j][i])
    allowed = qq_bar + 0.5 * (bar[i][i] + bar[j][j] + bar[i][j] + bar[j][i])
    residual = qq - predicted
    return {
        "pair": [i, j],
        "directional": qq,
        "predicted": predicted,
        "residual": residual,
        "allowed": allowed,
        "fails": abs(residual) > allowed,
    }
