"""Small Cartesian/normalization helpers shared by more than one module here.

``cartesian_powers``/``ncart`` mirror ``src/basis``'s canonical Cartesian
monomial order (a descending first -- see ``cartesian_powers`` in
``src/basis/ao.cppm``), which is also PySCF's own ``cart=True`` in-shell
order, so nothing here needs to permute between the two.

``primitive_norm`` is PySCF's own per-angular-momentum (not per-Cartesian-
-component) primitive Gaussian normalization constant, applied uniformly
across every Cartesian component of a shell. This repo's own
``ShellStamp::primitive_normalized`` is also uniform per shell but
additionally divides by ``sqrt((2l-1)!!)``, the (l,0,0) normalization
(``eri_quartet``'s CONVENTION).

Angular momentum is spelled ``ell`` throughout this package, not ``l``: ruff's
E741 (and plain readability) flags a bare ``l`` as too easily misread as the
digit 1.
"""

from __future__ import annotations

import math


def cartesian_powers(ell: int) -> list[tuple[int, int, int]]:
    """(a, b, c) monomial powers of total degree ell, a descending first."""
    return [
        (a, b, ell - a - b) for a in range(ell, -1, -1) for b in range(ell - a, -1, -1)
    ]


def ncart(ell: int) -> int:
    """Number of Cartesian monomials of total degree ell: (ell+1)(ell+2)/2."""
    return (ell + 1) * (ell + 2) // 2


def double_factorial_odd(n: int) -> float:
    """n!! for odd n >= 1; n <= 0 is the empty product, 1.0."""
    result = 1.0
    while n > 1:
        result *= n
        n -= 2
    return result


def primitive_norm(alpha: float, ell: int) -> float:
    """PySCF's per-ell primitive Gaussian normalization N(alpha, ell) =
    (2*alpha/pi)^(3/4) * (4*alpha)^(ell/2)."""
    return (2.0 * alpha / math.pi) ** 0.75 * (4.0 * alpha) ** (ell / 2.0)
