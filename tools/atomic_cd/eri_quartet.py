"""The raw AO-basis two-electron tensor for one atom's basis, via PySCF.
Returns the WHOLE dense tensor rather than one shell quartet at a time: numpy
has no per-call launch cost, so there is no reason for a block-at-a-time API.

CONVENTION. Coefficients as given, over a primitive-normalized basis -- this
repo's own convention (``BasisStamp::primitive_normalized`` in
``src/basis``): every Cartesian component of a shell scaled by the one (l,0,0)
primitive normalization ``primitive_norm(alpha, l) / sqrt((2l-1)!!)``, no
contraction renormalization. PySCF normalizes differently
(``angular.primitive_norm``, with no ``(2l-1)!!``, then a contraction
renormalization), so each AO is rescaled by
``k_mu = sqrt(S_ours[mu,mu] / S_pyscf[mu,mu])``, the ratio of the two
conventions' self-overlaps; the ERI picks up one factor per index.

BASIS INPUT. `Basis` is either a name PySCF's own loader resolves (a built-in
string, or a path to an NWChem-format file -- whatever ``gto.basis.load``/
``mol.basis`` already accept) or a pre-parsed shell list in PySCF's OWN
internal format, ``[[ell, [exponent, coeff, ...], ...], ...]`` -- exactly what
``gto.basis.load`` itself returns, and what ``basis_from_bse`` below produces
from a Basis Set Exchange document. Both ``build_mol`` and
``_our_overlap_diagonal`` accept either directly: a bare list is passed
straight through to `mol.basis` (it works unwrapped for the single-atom case
this tool always builds) and straight past ``gto.basis.load`` since it
already IS that call's return shape.
"""

from __future__ import annotations

import numpy as np
from pyscf import gto
from pyscf.data import elements

from .angular import cartesian_powers, double_factorial_odd, primitive_norm

Atom = tuple[str, tuple[float, float, float]]
# a basis name/path PySCF's loader resolves, or a pre-parsed PySCF shell list
Basis = str | list


def _cartesian_self_overlap(
    exponents: list[float], coefficients: list[float], powers: tuple[int, int, int]
) -> float:
    """S[mu,mu] for one Cartesian AO in this repo's own convention (every
    component scaled by the shell's (l,0,0) primitive normalization,
    contraction coefficients as given, no contraction renormalization) -- a
    self-overlap only, not a full engine. So only the (l,0,0)-type components
    come out unit-normalized; x*y of a d shell has self-overlap 1/3."""
    a, b, c = powers
    ell = a + b + c
    dfac = (
        double_factorial_odd(2 * a - 1)
        * double_factorial_odd(2 * b - 1)
        * double_factorial_odd(2 * c - 1)
    )
    shell_scale = 1.0 / np.sqrt(double_factorial_odd(2 * ell - 1))
    total = 0.0
    for ai, ci in zip(exponents, coefficients, strict=True):
        for aj, cj in zip(exponents, coefficients, strict=True):
            p = ai + aj
            radial = (np.pi / p) ** 1.5
            angular = dfac / (2.0 * p) ** ell
            total += (
                ci
                * cj
                * primitive_norm(ai, ell)
                * primitive_norm(aj, ell)
                * radial
                * angular
            )
    return shell_scale**2 * total


def _our_overlap_diagonal(atoms: list[Atom], basis: Basis) -> np.ndarray:
    """S_ours[mu,mu] for every AO, in our convention and PySCF's AO order."""
    diag: list[float] = []
    for sym, _xyz in atoms:
        shells = basis if isinstance(basis, list) else gto.basis.load(basis, sym)
        for shell in shells:
            ell = shell[0]
            prims = shell[1:]
            exponents = [p[0] for p in prims]
            ncol = len(prims[0]) - 1
            for col in range(ncol):
                coefficients = [p[1 + col] for p in prims]
                for powers in cartesian_powers(ell):
                    diag.append(
                        _cartesian_self_overlap(exponents, coefficients, powers)
                    )
    return np.asarray(diag)


def build_mol(atoms: list[Atom], basis: Basis) -> gto.Mole:
    """A PySCF Mole for `atoms` (symbol, (x,y,z) in bohr), Cartesian AOs
    (cart=True: the layout src/basis's kernels evaluate, before its
    spherical_transform_matrix maps to the spherical AOs the basis means; aCD
    here stays over the Cartesian ones). `spin` is set
    to the neutral-atom parity so odd-electron elements build without needing
    an explicit charge/spin from the caller -- occupation is irrelevant to the
    AO integrals this module reads off `mol`."""
    mol = gto.Mole()
    mol.atom = list(atoms)
    mol.unit = "Bohr"
    mol.basis = basis
    mol.cart = True
    mol.spin = sum(elements.charge(sym) for sym, _xyz in atoms) % 2
    mol.verbose = 0
    mol.build()
    return mol


def eri_ao(atoms: list[Atom], basis: Basis) -> tuple[np.ndarray, gto.Mole]:
    """The raw (ab|cd) tensor for `atoms` in `basis`, in this repo's AO
    convention (see module docstring). Returns (tensor, mol): shape
    (nao,nao,nao,nao); `mol` is handed back since callers typically also need
    its shell/AO structure (e.g. cli.py reads shell exponents off it for
    compact.py)."""
    mol = build_mol(atoms, basis)
    our_diag = _our_overlap_diagonal(atoms, basis)
    k = np.sqrt(our_diag / np.diag(mol.intor("int1e_ovlp")))
    eri = mol.intor("int2e")
    k4 = (
        k[:, None, None, None]
        * k[None, :, None, None]
        * k[None, None, :, None]
        * k[None, None, None, :]
    )
    return eri * k4, mol


def basis_from_bse(document: dict, element: str) -> list:
    """One element's electron shells out of a Basis Set Exchange document --
    the same shape ``cli.to_bse_spherical`` writes, or
    ``basis_set_exchange.get_basis()`` returns -- converted to PySCF's own
    internal basis format (see module docstring's BASIS INPUT section),
    directly usable as `basis` above.

    A shell's ``angular_momentum`` has more than one entry only for a
    combined shell (Pople-style "SP"/"SPD"): one coefficient column per
    momentum, all sharing that shell's one exponent set -- split into one
    PySCF shell per momentum. A single-momentum shell's coefficient columns
    are instead multiple contracted functions of that SAME momentum sharing
    one exponent set (e.g. cc-pVDZ's general contraction); PySCF's own
    format already stacks those as extra columns of one shell, so they carry
    over as given.
    """
    z = str(gto.charge(element))
    try:
        electron_shells = document["elements"][z]["electron_shells"]
    except KeyError as exc:
        raise ValueError(
            f"{element!r} (Z={z}) not in this BSE document's elements: "
            f"{sorted(document.get('elements', {}))}"
        ) from exc

    shells: list = []
    for shell in electron_shells:
        exponents = [float(e) for e in shell["exponents"]]
        momenta = shell["angular_momentum"]
        columns = [[float(c) for c in col] for col in shell["coefficients"]]
        if len(momenta) > 1:
            for ell, col in zip(momenta, columns, strict=True):
                rows = [[e, c] for e, c in zip(exponents, col, strict=True)]
                shells.append([ell, *rows])
        else:
            rows = [[e, *(col[i] for col in columns)] for i, e in enumerate(exponents)]
            shells.append([momenta[0], *rows])
    return shells
