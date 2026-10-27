"""aCD: which primitive pairs are worth a candidate. One atom's two-electron
integrals over its UNCONTRACTED orbital primitives, via PySCF, are
decomposed by pivoted Cholesky (``cholesky.pivoted_cholesky``, the atomic
form of Aquilante/Pedersen/Lindh's aCD), and each pivot (mu nu) puts the
product of the primitive shells mu and nu belong to into acCD's candidate
pool -- Lehtola, J. Chem. Theory Comput. 17, 6886 (2021), Sec. 3, the
"reduced" pool. A pair the decomposition never needed is spanned to the
threshold by the pairs it did.

OPT-IN HERE, not the default the paper makes it. Measured on C/H/O (CO and
CH4, cc-pVDZ..cc-pV5Z, DF-RHF against conventional): under the default SVD
contraction, screening never shrank the set -- contraction sets the size --
and cost accuracy wherever the difference stood above noise (cc-pVTZ CO
1e-6 Eh against 1e-7; at cc-pV5Z both sit near 1e-8, each ahead on one
molecule); uncontracted it cost up to 20x. What it does buy is ~28% fewer
primitives inside the contracted shells, which made PySCF's (mu nu|P) build
~18% faster for CO at cc-pVQZ and cc-pV5Z -- paid on every DF run, while
its own 7x generation cost at cc-pV5Z is paid once, offline. A speed knob,
then, not a size one.

The integrals are held whole, eight-fold-symmetry packed, rather than one
shell quartet at a time: numpy has no per-call launch cost, so there is no
reason for a block-at-a-time API.

CONVENTION. Spherical AOs, every primitive normalized (PySCF's own
convention): screening asks which pairs matter, and a unit-normalized
function is the scale on which that question means the same at every
exponent and l.

BASIS INPUT. `Basis` is either a name PySCF's own loader resolves (a built-in
string, or a path to an NWChem-format file -- whatever ``gto.basis.load``/
``mol.basis`` already accept) or a pre-parsed shell list in PySCF's OWN
internal format, ``[[ell, [exponent, coeff, ...], ...], ...]`` -- exactly what
``gto.basis.load`` itself returns, and what ``basis_from_bse`` below produces
from a Basis Set Exchange document. ``build_mol`` passes either straight
through to `mol.basis` (a bare list works unwrapped for the single-atom case
this tool always builds).
"""

from __future__ import annotations

import numpy as np
from pyscf import gto
from pyscf.data import elements

from . import cholesky

Atom = tuple[str, tuple[float, float, float]]
# a basis name/path PySCF's loader resolves, or a pre-parsed PySCF shell list
Basis = str | list
# one uncontracted orbital primitive, and an unordered pair of them
Primitive = tuple[int, float]
PrimitivePair = tuple[Primitive, Primitive]

_ORIGIN = (0.0, 0.0, 0.0)


def build_mol(atoms: list[Atom], basis: Basis, unit: str = "Bohr") -> gto.Mole:
    """A PySCF Mole for `atoms` (symbol, (x,y,z) in `unit`), spherical AOs --
    what every basis here means. `spin` is set to the neutral-atom parity so
    odd-electron elements build without needing an explicit charge/spin from
    the caller -- occupation is irrelevant to the AO integrals read off it."""
    mol = gto.Mole()
    mol.atom = list(atoms)
    mol.unit = unit
    mol.basis = basis
    mol.cart = False
    mol.spin = sum(elements.charge(sym) for sym, _xyz in atoms) % 2
    mol.verbose = 0
    mol.build()
    return mol


def primitives(element: str, basis: Basis) -> list[Primitive]:
    """`element`'s orbital basis fully uncontracted: its distinct (ell,
    exponent) primitives, sorted. The contraction coefficients play no role
    in which products acCD can need."""
    mol = build_mol([(element, _ORIGIN)], basis)
    return sorted(
        {
            (mol.bas_angular(ib), float(e))
            for ib in range(mol.nbas)
            for e in mol.bas_exp(ib)
        }
    )


def all_pairs(prims: list[Primitive]) -> list[PrimitivePair]:
    """Every unordered pair of `prims`, the unscreened pool."""
    return [(a, b) for i, a in enumerate(prims) for b in prims[i:]]


class PackedSymmetric:
    """A symmetric (n, n) matrix held as its packed lower triangle -- element
    (r, c), r >= c, at r(r+1)/2 + c, PySCF's ``aosym="s8"`` order -- read
    through just what ``cholesky`` needs: the diagonal, one full row
    (``m[p]``, the pivot gather) and blocks of lower-triangle rows. Half the
    memory of the dense matrix, and nothing here ever unpacks it whole."""

    def __init__(self, packed: np.ndarray, n: int) -> None:
        if packed.shape != (n * (n + 1) // 2,):
            raise ValueError(f"{packed.shape} is not a packed {n}x{n} triangle")
        self.packed = packed
        self.shape = (n, n)

    def diagonal(self) -> np.ndarray:
        r = np.arange(self.shape[0])
        return self.packed[r * (r + 1) // 2 + r]

    def __getitem__(self, p: int) -> np.ndarray:
        # (p, c<=p) is one contiguous segment; (q>p, p) one per later row
        start = p * (p + 1) // 2
        q = np.arange(p + 1, self.shape[0])
        return np.concatenate(
            [self.packed[start : start + p + 1], self.packed[q * (q + 1) // 2 + p]]
        )

    def lower_rows(self, r0: int, r1: int) -> np.ndarray:
        """Rows r0..r1-1, columns 0..r1-1, dense; entries above the diagonal
        are zero (their values are the transposed ones below it)."""
        block = np.zeros((r1 - r0, r1))
        for r in range(r0, r1):
            start = r * (r + 1) // 2
            block[r - r0, : r + 1] = self.packed[start : start + r + 1]
        return block


def screen_pairs(
    element: str, prims: list[Primitive], threshold: float
) -> tuple[list[PrimitivePair], dict]:
    """aCD over `element`'s primitives `prims`: pivoted Cholesky of their
    (mu nu|kappa lambda) at `threshold` (absolute, on normalized primitives),
    and the primitive pairs its pivots land on (module docstring). Returns
    (pairs, summary)."""
    mol = build_mol([(element, _ORIGIN)], [[ell, [e, 1.0]] for ell, e in prims])
    # read each shell back: PySCF orders one atom's shells its own way
    shell_prim = [
        (mol.bas_angular(ib), float(mol.bas_exp(ib)[0])) for ib in range(mol.nbas)
    ]
    ao_shell = np.repeat(np.arange(mol.nbas), np.diff(mol.ao_loc))
    mu, nu = np.tril_indices(mol.nao)
    eri = PackedSymmetric(mol.intor("int2e", aosym="s8"), mu.size)
    pivots, vectors = cholesky.pivoted_cholesky(eri, threshold)
    pairs = sorted(
        {
            tuple(sorted((shell_prim[ao_shell[mu[p]]], shell_prim[ao_shell[nu[p]]])))
            for p in pivots
        }
    )
    summary = {
        "nao": mol.nao,
        "vectors": len(pivots),
        "pairs": len(pairs),
        "of_pairs": len(prims) * (len(prims) + 1) // 2,
        "max_reconstruction_error": cholesky.reconstruction_error(eri, vectors),
    }
    return pairs, summary


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
