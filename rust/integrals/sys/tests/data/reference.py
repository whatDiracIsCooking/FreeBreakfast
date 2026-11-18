"""libcint's Cartesian three-center integrals (mn|P), PySCF's `int3c2e_cart`,
packed as `fb_three_center_packed_class` writes them, for fb-integrals-sys's
`tests/packed_witness.rs`.

Usage (from the checkout root; PySCF is in the `tools` dependency group):

    D=rust/integrals/sys/tests/data
    OMP_NUM_THREADS=12 uv run --group tools python $D/reference.py \\
        $D/packed_witness.npy

The basis is the test's own, written out twice, here and in the test
(`shells` and `aux_shells` there): an orbital shell of every momentum
through f on each of two centres, two primitives on A's and one on B's, and
an auxiliary shell of every momentum through i, the even ones on A and the
odd ones on a third centre C, an odd one two primitives. So every one of
the kernels' 70 classes (la <= lb <= 3, lP <= 6) meets a one-centre pair
and a two-centre one, a contracted shell, an aux shell on an orbital
centre and one off them. Coordinates are bohr on both sides.

The normalisation is the test's, reached through the overlap, as
rust/integrals/schwarz/tests/data/reference.py does: a test function is
its shell's coefficients exactly as given, on the raw Cartesian Gaussian
x^i y^j z^k exp(-a r^2) of every component, with no normaliser and no
contraction renormalised. PySCF's differs from it by one positive factor
per function, lambda_m = sqrt(S_test[m,m] / S_pyscf[m,m]), S_test the
Cartesian overlap diagonal in closed form and S_pyscf `int1e_ovlp_cart`'s,
on both bases, so (mn|P)_test = lambda_m lambda_n lambda_P (mn|P)_pyscf.
The script fails unless lambda is one value across each shell's
components.

The array is a float64 .npy of shape (n_aux, n_col), element (P, c) the
packed column c's aux function P: in the Fortran order fb-io-npy reads, P
fastest, which is the packed output row's order. The columns are every
shell pair once, in the order the test stages them: the classes (la, lb),
la <= lb, ascending; in each, the pairs i <= j in shell order whose
momenta are the class's, the bra the lower momentum (i on a tie); each
pair's block (a, b), a the bra's Cartesian outer and b the ket's inner.
The aux functions are in the test's shell order: A's s, d, g, i, then C's
p, f, h. Cartesian components are x's power highest, then y's (both
codes' order). It prints the shape and max |(mn|P)|.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
from pyscf import df, gto

A = (0.0, 0.1, -0.2)
B = (0.4, -0.9, 1.3)
C = (-0.7, 0.5, 0.8)


def shells() -> list[tuple[int, tuple, list]]:
    """The orbital shells: A's s, p, d, f, then B's, each (l, centre,
    [(exponent, coefficient)])."""
    out = []
    for centre, prims in (
        (A, lambda e: [(2.5 * e, 0.7), (e, 0.4)]),
        (B, lambda e: [(1.3 * e, 0.9)]),
    ):
        for ell in range(4):
            out.append((ell, centre, prims(0.6 + 0.35 * ell)))
    return out


def aux_shells() -> list[tuple[int, tuple, list]]:
    """The aux shells: A's s, d, g, i, then C's p, f, h."""
    out = []
    for centre, ells in ((A, range(0, 7, 2)), (C, range(1, 7, 2))):
        for ell in ells:
            e = 0.9 + 0.2 * ell
            prims = [(e, 1.0)] if ell % 2 == 0 else [(3.0 * e, 0.6), (e, 0.5)]
            out.append((ell, centre, prims))
    return out


def odd_double_factorial(n: int) -> float:
    """n!! for odd n >= -1, (-1)!! = 1."""
    return float(math.prod(range(n, 0, -2))) if n > 0 else 1.0


def cartesian_powers(ell: int) -> list[tuple[int, int, int]]:
    """x's power highest, then y's: PySCF's Cartesian order and the
    kernels'."""
    return [
        (a, b, ell - a - b) for a in range(ell, -1, -1) for b in range(ell - a, -1, -1)
    ]


def overlap_diagonal(basis: list) -> np.ndarray:
    """S_test[m,m] in closed form: sum_ij c_i c_j prod_axis (2k-1)!! / (2p)^k
    (pi/p)^(3/2), p = a_i + a_j, c as given."""
    out = []
    for ell, _, prims in basis:
        exps = np.array([e for e, _ in prims])
        c = np.array([k for _, k in prims])
        p = exps[:, None] + exps[None, :]
        for powers in cartesian_powers(ell):
            angular = math.prod(odd_double_factorial(2 * k - 1) for k in powers)
            out.append(float(c @ (angular * (np.pi / p) ** 1.5 / (2 * p) ** ell) @ c))
    return np.array(out)


def molecule(basis: list) -> gto.Mole:
    """A molecule of ghost atoms, one per shell, each carrying that shell
    alone, in bohr, Cartesian: PySCF lists the functions atom by atom, so
    in the shells' order. PySCF reads a coefficient as a normalised
    primitive's, multiplying it by gto_norm(l, a), which differs from one
    exponent to the next; each is divided by that here, so PySCF's
    contracted function is the test's up to the one factor lambda takes
    out."""
    atoms = [[f"X{k}", centre] for k, (_, centre, _) in enumerate(basis)]
    sets = {
        f"X{k}": [[ell, *[[e, c / gto.gto_norm(ell, e)] for e, c in prims]]]
        for k, (ell, _, prims) in enumerate(basis)
    }
    return gto.M(atom=atoms, basis=sets, unit="Bohr", cart=True, verbose=0)


def lambdas(mol: gto.Mole, basis: list) -> np.ndarray:
    """lambda_m = sqrt(S_test[m,m] / S_pyscf[m,m]), checked one value per
    shell."""
    lam = np.sqrt(overlap_diagonal(basis) / np.diag(mol.intor("int1e_ovlp_cart")))
    loc = mol.ao_loc_nr()
    for b in range(mol.nbas):
        block = lam[loc[b] : loc[b + 1]]
        if np.abs(block / block[0] - 1).max() > 1e-12:
            raise SystemExit(f"shell {b}: lambda is not one value across its shell")
    return lam


def columns(basis: list) -> list[tuple[int, int]]:
    """Every packed column's (m, n) in the test's order."""
    n_cart = [(ell + 1) * (ell + 2) // 2 for ell, _, _ in basis]
    offset = np.concatenate([[0], np.cumsum(n_cart)]).astype(int)
    out = []
    for la in range(4):
        for lb in range(la, 4):
            for i in range(len(basis)):
                for j in range(i, len(basis)):
                    li, lj = basis[i][0], basis[j][0]
                    bra, ket = (i, j) if li <= lj else (j, i)
                    if (basis[bra][0], basis[ket][0]) != (la, lb):
                        continue
                    for a in range(n_cart[bra]):
                        for b in range(n_cart[ket]):
                            out.append((offset[bra] + a, offset[ket] + b))
    return out


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out = Path(sys.argv[1])
    orbital, aux = shells(), aux_shells()
    mol, auxmol = molecule(orbital), molecule(aux)
    lam, lam_aux = lambdas(mol, orbital), lambdas(auxmol, aux)
    eri = df.incore.aux_e2(mol, auxmol, intor="int3c2e_cart", aosym="s1")
    eri = eri * lam[:, None, None] * lam[None, :, None] * lam_aux[None, None, :]
    cols = columns(orbital)
    packed = np.stack([eri[m, n, :] for m, n in cols], axis=1)
    np.save(out, np.asfortranarray(packed))
    print(
        f"{out}: (mn|P) packed, {packed.shape[0]} aux x {packed.shape[1]} columns,"
        f" max |(mn|P)| {np.abs(packed).max():.6e}"
    )


if __name__ == "__main__":
    main()
