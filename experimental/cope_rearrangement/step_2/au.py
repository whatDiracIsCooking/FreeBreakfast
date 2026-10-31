"""The a_u block of a C2h geometry: its symmetry operations, its a_u Cartesian
displacement basis, the internal coordinate s = (R16 - R34)/sqrt(2), and the
Hessian algebra (a_u frequencies, the relaxed force constant along s).

C2h characters (E, C2, i, sigma_h): A_g (1, 1, 1, 1), B_g (1, -1, 1, -1),
A_u (1, 1, -1, -1), B_u (1, -1, -1, 1). An a_u displacement keeps the C2 and
breaks i and sigma_h, so it takes C1-C6 and C3-C4 apart in opposite senses:
R16 - R34 is a_u.

For 1,5-hexadiene's C2h chair (16 atoms; C2, C5 and their hydrogens on the
mirror, the other twelve in three general sets of four) the Cartesian a_u
space is 11-dimensional: 3 per general set, 1 per on-mirror pair. One of the
11 is the translation along the C2 axis (T along the axis is A_u in C2h); the
rotations are A_g and B_g. So the a_u vibrational block is 10 x 10.

Everything here is in bohr and the geometry's own frame (PySCF's C2h frame,
whose symmetry elements lie along the axes).
"""

from __future__ import annotations

import itertools

import numpy as np

# Characters of A_u under each operation, by kind.
AU_CHARACTER = {"E": 1.0, "C2": 1.0, "i": -1.0, "sigma": -1.0}
AU_DIM = 11  # Cartesian a_u, the C2-axis translation included
VIB_DIM = 10

# R16 is atoms 0, 5; R34 atoms 2, 3.
R16, R34 = (0, 5), (2, 3)

# Harmonic wavenumber (cm^-1) of a unit mass-weighted eigenvalue,
# 1 Eh / (bohr^2 amu): sqrt(Eh / (a0^2 u)) / (2 pi c). CODATA 2018.
_EH = 4.3597447222071e-18  # J
_A0 = 5.29177210903e-11  # m
_AMU = 1.66053906660e-27  # kg
_C = 2.99792458e10  # cm/s
CM1_PER_AU = float(np.sqrt(_EH / (_A0**2 * _AMU)) / (2.0 * np.pi * _C))
# Eh/bohr^2 -> mdyn/A.
MDYN_PER_AU = _EH / _A0**2 * 1e-8


def _kind(r: np.ndarray) -> str:
    det, tr = np.linalg.det(r), np.trace(r)
    if np.isclose(tr, 3.0):
        return "E"
    if np.isclose(tr, -3.0):
        return "i"
    return "C2" if det > 0 else "sigma"


def operations(xyz: np.ndarray, tol: float = 1e-5):
    """The four C2h operations of `xyz` (natm x 3, bohr), as (kind, R, perm)
    with perm[a] the atom R takes atom a onto. The frame must hold the
    elements along its axes, as PySCF's C2h frame does: the operations are
    looked for among the eight diagonal sign matrices, and the set found is
    asserted to be C2h's."""
    found = []
    for signs in itertools.product((1.0, -1.0), repeat=3):
        r = np.diag(signs)
        moved = xyz @ r.T
        d = np.linalg.norm(moved[:, None, :] - xyz[None, :, :], axis=-1)
        perm = d.argmin(axis=1)
        if d[np.arange(len(xyz)), perm].max() < tol and len(set(perm)) == len(xyz):
            found.append((_kind(r), r, perm))
    kinds = sorted(k for k, _, _ in found)
    assert kinds == ["C2", "E", "i", "sigma"], f"not C2h in this frame: {kinds}"
    return found


def c2_axis(ops) -> np.ndarray:
    """Unit vector along the C2 axis (the +1 eigenvector of the C2 matrix)."""
    (r,) = [r for k, r, _ in ops if k == "C2"]
    w, v = np.linalg.eigh(r)
    return v[:, np.argmax(w)]


def rotations(xyz: np.ndarray) -> list[np.ndarray]:
    """The three infinitesimal rotations about the centroid, as 3N vectors."""
    c = xyz - xyz.mean(axis=0)
    return [np.cross(axis, c).ravel() for axis in np.eye(3)]


def _bond_bvector(xyz, a, b) -> np.ndarray:
    v = np.zeros(xyz.size)
    e = (xyz[a] - xyz[b]) / np.linalg.norm(xyz[a] - xyz[b])
    v[3 * a : 3 * a + 3] = e
    v[3 * b : 3 * b + 3] = -e
    return v


def s_bvector(xyz: np.ndarray) -> np.ndarray:
    """Wilson B row of s = (R16 - R34)/sqrt(2): ds/dx, 3N."""
    return (_bond_bvector(xyz, *R16) - _bond_bvector(xyz, *R34)) / np.sqrt(2.0)


def au_basis(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(Q, t). Q is 3N x 10, orthonormal, spanning the a_u Cartesian
    displacements orthogonal to t; t is the unit translation along the C2
    axis, the eleventh a_u vector. Column 0 of Q is s's own direction
    (s_bvector normalised; it is purely a_u, which is asserted), so H[0, 0]
    is the unrelaxed curvature along s. The other nine are fixed by an SVD
    and a sign convention, so a rerun reproduces them."""
    natm = len(xyz)
    ops = operations(xyz)
    proj = np.zeros((3 * natm, 3 * natm))
    for kind, r, perm in ops:
        d = np.zeros((3 * natm, 3 * natm))
        for a in range(natm):
            b = perm[a]
            d[3 * b : 3 * b + 3, 3 * a : 3 * a + 3] = r
        proj += AU_CHARACTER[kind] * d
    proj /= len(ops)
    w, v = np.linalg.eigh(proj)
    span = v[:, w > 0.5]
    assert span.shape[1] == AU_DIM, f"a_u dimension {span.shape[1]}, not {AU_DIM}"

    t = np.tile(c2_axis(ops), natm)
    t /= np.linalg.norm(t)
    assert np.linalg.norm(span.T @ t) > 1.0 - 1e-10, "C2-axis translation not a_u"
    for rot in rotations(xyz):
        assert np.linalg.norm(span.T @ rot) < 1e-8 * max(1.0, np.linalg.norm(rot))

    b = s_bvector(xyz)
    b /= np.linalg.norm(b)
    assert np.linalg.norm(span @ (span.T @ b) - b) < 1e-10, "s is not purely a_u"
    assert abs(b @ t) < 1e-10

    # s first, then the rest of the a_u space orthogonal to s and t.
    rest = span - np.outer(t, t @ span) - np.outer(b, b @ span)
    u, sv, _ = np.linalg.svd(rest, full_matrices=False)
    assert sv[VIB_DIM - 2] > 0.5 and sv[VIB_DIM - 1] < 1e-8, sv
    q = np.column_stack([b, u[:, : VIB_DIM - 1]])
    for k in range(1, VIB_DIM):
        if q[np.argmax(np.abs(q[:, k])), k] < 0:
            q[:, k] *= -1.0
    assert np.allclose(q.T @ q, np.eye(VIB_DIM), atol=1e-12)
    return q, t


def hessian_from_energies(e0, e_single, e_pair, h):
    """The a_u Hessian (10 x 10, Eh/bohr^2) in Q's basis from energies only.

    E is even in any a_u displacement (x and -x are images of each other
    under i, which fixes the C2h point), so
      E(h q_i) = E0 + h^2 H_ii / 2 + O(h^4), and
      E(h (q_i + q_j)) - E(h q_i) - E(h q_j) + E0 = h^2 H_ij + O(h^4).
    `e_single[i]` is E(h q_i); `e_pair[i][j]` (i < j) is E(h (q_i + q_j)).
    """
    n = len(e_single)
    hess = np.zeros((n, n))
    for i in range(n):
        hess[i, i] = 2.0 * (e_single[i] - e0) / h**2
    for i, j in itertools.combinations(range(n), 2):
        hess[i, j] = hess[j, i] = (e_pair[i][j] - e_single[i] - e_single[j] + e0) / h**2
    return hess


def analyse(hess, q, t, masses, xyz) -> dict:
    """The a_u frequencies and the force constants along s.

    The a_u normal modes solve H x = w^2 M x over the 11-dimensional a_u
    space. In the basis (Q, t), with H's t row and column exactly zero (E is
    translation invariant), that is a generalised eigenproblem whose one zero
    root is the translation; the other ten are the a_u vibrations. Their signs
    are those of the 10 x 10 H itself (Sylvester's law of inertia), so the
    count of imaginary a_u frequencies is H's count of negative eigenvalues.

    k_s is the force constant along s with every other a_u coordinate
    relaxed, 1 / (b H^-1 b) with b = ds/dx in Q's basis: the curvature of E
    along s on the path that minimises E at each s. Negative when the a_u
    block has one negative eigenvalue whose mode carries s; positive when the
    block is positive definite. k_s_unrelaxed is H[0, 0] / |b|^2, the
    curvature along s's own Cartesian direction with nothing relaxed.
    """
    basis = np.column_stack([q, t])
    m3 = np.repeat(masses, 3)
    m_q = basis.T @ (m3[:, None] * basis)
    h11 = np.zeros((VIB_DIM + 1, VIB_DIM + 1))
    h11[:VIB_DIM, :VIB_DIM] = hess
    wm, vm = np.linalg.eigh(m_q)
    m_isqrt = vm @ np.diag(wm**-0.5) @ vm.T
    lam, vec = np.linalg.eigh(m_isqrt @ h11 @ m_isqrt)
    modes = m_isqrt @ vec  # coefficients over (Q, t)
    # The translation is the root whose mode is most along t.
    t_weight = np.abs(modes[-1]) / np.linalg.norm(modes, axis=0)
    drop = int(np.argmax(t_weight))
    assert abs(lam[drop]) < 1e-10, lam[drop]
    keep = [k for k in range(VIB_DIM + 1) if k != drop]
    lam, modes = lam[keep], modes[:, keep]
    freq = np.sign(lam) * np.sqrt(np.abs(lam)) * CM1_PER_AU  # < 0: imaginary

    b_cart = s_bvector(xyz)
    b = q.T @ b_cart
    k_s = 1.0 / float(b @ np.linalg.solve(hess, b))

    # |cos| between the lowest mode's Cartesian displacement and s's direction.
    x_low = basis @ modes[:, 0]
    overlap_s = abs(x_low @ b_cart) / (np.linalg.norm(x_low) * np.linalg.norm(b_cart))

    h_eig = np.linalg.eigvalsh(hess)
    return {
        "hessian_eigenvalues": [float(x) for x in h_eig],
        "n_negative": int((h_eig < 0).sum()),
        "frequencies_cm1": [float(x) for x in freq],
        "lowest_mode_s_overlap": float(overlap_s),
        "k_s": k_s,
        "k_s_unrelaxed": float(hess[0, 0] / (b @ b)),
    }
