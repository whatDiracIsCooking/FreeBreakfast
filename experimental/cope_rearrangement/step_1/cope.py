"""Shared pieces of the step-1 scripts: the C2h chair geometry of the Cope
transition-structure region of 1,5-hexadiene, the symmetry-labelled CAS(6,6)
and the bookkeeping read off a converged CASSCF.

Numbering is the guide's: C1=C2-C3-C4-C5=C6, C3-C4 breaking, C1-C6 forming.
Atoms 0-5 are C1..C6 (so R16 is atoms 0,5 and R34 atoms 2,3; geomeTRIC's
1-based constraints say 1 6 and 3 4), then two hydrogens per terminal carbon
and one per central carbon.

`chair_c2h` documents its own frame; PySCF re-orients to its standard C2h
frame anyway. Inversion takes C1<->C4, C2<->C5, C3<->C6, and so
takes R16<->R34: R16 - R34 is a_u.
"""

from __future__ import annotations

import numpy as np

# The active space the guide prescribes: the allyl pi MOs of both fragments,
# combined in and out of phase. psi1 and psi3 give a_g + b_u each, psi2 gives
# b_g + a_u.
CAS_IRREP_NCAS = {"Ag": 2, "Bu": 2, "Au": 1, "Bg": 1}
# The dominant configuration at the TS: psi1(a_g)^2 psi1(b_u)^2 psi2(a_u)^2.
CAS_IRREP_NOCC = {"Ag": 1, "Bu": 1, "Au": 1, "Bg": 0}
NCAS, NELECAS = 6, 6


def _unit(v):
    return v / np.linalg.norm(v)


def chair_c2h(
    r_inter: float = 1.95,
    r_cc: float = 1.40,
    angle_ccc: float = 120.0,
    pucker: float = 0.30,
    r_ch: float = 1.08,
) -> list[tuple[str, np.ndarray]]:
    """Exactly C2h chair coordinates (Angstrom) with R16 = R34 = r_inter.

    Two allyls in parallel planes z = +-r_inter/2, terminal carbons stacked
    (C1 over C6, C3 over C4) and the central carbons on opposite sides (+y
    and -y): the chair. Each central carbon is lifted `pucker` away from the
    other fragment. Built from the unique atoms (C1, C2, H2, the two
    hydrogens on C1) and completed by the mirror x -> -x (which holds C2, C5
    and their hydrogens) and inversion through the origin, so the symmetry
    is exact to rounding. Their product, the C2 axis, is x: (x, y, z) ->
    (x, -y, -z), which takes C1 to C6 and C3 to C4. The guess only has to be
    a sensible chair: every geometry here is optimised from it.
    """
    a = r_cc * np.sin(np.radians(angle_ccc) / 2.0)
    c = np.sqrt(r_cc**2 - a**2 - pucker**2)
    z = r_inter / 2.0
    c1 = np.array([a, 0.0, z])
    c2 = np.array([0.0, c, z + pucker])
    c3 = np.array([-a, 0.0, z])

    # H2 bisects the outside of C1-C2-C3; the two on C1 sit at 120 degrees
    # from C1->C2 in the C1-C2-C3 plane.
    h2 = c2 + r_ch * _unit((c2 - c1) + (c2 - c3))
    h2[0] = 0.0
    n = _unit(np.cross(c1 - c2, c3 - c2))
    u = _unit(c2 - c1)
    w = _unit(np.cross(n, u))
    c120, s120 = np.cos(np.radians(120.0)), np.sin(np.radians(120.0))
    h1a = c1 + r_ch * (c120 * u + s120 * w)
    h1b = c1 + r_ch * (c120 * u - s120 * w)

    m = np.array([-1.0, 1.0, 1.0])
    # Inversion: C1 -> C4, C2 -> C5, C3 -> C6.
    carbons = [c1, c2, c3, -c1, -c2, -c3]
    hydrogens = [
        h1a, h1b,  # on C1
        h2,  # on C2
        h1a * m, h1b * m,  # on C3
        -h1a, -h1b,  # on C4
        -h2,  # on C5
        -h1a * m, -h1b * m,  # on C6
    ]  # fmt: skip
    return [("C", c) for c in carbons] + [("H", hh) for hh in hydrogens]


def atom_string(atoms) -> str:
    return "\n".join(f"{s} {x:.10f} {y:.10f} {z:.10f}" for s, (x, y, z) in atoms)


def mol_atoms(mol) -> list[tuple[str, np.ndarray]]:
    xyz = mol.atom_coords(unit="Angstrom")
    return [(mol.atom_pure_symbol(i), xyz[i]) for i in range(mol.natm)]


def interallyl(mol) -> tuple[float, float]:
    """(R16, R34) in Angstrom."""
    xyz = mol.atom_coords(unit="Angstrom")
    return (
        float(np.linalg.norm(xyz[0] - xyz[5])),
        float(np.linalg.norm(xyz[2] - xyz[3])),
    )


def allyl_normals(mol) -> np.ndarray:
    """Unit normal of each carbon's own allyl plane, one row per carbon."""
    xyz = mol.atom_coords()
    na = _unit(np.cross(xyz[0] - xyz[1], xyz[2] - xyz[1]))  # C1 C2 C3
    nb = _unit(np.cross(xyz[3] - xyz[4], xyz[5] - xyz[4]))  # C4 C5 C6
    return np.array([na, na, na, nb, nb, nb])


def pi_character(mol, mo_coeff) -> np.ndarray:
    """Per-MO weight on the six carbons' p functions along their allyl
    normal, in the Loewdin-orthogonalised AO basis: large for an allyl pi MO,
    small for sigma. Used once, to pick the active MOs out of the RHF
    canonicals."""
    from pyscf import lo

    c_orth = np.linalg.solve(lo.orth_ao(mol, "lowdin"), mo_coeff)
    normals = allyl_normals(mol)
    weight = np.zeros(mo_coeff.shape[1])
    labels = mol.ao_labels(fmt=False)
    for atom in range(6):
        shells: dict = {}
        for i, (ia, _sym, nl, m) in enumerate(labels):
            if ia == atom and nl.endswith("p"):
                shells.setdefault(nl, {})[m] = i
        for comps in shells.values():
            proj = sum(
                normals[atom][k] * c_orth[comps[ax]]
                for k, ax in enumerate(("x", "y", "z"))
            )
            weight += proj**2
    return weight


def orbital_irreps(mol, mo_coeff) -> list[str]:
    from pyscf import symm

    return [
        str(s) for s in symm.label_orb_symm(mol, mol.irrep_name, mol.symm_orb, mo_coeff)
    ]


def select_cas(mf) -> tuple[list[int], dict]:
    """0-based indices of the six RHF canonical MOs that make the active
    space: per irrep, the occupied (virtual) MOs with the most allyl-pi
    character, as many as CAS_IRREP_NOCC (NCAS - NOCC) asks for. Returns the
    indices and a per-MO report."""
    mol = mf.mol
    orbsym = orbital_irreps(mol, mf.mo_coeff)
    pi = pi_character(mol, mf.mo_coeff)
    occ = mf.mo_occ > 0
    chosen = []
    for irrep, ncas in CAS_IRREP_NCAS.items():
        nocc = CAS_IRREP_NOCC[irrep]
        for want_occ, count in ((True, nocc), (False, ncas - nocc)):
            pool = [
                i
                for i in range(len(orbsym))
                if orbsym[i] == irrep and occ[i] == want_occ
            ]
            pool.sort(key=lambda i: -pi[i])
            chosen += pool[:count]
    chosen.sort()
    report = {
        int(i): {
            "irrep": orbsym[i],
            "energy": float(mf.mo_energy[i]),
            "occ": float(mf.mo_occ[i]),
            "pi": float(pi[i]),
        }
        for i in chosen
    }
    return chosen, report


def active_natural(mc) -> list[dict]:
    """Natural orbitals of the active space, diagonalised irrep by irrep, as
    [{'irrep', 'occ'}], largest occupation first."""
    ncore, ncas = mc.ncore, mc.ncas
    orbsym = np.asarray(orbital_irreps(mc.mol, mc.mo_coeff[:, ncore : ncore + ncas]))
    dm1 = mc.fcisolver.make_rdm1(mc.ci, ncas, mc.nelecas)
    out = []
    for irrep in sorted(set(orbsym)):
        sel = np.where(orbsym == irrep)[0]
        w = np.linalg.eigvalsh(dm1[np.ix_(sel, sel)])
        out += [{"irrep": str(irrep), "occ": float(x)} for x in w]
    out.sort(key=lambda d: -d["occ"])
    return out


def active_irrep_counts(mc) -> dict:
    cas = mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]
    counts: dict = {}
    for s in orbital_irreps(mc.mol, cas):
        counts[s] = counts.get(s, 0) + 1
    return counts


def active_overlap(mol_a, cas_a, mol_b, cas_b) -> np.ndarray:
    """Singular values of the active-space overlap between two geometries
    (cross-geometry AO overlap): all ~1 when the active space carried over,
    one dropping toward 0 when an orbital swapped out."""
    from pyscf import gto

    s_ab = gto.intor_cross("int1e_ovlp", mol_a, mol_b)
    return np.linalg.svd(cas_a.T @ s_ab @ cas_b, compute_uv=False)
