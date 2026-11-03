#!/usr/bin/env python3
"""Write the brute-force QD-NEVPT2 derivative-coupling references (#744).

QD-NEVPT2's states are Psi_P = sum_M (|M> + |Psi_M^(1)>) R_MP, and their
derivative coupling (Park 2020, Eqs. 45-50) has a part no zeroth-order
overlap can see, d^PT2 = sum_MN T^PQ_MN <Psi_M^(1)|grad N> (Eq. 49), with
T^PQ = 1/2 (R_P R_Q^T - R_Q R_P^T). This tool builds Psi_M^(1) explicitly in
the full determinant space of qd_nevpt2_reference.py's three brute-force
cases (every orbital), and from it writes the only references that see
d^PT2:

1. **Psi_M^(1), SC and PC**, for every root M, class by class from the
   amplitudes qd_nevpt2_reference.py's brute-force H_eff validation uses:
   SC, per class and external occupation pattern t, -P_t V|M> / (Delta_t +
   H^M_t / N^M_t), over the patterns whose norm passes `_norm_to_energy`'s
   guard; PC, per class and tuple, -Q (Q^T (H_D - E_M^(0)) Q)^-1 Q^T V|M>
   over the span Q of the class's E_pq E_rs on |M> (`pc_brute_couplings`).
   Self-checks: <N|H|Psi_M^(1)>_k equals C_NM's class k (the `couplings` /
   `pc_couplings` goldens) to BRUTE_FORCE, and <N|Psi_M^(1)> = 0.
2. **The PT2 transition density** D^(MN)_xy = <Psi_M^(1)|E_xy|N> for every
   ordered pair, over all n_mo orbitals, on the qd_nevpt2/ fixture's own
   state (its CI vectors and orbitals, checked equal to the committed ones),
   asserted zero outside the (t,i), (a,t) and (a,i) blocks: E_xy|N> lies
   outside the first-order interacting space everywhere else.
3. **The whole coupling by finite differences** (Eq. 45). With
   Psi~_P = sum_M (|M> + |Psi_M^(1)>) R_MP and Psi_P^(0) = sum_M |M> R_MP,
   and <Psi_P^(0)(R)|Psi_Q^(1)(R)> = 0 at every geometry,

       d_PQ = d/dR <Psi_P^(0)(R0)|Psi_Q^(0)(R)>
            + 1/2 d/dR [<Psi_P^(1)(R0)|Psi_Q^(0)(R)> - <Psi_Q^(1)(R0)|Psi_P^(0)(R)>]

   Psi^(1) is built only at R0. Each cross-geometry overlap is a
   full-space determinant overlap (`fci.addons.overlap` over the MO cross
   overlap from `gto.intor_cross`). The first term is d^NEV + d^CAS (Eqs. 47
   + 48, #743's), the second d^PT2 alone; both are written, and their sum.

**The states.** Part 2 reads the fixture's state exactly as
qd_nevpt2_reference.py builds it (mc1step, converged, not polished). The
differences need a stationary reference, since an overlap feels a residual
orbital gradient to first order: from the fixture's orbitals the DF
SA-CASSCF is polished by Newton steps over the exact orbital Hessian (the CI
re-solved densely) until |g| stops falling (the best point kept, at most
POLISH_BAR), and Psi^(1), H_eff and R are rebuilt there (and step 1's
checks rerun). Each
displaced point carries those orbitals, Loewdin-orthonormalized in the
displaced overlap, re-converges the DF SA-CASSCF at the same weights and
cc-pVDZ-JKFIT the same way, and builds SC- and PC-QD-NEVPT2's H_eff over its
SA-canonical orbitals (qd_nevpt2_reference.py's `sc_qd`, `pc_qd`).

**Matching and phases.** Each displaced root is matched to the reference's
by the largest |<M(R0)|N(R)>| (a permutation, each within GUARD of 1) and
signed so that overlap is positive. H_eff(R) is carried into that matched,
signed basis; its eigenvectors are matched to R(R0) the same way, by
R(R0)^T R(R), and signed so the matched overlap is positive. At R0 the CI
vectors are signed by casscf_nac_reference.py's rule and each column of R
has its largest component positive (`pipeline/nevpt2`'s); the signs of
d_PQ follow those of Psi_P and Psi_Q, so a consumer with other phases
multiplies d_PQ by s_P s_Q.

**The differences.** Along the three directions `qd_nevpt2_gradient/`'s
generator draws per case (seed 696, translation and rotation projected out)
and, for `h8_ring_sto3g_sa3`, every Cartesian component, at that
directory's steps per case (the ring's at 0.6 of its): D(h) = (F(+h) -
F(-h)) / 2h at three halving steps, Richardson over the two finest, and the
error bar |R(fine) - R(coarse)| + noise / 2h_finest (#696's convention),
`noise` per scheme, term and pair a bound on any one point's overlap error:
NOISE_MARGIN times the measured one (`measure_noise`), since a point error
e moves the Richardson value by up to 1.5 e / h_finest.

Everything is deterministic: BLAS and OpenMP run on one thread, so a rerun
writes the same bytes.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        tools/emit/qd_nevpt2_nac_reference.py [case ...]

writes test/gtest/data/qd_nevpt2_nac/brute_force/ (every case, or those
named). Never hand-edited; qd_nevpt2_nac/README.md has the layouts.
"""

from __future__ import annotations

import os

# Before numpy loads its BLAS: one thread, so every reduction runs in one
# order and the output is reproducible bit for bit.
for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import itertools  # noqa: E402
import json  # noqa: E402
import pathlib  # noqa: E402
import shutil  # noqa: E402
import sys  # noqa: E402

import numpy  # noqa: E402
import pyscf  # noqa: E402
from pyscf import fci, gto, lib, scf  # noqa: E402
from pyscf.mcscf import mc1step  # noqa: E402
from pyscf.mrpt import dfnevpt2, nevpt2  # noqa: E402

# isort: split
# The sibling tools pin their own thread counts on import; numpy is loaded
# by then, and main() pins PySCF's OpenMP.
from casscf_reference import (  # noqa: E402
    POLISH_CURVATURE,
    casscf,
    derivatives,
    optimize,
)
from qd_nevpt2_reference import (  # noqa: E402
    BRUTE_FORCE,
    CASES,
    CLASSES,
    NCLASS,
    State,
    build_state,
    excitation_operators,
    full_space,
    integral_blocks,
    pc_brute_couplings,
    pc_qd,
    sa_canonical,
    sc_qd,
    tiny_molecule,
    transition_rdms,
)
from qd_nevpt2_reference import OUT as QD_OUT  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "test" / "gtest" / "data" / "qd_nevpt2_nac" / "brute_force"
GRADIENT = ROOT / "test" / "gtest" / "data" / "qd_nevpt2_gradient"
COMMAND = (
    "UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools "
    "tools/emit/qd_nevpt2_nac_reference.py"
)

# Per case: its steps (bohr, halving; qd_nevpt2_gradient/'s, where the
# SA-canonical virtual pair's gap sets them, but the ring's at 0.6 of its,
# for its SC pair (1, 2)'s Richardson error) and whether every Cartesian
# component is differenced too.
FD = {
    "h6_631g_sa2": {"steps": [1e-3, 5e-4, 2.5e-4], "full": False},
    "h8_sto3g_sa3": {"steps": [5e-4, 2.5e-4, 1.25e-4], "full": False},
    "h8_ring_sto3g_sa3": {"steps": [6e-5, 3e-5, 1.5e-5], "full": True},
}
# Each point's overlap noise: NOISE_MARGIN times the measured, at least
# NOISE_FLOOR.
NOISE_MARGIN = 3.0
NOISE_FLOOR = 1e-14
# The polish: Newton steps, at most POLISH_STEPS, until |g| is under
# POLISH_DONE or has not fallen for POLISH_STALL steps; the best point is
# kept, and must be under POLISH_BAR.
POLISH_STEPS = 20
POLISH_STALL = 3
POLISH_DONE = 1e-14
POLISH_BAR = 1e-11
# The continuity guard: every matched |<M(R0)|N(R)>| and |R(R0)^T R(R)|
# within GUARD of 1.
GUARD = 1e-3
# The finite differences: every error bar under ERROR_BAR (1/bohr), a fifth
# of the design's ~1e-6 gate. Not #696's 1e-7: the H8 ring's SC pair (1, 2),
# whose H_eff states are 4.7 mEh apart, couples at up to |d| ~ 8 and turns
# the orbitals' residual into overlap noise near 1e-12, so no step brings its
# Richardson error and its noise term both under 1e-7 at once (README).
ERROR_BAR = 2e-7
# The ladder ratio window, checked where the rung difference clears both
# LADDER_FLOOR and 5 noise / h.
LADDER_WINDOW = (3.2, 4.8)
LADDER_FLOOR = 1e-7
# Step 2: the density outside its three blocks, relative to its largest.
OFF_BLOCK = 1e-14
# The bar a consumer holds D^(MN) to (#746: path agreement, relative to the
# whole density's largest element), for the visibility table.
DENSITY_BAR = 1e-10
# Rigid motions in the directions: |u . rigid| under this.
RIGID = 1e-10


SCHEMES = ("sc", "pc")
TERMS = ("zeroth", "pt2", "total")


# ------------------------------------------------------------- Psi^(1)

KINDS = {
    "Sr": (0, 1),
    "Si": (1, 0),
    "Sijrs": (2, 2),
    "Sijr": (2, 1),
    "Srsi": (1, 2),
    "Srs": (0, 2),
    "Sij": (2, 0),
    "Sir": (1, 1),
}
assert list(KINDS) == list(CLASSES)


def sc_psi1(state: State, space: dict) -> numpy.ndarray:
    """SC's first-order functions, (n, 8, na, nb): per class and external
    occupation pattern t, -P_t V|J> / (<P_t V J|H_D|P_t V J> / N_t - E_J^(0)),
    V|J> on t being H|J> on t, over the patterns with N_t past the guard
    (`brute_force`'s, term for term)."""
    key, holes, parts = space["key"], space["holes"], space["parts"]
    hv, hdv, e0 = space["hv"], space["hdv"], space["e0"]
    n = len(state.ci)
    psi = numpy.zeros((n, NCLASS, space["na"], space["nb"]))
    for k, (nh, npart) in enumerate(KINDS.values()):
        mask = (holes == nh) & (parts == npart)
        _, inverse = numpy.unique(key[mask], return_inverse=True)
        for j in range(n):
            norm = numpy.bincount(inverse, weights=hv[j][mask] ** 2)
            dyall = numpy.bincount(inverse, weights=hv[j][mask] * hdv[j][mask])
            idx = numpy.abs(norm) > nevpt2.NUMERICAL_ZERO
            amplitude = numpy.zeros_like(norm)
            amplitude[idx] = -1.0 / (dyall[idx] / norm[idx] - e0[j])
            psi[j, k][mask] = amplitude[inverse] * hv[j][mask]
    return psi


def pc_psi1(state: State, space: dict, pc: dict) -> numpy.ndarray:
    """PC's first-order functions, (n, 8, na, nb), from the same span and
    solve `pc_brute_couplings` checks PC H_eff with."""
    n = len(state.ci)
    psi = numpy.zeros((n, NCLASS, space["na"], space["nb"]))
    pc_brute_couplings(state, space, pc["pc_dropped"], True, psi1=psi)
    return psi


def psi1_checks(name: str, space: dict, psi: numpy.ndarray, couplings) -> dict:
    """<N|H|Psi_M^(1)>_k against C_NM's class k, and <N|Psi_M^(1)> = 0."""
    hv, roots = space["hv"], space["roots"]
    n = psi.shape[0]
    coupling = numpy.einsum("nab,mkab->nmk", numpy.asarray(hv), psi)
    overlap = numpy.einsum("nab,mab->nm", numpy.asarray(roots), psi.sum(axis=1))
    miss = float(numpy.abs(coupling - couplings).max())
    assert miss < BRUTE_FORCE, f"{name}: <N|H|Psi^(1)> off C_NM by {miss}"
    total = float(numpy.abs(coupling.sum(axis=2) - couplings.sum(axis=2)).max())
    assert total < BRUTE_FORCE, f"{name}: <N|H|Psi^(1)> off H_eff by {total}"
    worst = float(numpy.abs(overlap).max())
    assert worst < BRUTE_FORCE, f"{name}: <N|Psi^(1)> = {worst}"
    norms = numpy.sqrt(numpy.einsum("mab,mab->m", psi.sum(axis=1), psi.sum(axis=1)))
    return {
        "coupling_vs_golden": miss,
        "heff_vs_golden": total,
        "overlap_with_references": worst,
        "psi1_norms": [float(x) for x in norms[:n]],
    }


def blocks_mask(ncore: int, ncas: int, nmo: int) -> dict:
    """The density's three non-zero blocks, (t,i), (a,t), (a,i), as masks."""
    c, t, a = slice(0, ncore), slice(ncore, ncore + ncas), slice(ncore + ncas, nmo)
    out = {}
    for label, (x, y) in {"ti": (t, c), "at": (a, t), "ai": (a, c)}.items():
        mask = numpy.zeros((nmo, nmo), dtype=bool)
        mask[x, y] = True
        out[label] = mask
    return out


def pt2_density(state: State, space: dict, psi: numpy.ndarray) -> tuple:
    """D^(MN)_xy = <Psi_M^(1)|E_xy|N>, (n, n, nmo, nmo), E_xy spin-summed,
    and its largest element outside the three blocks over its largest."""
    nmo, nelec = space["nmo"], space["nelec"]
    ea, eb = excitation_operators(nmo, nelec)
    total = psi.sum(axis=1)
    n = total.shape[0]
    density = numpy.zeros((n, n, nmo, nmo))
    for j, ket in enumerate(space["roots"]):
        excited = numpy.einsum("xyac,cb->xyab", ea, ket) + numpy.einsum(
            "xybc,ac->xyab", eb, ket
        )
        density[:, j] = numpy.einsum("mab,xyab->mxy", total, excited)
    masks = blocks_mask(state.casci.ncore, state.casci.ncas, nmo)
    inside = masks["ti"] | masks["at"] | masks["ai"]
    off = float(numpy.abs(density[:, :, ~inside]).max() / numpy.abs(density).max())
    assert off < OFF_BLOCK, f"{state.name}: D^(MN) off its blocks by {off}"
    return density, off


# ------------------------------------------------------------- the states


def moved_case(case: dict, coords: numpy.ndarray) -> dict:
    return dict(
        case,
        atoms=[
            (symbol, tuple(float(x) for x in r))
            for (symbol, _), r in zip(case["atoms"], coords, strict=True)
        ],
    )


def lowdin(mol: gto.Mole, mo: numpy.ndarray) -> numpy.ndarray:
    """mo orthonormalized in mol's overlap, symmetrically."""
    values, vectors = numpy.linalg.eigh(mo.T @ mol.intor("int1e_ovlp") @ mo)
    return mo @ vectors @ numpy.diag(values**-0.5) @ vectors.T


def converged_state(name: str, case: dict, start: numpy.ndarray, sign: bool):
    """The DF SA-CASSCF at case's geometry from start, polished while |g|
    falls (to POLISH_BAR or below), then QD-NEVPT2's inputs over its
    SA-canonical orbitals. Returns the State, |g| and the orbital Hessian's
    least eigenvalue."""
    mol, auxbasis, _ = tiny_molecule(case)
    nelecas, ncas = case["cas"]
    weights = case["weights"]
    mf = scf.RHF(mol)
    mf.mo_coeff = start
    mf.mo_occ = numpy.zeros(start.shape[1])
    mf.mo_occ[: mol.nelectron // 2] = 2
    mc = casscf(mf, ncas, nelecas, weights, auxbasis)
    optimize(mc, mc.kernel, start)
    assert mc.converged, f"{name}: mc1step did not converge"
    mo = mc.mo_coeff
    best, stalled = None, 0
    for _ in range(POLISH_STEPS):
        energies, _, _, g, hessian, _ = derivatives(mc, mo, weights)
        norm = float(numpy.linalg.norm(g))
        if best is None or norm < best[0]:
            best, stalled = (norm, mo, energies, hessian), 0
        else:
            stalled += 1
        if stalled == POLISH_STALL or best[0] < POLISH_DONE:
            break
        mo = mo @ mc1step.expmat(mc.unpack_uniq_var(-numpy.linalg.solve(hessian, g)))
    norm, mo, energies, hessian = best
    assert norm < POLISH_BAR, f"{name}: the polish stopped at |g| {norm}"
    curvature = float(numpy.linalg.eigvalsh(hessian)[0])
    assert curvature > POLISH_CURVATURE, f"{name}: H's least eigenvalue {curvature}"
    casci, energies, _ = sa_canonical(
        name, mf, auxbasis, mo, energies, weights, case["cas"], sign=sign
    )
    eris = dfnevpt2._ERIS(casci, casci.mo_coeff, casci.with_df)
    state = State(
        name=name,
        record={},
        casci=casci,
        mo=casci.mo_coeff,
        mo_energy=casci.mo_energy,
        ci=list(casci.ci),
        energies=numpy.asarray(energies),
        weights=weights,
        eris=eris,
        blocks=integral_blocks(name, casci, casci.mo_coeff, eris),
    )
    return state, mol, norm, curvature


def qd(state: State) -> dict:
    """SC's and PC's symmetrized H_eff over the state, and PC's cut counts."""
    rdms, _, _ = transition_rdms(state)
    sc, _ = sc_qd(state, rdms)
    pc, info = pc_qd(state, rdms, sc)
    assert not info["refusals"], f"{state.name}: PC refused: {info['refusals']}"
    return {
        "sc": sc["heff_sym"],
        "pc": pc["pc_heff_sym"],
        "pc_dropped": pc["pc_dropped"],
        "sc_full": sc,
        "pc_full": pc,
    }


def eigenvectors(heff: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
    """heff's eigenpairs, each column's largest component positive."""
    values, vectors = numpy.linalg.eigh(heff)
    for col in range(vectors.shape[1]):
        if vectors[numpy.argmax(numpy.abs(vectors[:, col])), col] < 0:
            vectors[:, col] *= -1
    return values, vectors


def embedded(state: State, nmo: int, nelec) -> list[numpy.ndarray]:
    """Each root as a full-space (all-orbital) determinant vector."""
    ncore, ncas = state.casci.ncore, state.casci.ncas
    cas = [fci.cistring.make_strings(range(ncas), n) for n in state.casci.nelecas]
    core = (1 << ncore) - 1
    rows, cols = (
        fci.cistring.strs2addr(nmo, n, core | (s << ncore))
        for n, s in zip(nelec, cas, strict=True)
    )
    na, nb = (fci.cistring.num_strings(nmo, n) for n in nelec)
    out = []
    for vector in state.ci:
        full = numpy.zeros((na, nb))
        full[numpy.ix_(rows, cols)] = vector
        out.append(full)
    return out


def matched(overlap: numpy.ndarray, what: str) -> tuple[numpy.ndarray, numpy.ndarray]:
    """For each reference index M, the displaced index matched to it (largest
    |overlap[M, :]|, a permutation) and the sign making that overlap
    positive; refuses a match short of 1 by GUARD."""
    perm = numpy.argmax(numpy.abs(overlap), axis=1)
    assert len(set(perm.tolist())) == len(perm), f"{what}: no one-to-one match"
    diag = overlap[numpy.arange(len(perm)), perm]
    assert numpy.abs(diag).min() > 1 - GUARD, f"{what}: matched overlap {diag}"
    return perm, numpy.sign(diag)


class Reference:
    """The stationary reference at R0: its state, H_eff, R and Psi^(1)
    (each scheme's, summed over classes), and the bras of every overlap."""

    def __init__(self, name: str, case: dict, start: numpy.ndarray):
        self.name, self.case = name, case
        self.state, self.mol, self.gradient_norm, self.curvature = converged_state(
            name, case, start, sign=True
        )
        self.space = full_space(self.state)
        self.qd = qd(self.state)
        self.nmo, self.nelec = self.space["nmo"], self.space["nelec"]
        self.values, self.r = {}, {}
        self.psi1, self.checks = {}, {}
        sc, pc = self.qd["sc_full"], self.qd["pc_full"]
        self.psi1["sc"] = sc_psi1(self.state, self.space)
        self.psi1["pc"] = pc_psi1(self.state, self.space, pc)
        for scheme, couplings in (("sc", sc["couplings"]), ("pc", pc["pc_couplings"])):
            self.checks[scheme] = psi1_checks(
                f"{name} {scheme} (polished)",
                self.space,
                self.psi1[scheme],
                couplings,
            )
            self.values[scheme], self.r[scheme] = eigenvectors(self.qd[scheme])
        # Every bra at R0: the roots, then each scheme's Psi_M^(1).
        self.bras = list(self.space["roots"]) + [
            p for scheme in ("sc", "pc") for p in self.psi1[scheme].sum(axis=1)
        ]

    def point(self, coords: numpy.ndarray, start: numpy.ndarray) -> dict:
        """One displaced point: the matched overlaps and each scheme's
        <Psi_P^(0)(R0)|Psi_Q^(0)(R)> (Z) and <Psi_P^(1)(R0)|Psi_Q^(0)(R)> (Y)."""
        case = moved_case(self.case, coords)
        mol = tiny_molecule(case)[0]
        state, _, norm, _ = converged_state(
            self.name, case, lowdin(mol, start), sign=False
        )
        heff = qd(state)
        dropped = numpy.array_equal(heff["pc_dropped"], self.qd["pc_dropped"])
        assert dropped, f"{self.name}: PC's cut changed at a displaced point"
        cross = gto.intor_cross("int1e_ovlp", self.mol, mol)
        s = self.state.mo.T @ cross @ state.mo
        kets = embedded(state, self.nmo, self.nelec)
        o = numpy.array(
            [
                [float(fci.addons.overlap(b, k, self.nmo, self.nelec, s)) for k in kets]
                for b in self.bras
            ]
        )
        n = len(kets)
        perm, sign = matched(o[:n], f"{self.name} roots")
        o = o[:, perm] * sign[None, :]
        out = {
            "gradient_norm": norm,
            "ci": float(1 - numpy.abs(numpy.diag(o[:n])).min()),
        }
        for index, scheme in enumerate(("sc", "pc")):
            h = heff[scheme][numpy.ix_(perm, perm)] * numpy.outer(sign, sign)
            _, vectors = numpy.linalg.eigh(h)
            r0 = self.r[scheme]
            vperm, vsign = matched(r0.T @ vectors, f"{self.name} {scheme} R")
            r = vectors[:, vperm] * vsign[None, :]
            o1 = o[n * (index + 1) : n * (index + 2)]
            out[scheme] = {
                "z": r0.T @ o[:n] @ r,
                "y": r0.T @ o1 @ r,
                "r_overlap": float(1 - numpy.abs(numpy.diag(r0.T @ r)).min()),
            }
        out["mo"] = state.mo
        return out


# ----------------------------------------------------------- differences


def rigid_motions(coords: numpy.ndarray) -> numpy.ndarray:
    """Orthonormal translations and rotations about the centroid, (6, 3N)."""
    centred = coords - coords.mean(axis=0)
    rows = []
    for axis in numpy.eye(3):
        rows.append(numpy.tile(axis, len(coords)))
        rows.append(numpy.cross(axis, centred).ravel())
    q, _ = numpy.linalg.qr(numpy.array(rows).T)
    return q.T


def components(name: str, coords: numpy.ndarray, full: bool) -> tuple:
    """The directions (qd_nevpt2_gradient/'s, checked unit, orthogonal and
    free of rigid motion), and with full every Cartesian unit vector."""
    record = json.loads((GRADIENT / f"{name}.json").read_text())
    directions = numpy.array(record["directions"], dtype=float)
    gram = directions @ directions.T
    assert numpy.abs(gram - numpy.eye(len(gram))).max() < RIGID, name
    rigid = numpy.abs(directions @ rigid_motions(coords).T).max()
    assert rigid < RIGID, f"{name}: a direction has rigid motion {rigid}"
    cartesian = numpy.eye(coords.size) if full else numpy.zeros((0, coords.size))
    return record["seed"], directions, cartesian


def ladder(values: list, steps: list, noise: numpy.ndarray) -> dict:
    """Central differences D(h) of the [-h, +h] pairs (arrays), Richardson
    over the two finest, the error bar, the Richardson value's distance from
    the finest D, and the ladder ratio (D(4h) - D(2h)) / (D(2h) - D(h))
    wherever the finer rung difference clears LADDER_FLOOR and 5 noise / h."""
    d = numpy.array(
        [
            (plus - minus) / (2 * h)
            for (minus, plus), h in zip(values, steps, strict=True)
        ]
    )
    fine = (4 * d[2] - d[1]) / 3
    coarse = (4 * d[1] - d[0]) / 3
    rung = d[1] - d[2]
    gated = numpy.abs(rung) > numpy.maximum(LADDER_FLOOR, 5 * noise / steps[2])
    return {
        "central": d,
        "richardson": fine,
        "error_bar": numpy.abs(fine - coarse) + noise / (2 * steps[2]),
        "richardson_minus_finest": numpy.abs(fine - d[2]),
        "ratios": (d[0] - d[1])[gated] / rung[gated],
    }


def measure_noise(ref: Reference, coords, starts: list) -> dict:
    """Each point's overlap error, per scheme, term and pair: the reference
    geometry re-converged from each start (another point's orbitals carried
    back), its Z and (Y - Y^T) / 2 against the reference's own (the identity
    and zero); the largest over the starts, times NOISE_MARGIN, at least
    NOISE_FLOOR. Returns {scheme: {term: (n, n) noise}} and the measured."""
    n = len(ref.state.ci)
    measured = {s: {t: numpy.zeros((n, n)) for t in ("zeroth", "pt2")} for s in SCHEMES}
    for start in starts:
        again = ref.point(coords, start)
        for s in SCHEMES:
            z, y = again[s]["z"], again[s]["y"]
            for term, miss in (("zeroth", z - numpy.eye(n)), ("pt2", 0.5 * (y - y.T))):
                measured[s][term] = numpy.maximum(measured[s][term], numpy.abs(miss))
    noise = {}
    for s in SCHEMES:
        bound = {
            t: numpy.maximum(NOISE_FLOOR, NOISE_MARGIN * measured[s][t])
            for t in ("zeroth", "pt2")
        }
        bound["total"] = bound["zeroth"] + bound["pt2"]
        noise[s] = bound
    return noise, measured


def differences(ref: Reference, fd: dict, noise: dict, coords, rows) -> tuple:
    """Per row u of rows, every pair P < Q's ladder per scheme and term
    (zeroth: Z; pt2: (Y - Y^T) / 2; total: their sum), the guard's worst over
    the points, and each point's raw Z and Y."""
    steps = fd["steps"]
    n = len(ref.state.ci)
    pairs = list(itertools.combinations(range(n), 2))
    index = tuple(numpy.array(pairs).T)
    guard = {"gradient_norm": 0.0, "ci": 0.0, "sc_vectors": 0.0, "pc_vectors": 0.0}
    out = {s: {t: [] for t in TERMS} for s in SCHEMES}
    displaced = []
    for u in rows:
        raw = {s: {"z": [], "y": []} for s in SCHEMES}
        norms = []
        for h in steps:
            points = [
                ref.point(coords + sign * h * u.reshape(coords.shape), ref.state.mo)
                for sign in (-1, 1)
            ]
            for q in points:
                guard["gradient_norm"] = max(guard["gradient_norm"], q["gradient_norm"])
                guard["ci"] = max(guard["ci"], q["ci"])
                for s in SCHEMES:
                    key = f"{s}_vectors"
                    guard[key] = max(guard[key], q[s]["r_overlap"])
            norms.append([q["gradient_norm"] for q in points])
            for s in SCHEMES:
                for k in ("z", "y"):
                    raw[s][k].append([q[s][k] for q in points])
        for s in SCHEMES:
            zeroth = raw[s]["z"]
            pt2 = [[0.5 * (y - y.T) for y in pair] for pair in raw[s]["y"]]
            total = [
                [a + b for a, b in zip(pa, pb, strict=True)]
                for pa, pb in zip(zeroth, pt2, strict=True)
            ]
            for term, values in zip(TERMS, (zeroth, pt2, total), strict=True):
                lad = ladder(values, steps, noise[s][term])
                out[s][term].append(
                    {
                        "central": [d[index].tolist() for d in lad["central"]],
                        "richardson": lad["richardson"][index].tolist(),
                        "error_bar": lad["error_bar"][index].tolist(),
                        "richardson_minus_finest": lad["richardson_minus_finest"][
                            index
                        ].tolist(),
                        "ratios": lad["ratios"].tolist(),
                    }
                )
        displaced.append(
            {"gradient_norm": norms}
            | {
                s: {k: [[m.tolist() for m in pair] for pair in raw[s][k]] for k in "zy"}
                for s in SCHEMES
            }
        )
    return out, guard, displaced


def ladder_checks(name: str, family: dict) -> dict:
    """The gates over every component of one family (directional or full):
    every error bar under ERROR_BAR, every gated ladder ratio in
    LADDER_WINDOW. Returns the worst numbers."""
    out = {}
    for s in SCHEMES:
        for term in TERMS:
            rows = family[s][term]
            bar = max(max(r["error_bar"]) for r in rows)
            ratios = [x for r in rows for x in r["ratios"]]
            assert bar < ERROR_BAR, f"{name} {s} {term}: error bar {bar}"
            lo, hi = LADDER_WINDOW
            assert all(lo < x < hi for x in ratios), f"{name} {s} {term}: {ratios}"
            out[f"{s}_{term}"] = {
                "error_bar": bar,
                "richardson_minus_finest": max(
                    max(r["richardson_minus_finest"]) for r in rows
                ),
                "ladder_components": len(ratios),
                "ladder_ratios": [min(ratios), max(ratios)] if ratios else None,
            }
    return out


def visibility(families: list, pairs: list, densities: dict) -> dict:
    """Per pair and scheme, each separately coded term's largest |value|
    against its bar: d^PT2 and d^NEV + d^CAS over every component, the bar
    twice the component's error bar (#749's test); D^(MN)'s three blocks
    in the antisymmetric combination 1/2 (D^(MN) - D^(NM)) #746 tests, the
    bar DENSITY_BAR of that combination's largest element."""
    out = []
    for k, (m, n) in enumerate(pairs):
        row = {"pair": [m, n]}
        for s in SCHEMES:
            entry = {}
            for term in ("zeroth", "pt2"):
                values = [
                    (abs(r["richardson"][k]), 2 * r["error_bar"][k])
                    for family in families
                    for r in family[s][term]
                ]
                entry[term] = {
                    "max": max(v for v, _ in values),
                    "bar": max(b for _, b in values),
                    "max_over_bar": max(v / b for v, b in values),
                }
            density, masks = densities[s]
            anti = 0.5 * (density[m, n] - density[n, m])
            bar = DENSITY_BAR * numpy.abs(anti).max()
            for label, mask in masks.items():
                top = float(numpy.abs(anti[mask]).max())
                entry[f"D_{label}"] = {
                    "max": top,
                    "bar": float(bar),
                    "max_over_bar": top / float(bar),
                }
            row[s] = entry
        out.append(row)
    return out


def save(path: pathlib.Path, array) -> None:
    """One '<f8' Fortran-order .npy (the layout `src/io/npy` reads)."""
    numpy.save(path, numpy.asfortranarray(array, dtype="<f8"), allow_pickle=False)


def run_case(entry: dict) -> dict:
    case = entry["case"]
    name = case["name"]
    fd = FD[name]
    coords = numpy.array([c for _, c in case["atoms"]], dtype=float)

    # Part 2, and step 1's checks, on the fixture's own state.
    fixture = build_state(entry)
    committed = QD_OUT / name
    ci_miss = float(
        numpy.abs(numpy.load(committed / "ci.npy") - numpy.stack(fixture.ci)).max()
    )
    assert ci_miss == 0.0, f"{name}: the fixture's CI vectors differ by {ci_miss}"
    space = full_space(fixture)
    rdms, _, _ = transition_rdms(fixture)
    sc, _ = sc_qd(fixture, rdms)
    pc, info = pc_qd(fixture, rdms, sc)
    assert not info["refusals"], f"{name}: PC refused"
    heff_miss = max(
        float(numpy.abs(numpy.load(committed / "heff.npy") - sc["heff"]).max()),
        float(numpy.abs(numpy.load(committed / "pc_heff.npy") - pc["pc_heff"]).max()),
    )
    assert heff_miss < BRUTE_FORCE, f"{name}: H_eff off the committed by {heff_miss}"
    nmo = space["nmo"]
    masks = blocks_mask(fixture.casci.ncore, fixture.casci.ncas, nmo)
    out = OUT / name
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    fixture_checks = {"ci_vs_committed": ci_miss, "heff_vs_committed": heff_miss}
    densities = {}
    for s, psi, couplings in (
        ("sc", sc_psi1(fixture, space), sc["couplings"]),
        ("pc", pc_psi1(fixture, space, pc), pc["pc_couplings"]),
    ):
        checks = psi1_checks(f"{name} {s}", space, psi, couplings)
        density, off = pt2_density(fixture, space, psi)
        checks["density_off_blocks"] = off
        fixture_checks[s] = checks
        densities[s] = (density, masks)
        save(out / f"{s}_pt2_density.npy", density)
    print(f"{name}: fixture {json.dumps(fixture_checks)}", flush=True)

    # Part 3: the stationary reference, the noise, the differences.
    ref = Reference(name, case, fixture.mo)
    seed, directions, cartesian = components(name, coords, fd["full"])
    starts = [
        ref.point(coords + fd["steps"][0] * u.reshape(coords.shape), ref.state.mo)["mo"]
        for u in directions
    ]
    noise, measured = measure_noise(ref, coords, starts)
    worst = max(float(m.max()) for v in measured.values() for m in v.values())
    print(f"{name}: |g| {ref.gradient_norm:.1e}, noise {worst:.1e}", flush=True)
    directional, guard, raw = differences(ref, fd, noise, coords, directions)
    families = [directional]
    full = None
    if fd["full"]:
        full, guard_full, raw_full = differences(ref, fd, noise, coords, cartesian)
        families.append(full)
        guard = {k: max(v, guard_full[k]) for k, v in guard.items()}
    n = len(ref.state.ci)
    pairs = [list(p) for p in itertools.combinations(range(n), 2)]
    index = tuple(numpy.array(pairs).T)
    checks = {"directional": ladder_checks(f"{name} directional", directional)}
    if full is not None:
        checks["full"] = ladder_checks(f"{name} full", full)
    state = ref.state
    ncore, ncas = state.casci.ncore, state.casci.ncas
    nocc = ncore + ncas
    record = {
        "name": name,
        "basis": case["basis"],
        "aux_basis": "cc-pVDZ-JKFIT",
        "atoms": [{"symbol": s, "coord": list(c)} for s, c in case["atoms"]],
        "ncore": ncore,
        "ncas": ncas,
        "nelecas": case["cas"][0],
        "nmo": nmo,
        "weights": case["weights"],
        "steps": fd["steps"],
        "noise": {
            s: {t: v[index].tolist() for t, v in noise[s].items()} for s in SCHEMES
        },
        "noise_measured": {
            s: {t: v[index].tolist() for t, v in measured[s].items()} for s in SCHEMES
        },
        "seed": seed,
        "directions": directions.tolist(),
        "pairs": pairs,
        "fixture": fixture_checks,
        "reference": {
            "gradient_norm": ref.gradient_norm,
            "hessian_least_eigenvalue": ref.curvature,
            "casci_energies": state.energies.tolist(),
            "ci": [c.tolist() for c in state.ci],
            "inactive_gaps": {
                "core": float(numpy.diff(state.mo_energy[:ncore]).min())
                if ncore > 1
                else None,
                "virtual": float(numpy.diff(state.mo_energy[nocc:]).min()),
            },
            "psi1": ref.checks,
        }
        | {
            f"{s}_{k}": v
            for s in SCHEMES
            for k, v in (
                ("heff", ref.qd[s].tolist()),
                ("energies", ref.values[s].tolist()),
                ("vectors", ref.r[s].tolist()),
                ("heff_gaps", numpy.diff(ref.values[s]).tolist()),
            )
        },
        "directional": directional,
        "full": full,
        "guard": guard,
        "checks": checks,
        "visibility": visibility(families, pairs, densities),
        "displaced": {"directional": raw, "full": raw_full if full else None},
        "tolerances": {
            "brute_force": BRUTE_FORCE,
            "off_block": OFF_BLOCK,
            "polish": POLISH_BAR,
            "guard": GUARD,
            "error_bar": ERROR_BAR,
            "ladder_window": list(LADDER_WINDOW),
            "ladder_floor": LADDER_FLOOR,
            "noise_margin": NOISE_MARGIN,
            "noise_floor": NOISE_FLOOR,
            "density_bar": DENSITY_BAR,
        },
        "provenance": {
            "pyscf": pyscf.__version__,
            "numpy": numpy.__version__,
            "threads": 1,
            "command": COMMAND,
        },
    }
    (OUT / f"{name}.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def main() -> None:
    lib.num_threads(1)
    assert lib.num_threads() == 1
    names = sys.argv[1:] or list(FD)
    assert set(names) <= set(FD), f"no such case: {set(names) - set(FD)}"
    OUT.mkdir(parents=True, exist_ok=True)
    for entry in CASES:
        if entry.get("brute_force") and entry["case"]["name"] in names:
            record = run_case(entry)
            print(
                f"{record['name']}: checks {json.dumps(record['checks'])}\n"
                f"  guard {json.dumps(record['guard'])}",
                flush=True,
            )


if __name__ == "__main__":
    main()
