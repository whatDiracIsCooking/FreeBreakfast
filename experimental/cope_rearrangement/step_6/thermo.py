"""Harmonic analysis and RRHO thermochemistry from a Cartesian Hessian,
in numpy.

Everything takes the Cartesian Hessian free-breakfast writes (hessian.json's
"hessian", hartree/bohr^2, already symmetrised) and the geometry (bohr).

Units: energies in hartree (Eh) unless named _kcal; entropies in Eh/K or
cal/(mol K) (_eu); frequencies in cm^-1, an imaginary one negative.
Ideal gas, rigid rotor, harmonic oscillator; 1 atm standard state (it
cancels in a unimolecular ΔG‡).
"""

from __future__ import annotations

import numpy

# CODATA 2018.
H_PLANCK = 6.62607015e-34  # J s
K_B = 1.380649e-23  # J/K
C_CM = 2.99792458e10  # cm/s
N_A = 6.02214076e23
AMU = 1.66053906660e-27  # kg
EH_J = 4.3597447222071e-18  # J
A0 = 5.29177210903e-11  # m
CAL = 4.184  # J
EH_KCAL = EH_J * N_A / CAL / 1000.0  # 627.5094740631
R_EH = K_B / EH_J  # Eh/K per molecule
EU_PER_EH_K = EH_J * N_A / CAL  # cal/(mol K) per Eh/K
CM1_PER_AU = numpy.sqrt(EH_J / (A0**2 * AMU)) / (2.0 * numpy.pi * C_CM)
PRESSURE = 101325.0  # Pa

# Most abundant isotopes (as Gaussian's thermochemistry uses), and the
# isotope-averaged standard weights free-breakfast's own harmonic analysis
# uses (PySCF's elements.MASSES): the latter only to check against the
# tree's frequencies.
MASS = {"H": 1.00782503223, "C": 12.0}
MASS_AVERAGED = {"H": 1.008, "C": 12.011}


def masses(symbols, table=MASS) -> numpy.ndarray:
    """Per-atom masses (amu)."""
    return numpy.array([table[s] for s in symbols])


def rigid_body_basis(x, m) -> numpy.ndarray:
    """Orthonormal 3N x 6 basis of the mass-weighted translations and
    infinitesimal rotations about the centre of mass."""
    sq = numpy.sqrt(m)
    xc = x - (m @ x) / m.sum()
    vecs = []
    for k in range(3):
        t = numpy.zeros_like(x)
        t[:, k] = 1.0
        vecs.append((t * sq[:, None]).ravel())
    for axis in numpy.eye(3):
        vecs.append((numpy.cross(axis, xc) * sq[:, None]).ravel())
    q, r = numpy.linalg.qr(numpy.array(vecs).T)
    assert abs(numpy.diag(r)).min() > 1e-6, "linear or degenerate geometry"
    return q


def harmonic(hessian, x, m):
    """(frequencies cm^-1 ascending, imaginary negative; Cartesian modes
    3N-6 x 3N, each the mass-weighted unit eigenvector over sqrt(m)).
    The rigid-body space is removed exactly by diagonalising in its
    orthogonal complement, not by a projector and a threshold."""
    hessian = numpy.asarray(hessian)
    sq = numpy.repeat(numpy.sqrt(m), 3)
    hmw = hessian / numpy.outer(sq, sq)
    tr = rigid_body_basis(x, m)
    full, _ = numpy.linalg.qr(numpy.hstack([tr, numpy.eye(len(sq))]))
    comp = full[:, 6:]
    w, v = numpy.linalg.eigh(comp.T @ hmw @ comp)
    freqs = numpy.sign(w) * numpy.sqrt(abs(w)) * CM1_PER_AU
    modes = (comp @ v).T / sq[None, :]
    return freqs, modes


def rigid_body_eigenvalues(hessian, x, m) -> numpy.ndarray:
    """The six 'frequencies' (cm^-1) of the mass-weighted Hessian in the
    rigid-body space itself: how near zero translation and rotation are."""
    sq = numpy.repeat(numpy.sqrt(m), 3)
    hmw = numpy.asarray(hessian) / numpy.outer(sq, sq)
    tr = rigid_body_basis(x, m)
    w = numpy.linalg.eigvalsh(tr.T @ hmw @ tr)
    return numpy.sign(w) * numpy.sqrt(abs(w)) * CM1_PER_AU


def moments(x, m) -> numpy.ndarray:
    """Principal moments of inertia (amu bohr^2), ascending."""
    xc = x - (m @ x) / m.sum()
    i = numpy.einsum("a,ax,ay->xy", m, xc, xc)
    i = numpy.eye(3) * numpy.trace(i) - i
    return numpy.linalg.eigvalsh(i)


def rrho(e_el, freqs, x, m, temperature, sigma=1, scale=1.0) -> dict:
    """RRHO terms at `temperature` for one species. `freqs` the 3N-6 from
    harmonic(); a TS's imaginary one (negative) is left out, as the
    reaction coordinate. `scale` multiplies every real frequency (1:
    unscaled). Returns Eh and Eh/K, and ln q_rot + ln q_vib (q_vib from the
    bottom of the well, so e_el + zpe is not double counted: ln_q_vib_bot
    includes -zpe/kT)."""
    t = temperature
    nu = numpy.asarray(freqs)
    real = nu[nu > 0] * scale
    u = H_PLANCK * C_CM * real / (K_B * t)
    zpe = 0.5 * H_PLANCK * C_CM * real.sum() / EH_J
    e_vib = zpe + R_EH * t * (u / numpy.expm1(u)).sum()
    s_vib = R_EH * (u / numpy.expm1(u) - numpy.log1p(-numpy.exp(-u))).sum()
    ln_q_vib_bot = -zpe / (R_EH * t) - numpy.log1p(-numpy.exp(-u)).sum()

    i_si = moments(x, m) * AMU * A0**2
    ln_q_rot = numpy.log(
        numpy.sqrt(numpy.pi)
        / sigma
        * numpy.prod(numpy.sqrt(8 * numpy.pi**2 * i_si * K_B * t / H_PLANCK**2))
    )
    s_rot = R_EH * (ln_q_rot + 1.5)
    e_rot = 1.5 * R_EH * t

    mass = m.sum() * AMU
    q_trans = (2 * numpy.pi * mass * K_B * t / H_PLANCK**2) ** 1.5 * K_B * t / PRESSURE
    s_trans = R_EH * (numpy.log(q_trans) + 2.5)
    h_trans = 2.5 * R_EH * t  # E_trans + pV

    h = e_el + e_vib + e_rot + h_trans
    s = s_vib + s_rot + s_trans
    return {
        "temperature": t,
        "e_el": e_el,
        "zpe": zpe,
        "e_vib": e_vib,
        "h_thermal": e_vib + e_rot + h_trans,
        "h": h,
        "s": s,
        "s_vib": s_vib,
        "s_rot": s_rot,
        "s_trans": s_trans,
        "g": h - t * s,
        "ln_q_rot": float(ln_q_rot),
        "ln_q_vib_bot": float(ln_q_vib_bot),
        "ln_q_trans": float(numpy.log(q_trans)),
        "sigma": sigma,
    }


def ensemble(members: list[tuple[dict, float]]) -> dict:
    """A Boltzmann ensemble of species at one temperature. Each member is
    (rrho dict, g): g the number of distinct structures it stands for
    (2 for a chiral conformer's two enantiomers; the rotational sigma is
    already in its rrho). Returns h, s, g of the ensemble and the weights:
    G = -kT ln sum_i g_i exp(-G_i/kT), H = sum_i w_i H_i,
    S = (H - G)/T."""
    t = members[0][0]["temperature"]
    gs = numpy.array([r["g"] - R_EH * t * numpy.log(g) for r, g in members])
    g0 = gs.min()
    w = numpy.exp(-(gs - g0) / (R_EH * t))
    z = w.sum()
    w /= z
    g_tot = g0 - R_EH * t * numpy.log(z)
    h_tot = float(sum(wi * r["h"] for wi, (r, _) in zip(w, members, strict=True)))
    return {
        "temperature": t,
        "h": h_tot,
        "g": float(g_tot),
        "s": (h_tot - g_tot) / t,
        "weights": [float(v) for v in w],
    }


def activation(ts: dict, reactant: dict) -> dict:
    """ΔH‡, ΔS‡, ΔG‡ (kcal/mol, cal/(mol K)) and the Eyring rate constant
    k = (kT/h) exp(-ΔG‡/RT) (s^-1) from two rrho/ensemble dicts."""
    t = ts["temperature"]
    dh = (ts["h"] - reactant["h"]) * EH_KCAL
    dg = (ts["g"] - reactant["g"]) * EH_KCAL
    ds = (ts["s"] - reactant["s"]) * EU_PER_EH_K
    k = K_B * t / H_PLANCK * numpy.exp(-(ts["g"] - reactant["g"]) / (R_EH * t))
    return {
        "temperature": t,
        "dh_kcal": dh,
        "ds_eu": ds,
        "dg_kcal": dg,
        "k_per_s": float(k),
    }
