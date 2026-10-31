#!/usr/bin/env python3
"""Step 2b: the a_u curvature at one step-1 geometry, CASSCF and NEVPT2.

1. CASSCF: the full 10 x 10 a_u Hessian from analytic DF-CASSCF gradients.
   Displace by h along each a_u basis vector q_j (au.au_basis); the a_u part
   of the gradient is odd in an a_u displacement (x and -x are images under
   i), so H[:, j] = Q^T g(x0 + h q_j) / h has the error of a central
   difference, O(h^2), from one gradient.
2. NEVPT2 (PC, and SC beside it): energies only, so the Hessian is taken in
   the subspace V of the NSUB softest CASSCF a_u eigenvectors:
   E(h v_a) and E(h (v_a + v_b)), au.hessian_from_energies. The CASSCF
   energies at those same points give a second, energy-based CASSCF Hessian
   in V, which must match the gradient-based one projected onto V: the check
   that h and the convergence thresholds are right.
   One more point, E(-h v_0), must equal E(h v_0) by symmetry: the noise
   floor of the PT2 energies.

The NEVPT2 lowest eigenvalue in V is a Rayleigh-Ritz upper bound on the full
block's: a negative value there means the full a_u block has a negative
eigenvalue (concerted); a positive one is only conclusive as far as V spans
the soft modes. `hybrid` puts NEVPT2's correction inside V onto the full
CASSCF block, H_cas + V (H_pt2,V - H_cas,V) V^T, and reports its frequencies;
outside V it is CASSCF's.

    UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --group tools \\
        experimental/cope_rearrangement/step_2/curvature.py R [--threads N]

Writes curvature/R<r>.json; every expensive evaluation is cached under
logs/cache_R<r>/ so a killed run resumes.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import au  # noqa: E402
import pt2  # noqa: E402
from scan_pt2 import load_step1, tag  # noqa: E402

H_STEP = 0.02  # bohr, along a unit-norm Cartesian a_u vector
NSUB = 3
OUT = HERE / "curvature"
# Energy- vs gradient-based CASSCF Hessian in V, Eh/bohr^2.
CAS_AGREE = 2e-3


def subspace(h_cas, b):
    """V, 10 x NSUB orthonormal: first the CASSCF relaxed-s direction
    H^-1 b (the a_u displacement that changes s with every other a_u
    coordinate at its CASSCF minimum, which is what k_s measures), then the
    softest CASSCF a_u eigenvectors, orthogonalised against it. The soft
    modes alone can miss s: at R = 1.65 A the three softest overlap s's
    direction by |cos| = 0.01."""
    w, v = np.linalg.eigh(h_cas)
    relaxed = np.linalg.solve(h_cas, b)
    cols = [relaxed / np.linalg.norm(relaxed)]
    for k in range(len(w)):
        if len(cols) == NSUB:
            break
        x = v[:, k] - sum((c @ v[:, k]) * c for c in cols)
        if np.linalg.norm(x) > 0.3:  # skip a soft mode that is mostly s
            cols.append(x / np.linalg.norm(x))
    v_sub = np.column_stack(cols)
    for a in range(NSUB):  # reproducible signs
        if v_sub[np.argmax(np.abs(v_sub[:, a])), a] < 0:
            v_sub[:, a] *= -1.0
    return v_sub


class Point:
    """CASSCF (+ optionally gradient, NEVPT2) at x0 + disp, cached on disk.
    Each starts from the reference point's orbitals and CI, re-orthonormalised
    in the displaced geometry's overlap (pt2.casscf)."""

    def __init__(self, r, atoms0, mo0, ci0):
        self.sym = [s for s, _ in atoms0]
        self.x0 = np.array([x for _, x in atoms0])
        self.mo0, self.ci0 = mo0, ci0
        self.cache = HERE / "logs" / f"cache_{tag(r)}"
        self.cache.mkdir(parents=True, exist_ok=True)

    def __call__(self, key, disp, grad=False, pt=False):
        # A key names a displacement only together with its direction.
        digest = hashlib.sha1(np.round(disp, 10).tobytes()).hexdigest()[:8]
        f = self.cache / f"{key}_{digest}.json"
        if f.exists():
            rec = json.loads(f.read_text())
            if (not grad or "grad" in rec) and (not pt or rec.get("sc") == "natorb"):
                return rec
        t0 = time.perf_counter()
        x = self.x0 + disp.reshape(-1, 3)
        mol = pt2.make_mol(list(zip(self.sym, x, strict=True)))
        mc = pt2.casscf(mol, self.mo0, self.ci0)
        rec = {"e_casscf": float(mc.e_tot)}
        if grad:
            g = mc.nuc_grad_method().kernel()
            rec["grad"] = np.asarray(g).ravel().tolist()
        if pt:
            res = pt2.nevpt2(mc)
            rec["e_pc"] = float(mc.e_tot) + res["e_corr_pc"]
            rec["e_sc"] = float(mc.e_tot) + res["e_corr_sc"]
            rec["sc"] = "natorb"
        rec["seconds"] = time.perf_counter() - t0
        f.write_text(json.dumps(rec) + "\n")
        print(
            f"  {key}: E_cas {rec['e_casscf']:.10f} ({rec['seconds']:.0f}s)", flush=True
        )
        return rec


def run(r):
    _rec, atoms, mo = load_step1(r)
    x0 = np.array([x for _, x in atoms])
    q, t = au.au_basis(x0)
    mol0 = pt2.make_mol(atoms)
    masses = mol0.atom_mass_list(isotope_avg=False)

    mc0 = pt2.casscf(mol0, mo)
    pts = Point(r, atoms, mc0.mo_coeff, mc0.ci)
    ref = pts("ref", np.zeros(x0.size), grad=True, pt=True)
    g0 = np.array(ref["grad"])
    print(
        f"[{tag(r)}] |Q^T g0| = {np.linalg.norm(q.T @ g0):.1e} (zero by symmetry)",
        flush=True,
    )

    # 1. CASSCF a_u Hessian from gradients.
    h_cas = np.zeros((au.VIB_DIM, au.VIB_DIM))
    for j in range(au.VIB_DIM):
        g = np.array(pts(f"q{j}", H_STEP * q[:, j], grad=True)["grad"])
        h_cas[:, j] = q.T @ g / H_STEP
    asym = float(np.abs(h_cas - h_cas.T).max())
    h_cas = 0.5 * (h_cas + h_cas.T)
    v_sub = subspace(h_cas, q.T @ au.s_bvector(x0))

    # 2. Energies in V.
    dirs = q @ v_sub  # Cartesian, unit norm
    e0 = {k: ref[k] for k in ("e_casscf", "e_pc", "e_sc")}
    single = [pts(f"v{a}", H_STEP * dirs[:, a], pt=True) for a in range(NSUB)]
    pair = [[None] * NSUB for _ in range(NSUB)]
    for a, b in itertools.combinations(range(NSUB), 2):
        pair[a][b] = pts(f"v{a}v{b}", H_STEP * (dirs[:, a] + dirs[:, b]), pt=True)
    mirror = pts("v0minus", -H_STEP * dirs[:, 0], pt=True)

    def sub_hessian(k):
        return au.hessian_from_energies(
            e0[k],
            [s[k] for s in single],
            [[p[k] if p else None for p in row] for row in pair],
            H_STEP,
        )

    h_v = {k: sub_hessian(k) for k in ("e_casscf", "e_pc", "e_sc")}
    h_cas_v = v_sub.T @ h_cas @ v_sub
    cas_diff = float(np.abs(h_v["e_casscf"] - h_cas_v).max())
    if cas_diff > CAS_AGREE:
        print(
            f"[{tag(r)}] WARNING energy vs gradient CASSCF Hessian in V: {cas_diff:.2e}"
        )

    out = {
        "r": r,
        "h_step_bohr": H_STEP,
        "nsub": NSUB,
        "energies": e0,
        "casscf": {
            "hessian_asymmetry": asym,
            **au.analyse(h_cas, q, t, masses, x0),
        },
        "subspace_check": {
            "energy_vs_gradient_casscf_max_abs": cas_diff,
            "agrees": cas_diff <= CAS_AGREE,
            "mirror_pc_minus_plus": mirror["e_pc"] - single[0]["e_pc"],
            "mirror_casscf_minus_plus": mirror["e_casscf"] - single[0]["e_casscf"],
        },
    }
    for k, name in (("e_pc", "pc_nevpt2"), ("e_sc", "sc_nevpt2")):
        hv = h_v[k]
        hybrid = h_cas + v_sub @ (hv - h_cas_v) @ v_sub.T
        out[name] = {
            "subspace_hessian": hv.tolist(),
            "subspace_eigenvalues": [float(x) for x in np.linalg.eigvalsh(hv)],
            "hybrid": au.analyse(hybrid, q, t, masses, x0),
        }
    out["casscf"]["subspace_eigenvalues"] = [
        float(x) for x in np.linalg.eigvalsh(h_cas_v)
    ]
    OUT.mkdir(exist_ok=True)
    (OUT / f"{tag(r)}.json").write_text(json.dumps(out, indent=1) + "\n")
    c, p = out["casscf"], out["pc_nevpt2"]
    print(
        f"[{tag(r)}] CASSCF: {c['n_negative']} neg, lowest a_u "
        f"{c['frequencies_cm1'][0]:.0f} cm-1, k_s {c['k_s']:.4f} | "
        f"PC in V: {p['subspace_eigenvalues'][0]:.4f}, "
        f"hybrid lowest {p['hybrid']['frequencies_cm1'][0]:.0f} cm-1, k_s "
        f"{p['hybrid']['k_s']:.4f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("r", type=float, nargs="+")
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args()
    from pyscf import lib

    if args.threads:
        lib.num_threads(args.threads)
    for r in args.r:
        run(r)


if __name__ == "__main__":
    main()
