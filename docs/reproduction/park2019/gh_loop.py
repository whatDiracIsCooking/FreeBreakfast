"""Tier 2b (#849): the g-h loop around PSB3's 13-14CI, Park's Figure 5.

Writes tier2b/theta_NNN.xyz and .toml at x(t) = x0 + r (cos t g^ + sin t h^),
r = 0.002 Angstrom, t = 0, 15, ..., 345 degrees. g^ = (g_Q - g_P)/|g_Q - g_P|
and h^ is h_PQ with its g^ component removed, normalized (Gram-Schmidt, g^
first), both over the 3N Cartesians. h_PQ is used as the intersection signs
it, so h^'s sign, and with it the loop's direction, follows the CI phases.
Run from this directory: python gh_loop.py (tier2b.md, "The plane").
"""

import json
import math
from pathlib import Path

HERE = Path(__file__).parent
CONFIG = """# Tier 2b (#849): the g-h loop point theta = {deg} degrees, written by
# gh_loop.py. The MECI's [casscf] (tier2a_psb3_1314_qdpc.toml); [nevpt2] is
# the observable. [optimize] stops at its first evaluation (every bar 1), only
# so optimize.json records this point's CASSCF |g| and rho (tier2b.md).
[molecule]
xyz = "tier2b/theta_{deg:03d}.xyz"
charge = 1
[basis_set]
basis = "cc-pvdz"
[eri]
mode = "density_fitting"
aux_basis = "cc-pvdz-jkfit"
[scf]
max_iterations = 400
[casscf]
ncas = 6
nelecas = 6
nroots = 3
conv_tol_grad = 1e-9
[nevpt2]
quasi_degenerate = true
[nevpt2_gradient]
quasi_degenerate = true
variant = "pc"
[optimize]
meci = [0, 1]
max_gap = 1.0
max_force = 1.0
rms_force = 1.0
max_displacement = 1.0
rms_displacement = 1.0
max_steps = 0
"""


def unit(v):
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


space = json.loads((HERE / "tier2b" / "branching_space.json").read_text())
flat = {k: [x for row in space[k] for x in row] for k in ("g_p", "g_q", "h_pq")}
g = unit([q - p for p, q in zip(flat["g_p"], flat["g_q"], strict=True)])
gh = sum(a * b for a, b in zip(g, flat["h_pq"], strict=True))
h = unit([b - gh * a for a, b in zip(g, flat["h_pq"], strict=True)])
lines = (HERE / "geometries" / "psb3_1314_meci_ccpvdz.xyz").read_text().splitlines()
atoms = [line.split() for line in lines[2 : 2 + int(lines[0])]]
for deg in range(0, 360, 15):
    t = math.radians(deg)
    frame = [lines[0], f"PSB3 13-14CI g-h loop, r = 0.002 A, theta = {deg} deg"]
    d = [0.002 * (math.cos(t) * a + math.sin(t) * b) for a, b in zip(g, h, strict=True)]
    for i, (el, *xyz) in enumerate(atoms):
        new = [float(x) + d[3 * i + k] for k, x in enumerate(xyz)]
        frame.append(el + "".join(f" {v:20.12f}" for v in new))
    (HERE / "tier2b" / f"theta_{deg:03d}.xyz").write_text("\n".join(frame) + "\n")
    (HERE / "tier2b" / f"theta_{deg:03d}.toml").write_text(CONFIG.format(deg=deg))
