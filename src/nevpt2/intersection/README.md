# `free_breakfast.nevpt2.intersection`

What a QD-NEVPT2 minimum-energy conical intersection search needs of one QD
pair (P, Q), SC or PC, on a DF-SA-CASSCF: the gradients g_P and g_Q and the
interstate coupling h_PQ at one set of H_eff eigenvectors R, with no refusal
of a degenerate pair, and two gap sensitivities that bound how far the
pair's gap moves when the SA-CASSCF is not fully converged. Link
`free_breakfast::nevpt2::intersection`; `free_breakfast.nevpt2` re-exports
it.

    qd_sc_intersection(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas,
                       p, q, batch, options = {}) -> QdIntersection
        options.shared  one several-W gradient call (default) or three W-form calls
        options.nfrozen the frozen core, 0 by default: H_eff and every G(W) over it
    qd_pc_intersection(...the same...) -> QdIntersection
        g_p, g_q       dE_P/dR = G(u_P u_Pᵀ), dE_Q/dR = G(u_Q u_Qᵀ), n_atom × 3 (hartree / bohr)
        h_pq           h_PQ = u_Pᵀ (∇H_eff) u_Q = G(W^PQ), W^PQ = ½ (u_P u_Qᵀ + u_Q u_Pᵀ)
        eigenvalues    H_eff's, ascending: the QD energies
        eigenvectors   R, column K at [K n], each signed so its largest component is positive
        heff_gap       |E_Q − E_P|, reported, never gated
        root_gap       the smallest CASSCF |E_r − E_s|
        gap_sensitivity_orbital   s_o
        gap_sensitivity_ci        s_c

The arguments are `qd_sc_coupling`'s (`../gradient/`), without its fold.
`options.nfrozen` is the frozen core (#875): `qd_inputs` and the W forms
take it, so H_eff, R and the three G(W) are all the frozen-core
functional's (`../gradient/README.md`, "The frozen core"); s_o and s_c read
the multipliers that come back, and need nothing of their own. The
consumer is the QD MECI mode of the optimize step (#785), which hands g_P,
g_Q and h_PQ to `geometry::optimize_meci` and reads s_o and s_c for its gap
bar.

## How it is computed

1. H_eff = diag(E_r) + ½ (C + Cᵀ) over `nevpt2::qd_inputs` and
   `sc_effective_hamiltonian` (or `pc_effective_hamiltonian`), diagonalized
   once on the device. R is signed as `QdScLagrangian::eigenvectors`.
2. G(W) at u_P u_Pᵀ, u_Q u_Qᵀ and W^PQ, all built from that one R: by
   default one several-W call of `qd_sc_gradient` (or `qd_pc_gradient`),
   which shares the adjoint's forward arrays, the CI-relaxed Hessian and one
   `mcscf::zvectors` solve of the three columns (`../gradient/README.md`,
   "Several W's at once"); with `options.shared = false`, three W-form
   calls, each its own Lagrangian and Z-vector solve (#784's path).
3. Each solve's true residual (`ZVector::residual`, one more product at the
   solution, not the conjugate gradients' recurrence) must be at or below the
   solve's own tolerance, `options.zvector_tolerance` (1e-10 by default, the
   one parameter the W-form calls solve to), or the call returns `Numerical`
   naming the W (the gradient layer checks every column too). The CI
   response inside H̃ is converged to `mcscf::response_tolerance_for` it
   (`../gradient/README.md` "The Z-vector's floor"). Near a degeneracy the recurrence
   can report convergence while the true residual has not (#786).
4. s_o and s_c from the three solves' multipliers, below.

**No refusal of a degenerate pair.** `qd_*_coupling` and the state form of
`qd_*_gradient` refuse when E_P and E_Q are within `mcscf::kRootGapFloor`,
because each needs u_K's own derivative. The W forms do not: G(W) is linear
in W and nothing divides by E_Q − E_P. At exact degeneracy u_P and u_Q are
any orthonormal pair of the two-state space, and g_P, g_Q, h_PQ turn with
them, but the branching plane span{g_Q − g_P, h_PQ} and ḡ = (g_P + g_Q)/2
do not. That is all the optimizer uses. An MECI's own convergence takes the
gap below the floor, so this is the case that matters.

**Unequal SA weights are allowed.** The only division by a gap is
`mcscf::root_pair_multipliers`' by the CASSCF roots', which keeps its
refusal. (`mcscf::df_intersection` refuses unequal weights because there the
roots themselves are the crossing pair; here the QD states are.)

## Sharing the three calls

`QdIntersectionOptions::shared` (#787) picks the path. Both give the same
result to the Z-vector solves' convergence (the tests hold them to 1e-9 of
the largest summand at the same R; PC's s_c to 1e-7, #784's ruling). The
shared path builds the adjoint's forward arrays and RDMs once instead of
three times, and the Hessian and the CI objects once, and solves the three
right-hand sides in one block conjugate-gradient run. H_eff's own walk
(step 1) is still separate: R must exist before any W does.

The derivative passes stay three (four when a W's ρ needs the split pass):
stacking them would need a multi-density derivative kernel, since one
`geri3` kernel is ~50 ms of each ~57 ms `df_gradient` call (#773, PR #776).
That is out of scope here, and it is why the gain is modest.

Measured on the RTX 3080 (CUDA), pair (0, 1), `Nevpt2QdIntersection.Timing`
(one warm-up of each path, then three of each, alternating; min / mean):

| case | scheme | three calls | shared |
|---|---|---:|---:|
| `h2co_skew_ccpvdz_sa3` | SC | 0.636 / 0.637 s | 0.568 / 0.568 s |
| `h2co_skew_ccpvdz_sa3` | PC | 0.653 / 0.654 s | 0.585 / 0.585 s |
| ethylene SA2-CAS(2,2)/cc-pVDZ | SC | 0.566 / 0.567 s | 0.538 / 0.539 s |
| ethylene SA2-CAS(2,2)/cc-pVDZ | PC | 0.558 / 0.559 s | 0.530 / 0.531 s |

The shared path is faster on every case, so it is the default. The
ethylene case is `data/casscf_meci/`'s start geometry with its SA2 weights:
#787 asked for #785's SA3, which this tree's CASSCF did not converge from
RHF orbitals at the MECI, the start or planar ethylene (the test's comment
has the numbers). The timing case is skipped unless
`FBF_QD_INTERSECTION_TIMING` is set.

## The gap sensitivities

Write the SA-CASSCF's parameters p (orbital rotations x over the packed
pairs, the CI vectors in `gen_g_hop`'s c → c + δc coordinates, and the
rotations Θ among the roots) and its stationarity conditions G(p) = (g,
2 w_r P (H₀ − E_r) c_r, Θ's). A solution with residual G sits δp = A⁻¹G +
O(‖G‖²) from the exact one. An element a_W = tr(W H_eff) is not stationary in
p, so it moves at first order, δa_W = −z^W · G, where z^W is the Z-vector of
the W form G(W): exactly the `lagrangian.multipliers` (z_o, z_c, Θ) of the
W-form call (#785's derivation).

In H_eff's eigenbasis the pair's block is [[a, b], [b, d]], with gap
√((a − d)² + 4b²), a = a_{u_P u_Pᵀ}, d = a_{u_Q u_Qᵀ} and b = a_{W^PQ}. The gap
has no derivative at a degeneracy, but it moves by at most √(δ(a − d)² +
4 δb²), and the triangle inequality in that plane gives

    |δgap| ≤ s_o ‖g‖ + s_c √n_roots ρ
    s_o = √(‖z_o^Q − z_o^P‖² + 4 ‖z_o^PQ‖²)
    s_c = √(‖ẑ_c^Q − ẑ_c^P‖² + 4 ‖ẑ_c^PQ‖²),   ẑ_c = (2 w_r z_c,r)_r

**The pairing, pinned.** Each block of the multipliers pairs with one
residual the SA-CASSCF reports:

- **The orbital multipliers, factor 1, with ‖g‖** =
  `mcscf::Solution::gradient_norm`. That norm is `response::Fock`'s
  gradient alone, g_pq = 2 (F_pq − F_qp) = dE_SA/dx_pq over the packed
  pairs, the very vector the Z-vector system's orbital rows are the
  derivative of.
- **The CI multipliers, each root's block scaled by 2 w_r, with ρ** =
  `ci::Roots::davidson.max_residual_norm`. The Z-vector system's CI rows are
  the true ∂L/∂c_r divided by 2 w_r (`../../mcscf/zvector/README.md`), so
  z_c,r multiplies the condition 2 w_r P (H₀ − E_r) c_r. That is the
  Davidson residual, which is orthogonal to every root (Galerkin), so P
  drops nothing. The spin penalty adds nothing to z_c · r: z_c is in the
  roots' spin and λ (S² − S(S+1)) c is not. √n_roots turns the largest
  residual into a bound on the stacked one.
- **Θ needs no term.** Its condition c_s · H₀ c_r = c_s · r_r is zero by
  Rayleigh–Ritz (r_r is orthogonal to the subspace c_s lies in), at any
  weights.
- **z^c needs none either.** The core and virtual canonicalization over the
  SA Fock is a direct diagonalization, its off-diagonal residual round-off.

Both sensitivities are invariant to turning u_P and u_Q within their pair:
(z^Q − z^P)/2 and z^PQ turn as the pair (x/2, h) does, by twice the angle,
and each norm is Euclidean.

## A MECI inside a degenerate CASSCF pair (#902)

**Theory only; nothing below is implemented.** #847's all-electron
QD-PC/aug-cc-pVDZ walk to ethylene's C3v MECI
(`docs/reproduction/park2019/tier1_ethylene_c3v_augdz.toml`) stops at
evaluation 10. The QD pair there is healthy (H_eff gap 6.6e-10), but the
CASSCF ¹E pair under it is 7.9e-9 Eh apart, and
`mcscf::root_pair_multipliers` refuses it (`kRootGapFloor`). #892 showed
that the refusal is right for state gradients
(`../gradient/README.md`, "At a degenerate CASSCF pair"):

- SC's g_K diverge as 1/gap;
- PC's g_K are bounded, but their limit depends on the direction of
  approach;
- only PC's τ_P + τ_Q vanishes from both directions measured (≤ 5e-12), so
  only ḡ has a limit.

This section asks what a MECI walk needs there, and whether that much
exists. In short: **a PC MECI at a symmetry-pinned reference degeneracy is
well posed, and a walk needs only ḡ there.** ḡ has a limit, by an identity
and a measurement below. The branching plane has no limit, but at
the MECI the walk does not need it. SC has no MECI there to find.

### What the walk consumes

`geometry::optimize_meci` (`../../geometry/README.md`, "Minimum-energy
crossings") reads, at each evaluation:

1. the two energies, so the gap;
2. x = g_Q − g_P: its direction x̂, which spans the branching plane, and its
   length, which scales the gap's Newton step −gap/‖x‖ x̂;
3. h_PQ's direction ŷ, the plane's second vector, after x̂;
4. ḡ = (g_P + g_Q)/2, through P ḡ, P the projector off the rigid motions
   and the plane. P ḡ drives the BFGS step, the BFGS update and the force
   criteria.

The trust-radius model also reads ḡ · p and x · p. Nothing reads g_P or
g_Q alone. Items 2 and 3 enter only through P and the gap step, and item 4
is the only gradient the convergence test reads.

### Which of them exist

Write each QD energy as a function of the geometry and of the references'
rotation within the CASSCF pair, E_K(R) = Ẽ_K(R, θ(R)). Ẽ_K is smooth in
both arguments. θ(R) is the angle the CASSCF eigenvectors take, smooth away
from the reference seam but, at the seam, a function of the direction in
the reference branching plane alone: θ turns by half the polar angle φ in
that plane (the reference pair's geometric phase). So
dE_K/dR = ∂_R Ẽ_K + τ_K ∂_R θ, and ∂_R θ goes as 1/ρ, ρ the distance to
the reference seam. That is the ζ_K t term, t the pair's transition
gradient. g_K has a limit only if τ_K = ∂Ẽ_K/∂θ is O(ρ), and a unique one
only if τ_K/ρ is independent of φ as well.

**Rotating the references by π/2 relabels them**, (c₀, c₁) → (c₁, −c₀).
H_eff is built per reference, so it comes back permuted and its spectrum
is unchanged: each Ẽ_K has period π/2 in θ. One reference's own diagonal
element is a function of that reference alone,
f(θ) = H_eff[0, 0](θ) − E^(0)(θ), its state-specific (SC or PC) NEVPT2
correction. c and −c are the same state, so f has period π and harmonics
cos 2nθ, sin 2nθ. With three references,

    E_P + E_Q = tr H_eff − E_2
              = Σ_r E_r + f(θ) + f(θ + π/2) + C_22 − E_2(θ)

(the zeroth order's trace is fixed). **f(θ) + f(θ + π/2) cancels every odd
n**, and E_2 moves with θ only through the pair's coupling to the third
state, second order over the third-state margin (0.043 Eh here). So ḡ's
τ carries only f's even harmonics. Each E_K has period π/2 as well, so
only even harmonics, but it gets them by mixing f's n = 1 with the
off-diagonal elements, not from f's own n = 2. That is why one E_K moves
with θ while the pair's sum does not.

What C3v allows at first order in a displacement: C3 turns the E pair by
2π/3, so (cos 2nθ, sin 2nθ) transforms as E for n = 1 and 2, and as A for
n = 3. An e displacement can drive both n = 1 and n = 2 at first order.
**Symmetry does not forbid τ_P + τ_Q = O(ρ).** ḡ's limit rests on f's
n = 2 harmonic being negligible, and that is a measurement, not an
identity. The toy below measures it. **For PC, f's n = 2 is second order
in the displacement**: 2.5e-13 at 1e-3 Å and 2.4e-15 at 1e-4 Å along d₂,
against an n = 1 of 7.9e-8 and 7.9e-9. So τ_P + τ_Q = O(ρ²), the pair's
ζ = O(ρ) vanishes, and **ḡ has one limit from every direction**. The
first-order symmetry allowance is unused. For SC, f's n = 2 is O(1),
1e-6 at every distance, and survives in the trace, so SC's ḡ diverges
as 1/gap, like its g_K.

**The covariant part.** Write H_eff's pair block as a covariant part, one
that turns with the references as a matrix, plus the rest. The covariant
part gives τ(W) = tr([W, H_eff] J) for any W, J the rotation's generator,
which is zero for W = u_P u_Pᵀ + u_Q u_Qᵀ: that is H_eff's own spectral
projector, so it commutes with H_eff. ḡ's ζ therefore comes only from the
non-covariance, which is the harmonic argument above. For W^PQ the
covariant part gives τ = ±(E_Q − E_P), so

    ζ(W^PQ) = ∓ (E_Q − E_P) / (E₁ − E₀),

a ratio of two cone gaps. It is bounded, but it depends on the direction,
and it multiplies t, which near the reference seam is the reference pair's
own coupling vector. Its size is that of h_PQ itself. This is the
references' frame turning, and it is no artifact: dropping it would give an
O(1)-wrong h_PQ. The same holds for x. **So the plane cannot be had below
the floor by dropping ζ.** Away from the floor the plane is
span{x, h_PQ}, and both carry this term in full.

**Neither x nor h_PQ has a limit; the plane has none either.** In a smooth
(diabatic) reference frame, the covariant part of the QD pair's traceless
block is linear in the displacement δ. The non-covariant part adds
M(θ(δ̂)) δ, homogeneous of degree 1 but not linear, because θ depends on
δ's direction in the reference plane. #892's PC ζ_K, 0.011 along d₁ and
under 4e-4 along d₂, measures M directly. A covariant H_eff has τ_K = 0
identically, since eigenvalues do not move under a change of basis. So
the QD gap near the reference seam is |L(δ)|, with L homogeneous of degree
1 but not linear. Its zero set is a cone, not a plane, and the
"branching plane" depends on the direction of approach. The toy's E_P
moves with θ by 2.4% of the QD gap along d₁, and by 0.9% along d₂: the
cone's slope depends on the direction, by a few percent here.

| quantity the walk reads | limit at a PC reference-and-QD degeneracy |
|---|---|
| E_P, E_Q, the gap | yes (energies only; PC's pair stays degenerate at the symmetric point to 5e-11) |
| ḡ, hence P ḡ given a plane | yes: its ζ is O(ρ), since f's n = 2 is O(δ²) (the toy) |
| x̂, ŷ (the plane) | no: the cone's slope depends on the direction, by a few percent here |
| ‖x‖ (the gap step's scale) | no, as at any cone, and also through M |
| g_P, g_Q, h_PQ alone | no |

### Is it an isolated MECI?

The count is per point group, since a QD degeneracy only coincides with a
reference degeneracy by symmetry. Without symmetry, the QD seam and the
reference seam are each of codimension 2, so they meet in codimension 4:
8 of 3N − 6 = 12 dimensions for ethylene. The QD seam's Ē has no reason
to be minimal on that 8-dimensional set. A walk can pass the floor on the
way there, but its MECI is not inside it.

**At C3v both pairs are E pairs.** H₃C–CH with the lone H on the axis has
4 a₁ + 4 e vibrations, as CH₃CN does. The C3v manifold M (fixed symmetry
elements, a linear subspace in Cartesians) is 4-dimensional, and on all of
it both the CASSCF pair and PC's QD pair are degenerate by symmetry. Off
M, each pair is split linearly by the e displacements. In Jahn–Teller
form, each splitting is |Σ_k κ_k (Q_kx, Q_ky)| over the four e modes, with
its own κ: κ^ref for the CASSCF pair, κ^QD for the QD one. Each seam is of
codimension 2 (10-dimensional) and contains M. The two seams' branching
planes, span{Σ κ_k e_kx, Σ κ_k e_ky}, are different planes in the same
8-dimensional e space, so the seams meet in 8 dimensions, a set that
contains M. The QD seam's branching plane lies in the e space, so its
6 remaining e directions lie along the seam.

The MECI conditions on the QD seam are 2 (the gap) + 10 (P ḡ = 0). On M:

- the gap's 2 hold identically;
- ḡ is a₁ (Ē is invariant), so its components along the 6 seam e
  directions vanish identically;
- what remains is the 4 a₁ components of ḡ, against the 4 coordinates
  of M.

**4 equations in 4 unknowns: the C3v MECI is isolated**, a stationary
point of Ē on M, and a stationary point of Ē on the whole QD seam by
symmetry. It is a minimum if Ē's e-e curvature is positive on the seam's
6 e directions, which no measurement here addresses. Park's search
reports one. The reference degeneracy adds no condition. It is a property
of every point of M, not of the MECI, so a walk that reaches M stays below
the floor for the rest of its steps along M's a₁ directions. That is
#847's walk: it met the gap criterion (6.6e-10 < `max_gap` 1e-7) by
evaluation 10, and its remaining work was all in the a₁ forces.

The seam near M is a cone, not a plane (above), so "the QD branching
plane at the MECI" has no meaning, but the seam itself, the zero set of
|L|, is still a closed set through M. The minimum of the smooth Ē over it
is still well posed.

**SC has none of this.** Its energies depend on θ at the symmetric point
itself (E₀, E₁ over θ by 1.1e-6 and 4.4e-6 at Park's geometry; at the
exact-C3v point its QD gap runs 4.1e-7 to 1.2e-5 with θ), so its QD pair
is not degenerate on M. Its "seam" is not a function of geometry there.
There is no SC MECI at a reference E pair, and the refusal stays for SC
whatever the walk.

### A PC-only algorithm

The observation that makes it work: **on M, P ḡ does not depend on the
plane, as long as the plane lies in the e space.** ḡ is a₁ there, and the
e space is orthogonal to the a₁ space (the symmetry operations are
orthogonal). Removing any two e vectors from ḡ removes nothing, so P ḡ = ḡ
minus its rigid motions (T_z is a₁) for every such plane. M is a linear
subspace in Cartesians with fixed symmetry elements, and a₁ steps keep the
elements fixed. So the e space is the same at every point of M, and a
plane taken anywhere near M stays in it. Off M by δ, the error a wrong
plane makes in P ḡ is about the plane's tilt times ḡ's e part, which is
itself O(δ). That is second order, and zero at the MECI. So below the
floor the walk needs a fresh ḡ, and only a plane from nearby.

1. **ḡ below the floor.** In `qd_pc_intersection`, when the smallest
   CASSCF gap is under `kRootGapFloor` and that pair lies under the QD pair
   (the two references carry most of u_P and u_Q), compute one W-form
   column, W = u_P u_Pᵀ + u_Q u_Qᵀ, with the pair's ζ set to its limit, 0.
   This needs a mode in `mcscf::root_pair_multipliers` that, for a named
   equal-weight pair, returns ζ = 0 instead of refusing. It should still
   report τ(W), and refuse when |τ(W)| is above a gate, say 1e-11, well
   above the tree's τ round-off (8.9e-16, "What is bitwise" in
   `../../mcscf/gradient/README.md`) and below #892's τ_K. The gate cannot
   prove that ζ's limit is 0, since τ/gap is 0/0 whatever the gate. But it
   catches the cases the argument does not cover: SC, a W outside the
   pair, or a pair that is not the QD pair's. One column instead of three,
   so cheaper than today's call. g_P, g_Q and h_PQ are not returned. The
   result carries ḡ and a `degenerate_reference` flag.
2. **The plane is carried.** `geometry::optimize_meci`'s callback gains a
   way to return ḡ alone. Then the evaluation reuses the last evaluation's
   x̂, ŷ and ‖x‖, the last one taken above the floor, and recomputes only
   the rigid-motion basis at the new geometry. P is then the same before
   and after a carried step, so the BFGS update learns Ē's curvature only,
   not the plane's turn. Near M the plane does not turn in the sense that
   matters (any e plane gives the same P ḡ on M). A walk whose first
   evaluation is already below the floor has no plane to carry, and is
   refused (`Numerical`, "start off the symmetric point"). Park's start
   is 2.1e-6 Eh above it.
3. **The gap step is kept, on the carried x.** Below the reference floor,
   the QD gap is of the same order (the walk's 6.6e-10 against 7.9e-9), so
   −gap/‖x‖ x̂ moves the geometry by about 1e-8 bohr. A wrong x̂ within the
   e space costs a slower contraction toward M, never the fixed point.
4. **What the step reports.** `optimize.json` gains, per evaluation,
   `plane_carried` and the gate's τ. At the head, `g_p`, `g_q` and `h_pq`
   become null when the last evaluation was carried, and a `g_mean` row
   is added. Readers of the head's branching space must take the plane as
   the last carried one, or recompute it at a displaced point.

Everything else is unchanged: above the floor, every evaluation is today's.
Every existing MECI golden runs above the floor, so none can move. The
refusal stays for state gradients (`[nevpt2_gradient]` with a `state`), for
SC, and for a degenerate CASSCF pair that is not the QD pair's.

### What would validate it

**Directional FD of ḡ at a near-C3v geometry, at #853's bar.** Take the
exact-C3v symmetrization of Park's aug-cc-pVDZ geometry (#892's), displaced
5e-6 Å along d₁ (CASSCF gap about 2.5e-9, under the floor; #892's line has
gap ≈ 4.9e-4 Eh/Å × |δ|), and the exact point itself (gap 1.2e-10).
Compare ḡ · d from item 1 against differences of Ē = (E_P + E_Q)/2 from
QD-PC single points (energies only, which refuse nothing). Every point is
re-converged, with steps 2e-3, 1e-3 and 5e-4 bohr, Richardson-extrapolated.
The bar is max(2 × the Richardson bar, 3 σ_E / h_min), σ_E from 5 repeats
(`../gradient/README.md`, #853's method). Directions:

- ḡ's own (an a₁ direction, the one the walk uses);
- two random unit directions, `default_rng(902)`;
- two pure e directions: d₁'s e part and a random one's, a₁ part
  projected out by the C3v symmetrizer.

Central differences cannot see the cone term. Ẽ's period π/2 in θ is
period π in φ, so a ρ c(φ) term in Ē is even under δ → −δ, and the central
difference cancels it. So along each e direction, also compare the
**forward and backward one-sided** Richardson slopes with ḡ · d and with
each other. A surviving n = 2 harmonic shows as a forward/backward split
of 2 c(φ). The pass is every miss under the bar, one-sided included. The
gate's τ(W), reported by item 1, must be under the gate at both points.

**Then the walk.** `tier1_ethylene_c3v_augdz.toml` converging at its own
criteria (gap 1e-7, Opt=Tight), with `plane_carried` from the evaluation
where the CASSCF gap crosses the floor. Its final geometry's C3v
deviations (C–H spread, H–C–C spread, the lone H off the axis) should fall
toward the displacement criteria, a check that the walk found M's
stationary point rather than a point merely near M. Every existing MECI
golden must be unchanged.

### The toy

`tools/emit/qd_nevpt2_reference.py`'s `sc_qd` and `pc_qd`, over #892's
rotation (PR #901's `issue892_rotation.py`: DF-SA3-CASSCF(2,2)/aug-cc-pVDZ,
aug-cc-pVDZ-JKFIT, SA-canonical inactive orbitals). It is QD-PC's own H_eff,
not a CASSCF stand-in, and it reproduced the tree's evaluation 0 in #892,
so its numbers map onto the walk directly. Energies only: τ is
d/dθ of what it prints. At the exact-C3v symmetrization of Park's
geometry plus |δ| along d₁ (Park's offset) or d₂ (`default_rng(892)`,
#892's), the references are rotated by θ at 12 points in [0, π). Each
column is the amplitude of cos 2nθ (n = 1, 2) in

- f, reference 0's own diagonal correction;
- E_P + E_Q;
- E_P alone.

Peak-to-peak over θ is in parentheses.

| point | CASSCF gap | QD gap | f: n = 1 | f: n = 2 | E_P + E_Q: n = 2 (ptp) | E_P: n = 2 |
|---|---|---|---|---|---|---|
| PC, d₁, 1e-3 Å | 4.86e-7 | 1.95e-7 | 1.5e-7 | 1.9e-14 | 3.2e-14 (1.4e-13) | 4.7e-9 |
| PC, d₁, 1e-4 Å | 4.93e-8 | 2.06e-8 | 1.5e-8 | 8e-25 | 1.3e-14 (1.1e-13) | 4.7e-10 |
| PC, d₂, 1e-3 Å | 4.41e-6 | 4.56e-6 | 7.9e-8 | 2.5e-13 | 5.1e-13 (1.1e-12) | 4.2e-8 |
| PC, d₂, 1e-4 Å | 4.41e-7 | 4.56e-7 | 7.9e-9 | 2.4e-15 | 2.2e-14 (1.1e-13) | 4.2e-9 |
| SC, d₁, 1e-3 Å | 4.86e-7 | 7.17e-7 | 2.2e-8 | 1.1e-6 | 2.1e-6 (4.2e-6) | 4.8e-7 |
| SC, d₁, 1e-4 Å | 4.93e-8 | 6.46e-7 | 1.6e-7 | 1.1e-6 | 2.2e-6 (4.2e-6) | 4.7e-7 |
| SC, d₂, 1e-3 Å | 4.41e-6 | 9.17e-6 | 2.4e-7 | 3.2e-7 | 6.3e-7 (1.3e-6) | 1.6e-6 |
| SC, d₂, 1e-4 Å | 4.41e-7 | 7.93e-6 | 2.1e-7 | 3.3e-7 | 6.5e-7 (1.3e-6) | 1.6e-6 |

Every n = 1 of E_P + E_Q and of f + f(θ + π/2) is ≤ 5e-14, as the identity
requires. E_2 moves by ≤ 9e-14 for PC. In the PC rows:

- f's n = 1 is linear in |δ|. That is the reference pair's first-order
  non-covariance, the term the symmetry argument allows.
- E_P's n = 2 is linear in |δ| as well. That is #892's τ_K = O(ρ), and
  each g_K is bounded but direction-dependent.
- f's n = 2 is quadratic, or at round-off, and E_P + E_Q is flat to
  1e-12 or better.

In the SC rows, nothing shrinks with |δ|: its n = 2 is O(1), in f and in
the trace alike. That is SC's non-invariance at the symmetric point
itself, so SC's ḡ, like its g_K, has no limit.

The toy is not kept under `tools/`. It is a θ scan over #901's helpers,
which are not in the tree either, and its one conclusion is recorded
here. The script is in #902's PR.

## Tests

`test/gtest/nevpt2_qd_intersection_gpu.cpp`, SC and PC, on this tree's own
DF-SA-CASSCF of `test/gtest/data/qd_nevpt2/`'s cases at #696's bars:

- `Nevpt2QdIntersectionWForms.MatchesWForms/<case>_<sc|pc>`: g_P, g_Q and
  h_PQ each against its own W-form call at the result's own u's, every pair
  of `h2co_skew_ccpvdz_sa3` and `h2co_skew_ccpvdz_sa3_weighted`, to 1e-9 of
  the largest summand of the derivative passes; s_o and s_c against their
  definitions on those calls' multipliers, to 1e-9 of the largest norm
  summed. Independently converged Z-vectors, hence 1e-9 (#784's bar
  amendment); PC's s_c alone at 1e-7, because PC's CI adjoint ȳ is
  reproducible between identical calls only to ~5e-8 (#729), and z_c is
  solved from it: two runs' s_c measured up to 4.8e-9 apart, relative. The
  check is for a wrong pairing factor (2 or 2 w_r), which is O(1)
  ([#784's ruling](https://github.com/whatDiracIsCooking/FreeBreakfast/issues/784#issuecomment-6616656181)).
- In both of the above, the shared path (#787): one several-W call at the
  same u's against the three W-form calls, g_P, g_Q and h_PQ to 1e-9 of
  the larger side's largest summand, s_o and s_c to 1e-9 of the largest
  norm summed (PC's s_c 1e-7). `MatchesWForms` also runs the intersection's
  non-default path once per case, pair (0, 1), against the same calls.
- `Nevpt2QdIntersection.DegeneratePair`: `h2co_ccpvdz_sa2` (C2v, roots 1A1
  and 1A2) made exactly degenerate at a consistent solution: h + λ v vᵀ,
  v = S C e_k on the active orbital the roots' occupations differ most in,
  the DF-SA-CASSCF re-converged at #696's bars for each λ, λ by secant
  until H_eff's diagonal agrees to 1e-11. There `qd_*_coupling` refuses,
  and the intersection returns and matches the W forms; s_o and s_c are
  unchanged when u_P, u_Q are turned within the pair, against separate
  W-form calls (1e-9; PC's s_c 1e-7) and by linearity on one set of
  multipliers (1e-10). #784 named `Nevpt2QdNac.Refuses`' setup instead, one CASSCF root energy
  moved by hand; there the roots are no longer eigenvectors at their
  energies, and every W form's own gate refuses its Lagrangian (SC
  active-active asymmetry 2.4e-1), so it cannot carry a W-form comparison.
- `Nevpt2QdIntersectionBound.HoldsForAStoppedSolve/<sc|pc>`: a DF-SA-CASSCF
  of `h2co_skew_ccpvdz_sa3` from the converged orbitals turned by a fixed κ,
  stopped at `conv_tol_grad` = 1e-5 and canonicalized; every pair's gap
  moves by at most s_o ‖g‖ + s_c √n_roots ρ, the stopped solve's own ‖g‖
  and ρ, s_o and s_c the converged point's.
- `Nevpt2QdIntersection.ZvectorTolerance`: `h2co_ccpvdz_sa2`'s pair
  (0, 1), SC, at `options.zvector_tolerance` 1e-12. The intersection
  succeeds and g_P, g_Q and h_PQ agree with the default call's to 1e-9 of
  the W form's largest summand. The W form at u_P u_Pᵀ with the same
  trailing tolerance reports a true residual of at most 1e-12. A tolerance
  of 0 or below is `InvalidConfig`.
- `Nevpt2QdIntersection.Psb3TrueResidual` (slow tier): PSB3's
  DF-SA3-CASSCF(6,6)/cc-pVDZ with C13=C14 twisted 90° (root gap 2.9e-2 Eh).
  The SC intersection of QD states 0 and 1, and the SC state gradient of
  state 0 (one column, whose true residual is now checked too), both
  succeed: every true residual is at most 1e-10. Before the CI response
  inside H̃ was converged to `mcscf::response_tolerance_for` the Z-vector
  tolerance, the intersection was refused at 2.55e-10 after 31 products
  (`../../mcscf/intersection/` "The floor under the true residual"). Now
  the three columns end at their recurrence's 9.8e-11, 6.0e-11 and
  4.2e-11.
- `Nevpt2QdIntersection.Refuses`: P = Q, a state outside the average, and a
  class PC refuses (PC's error, as `qd_pc_coupling`'s).
- `Nevpt2QdIntersection.Timing`: the table above; a measurement, skipped
  unless `FBF_QD_INTERSECTION_TIMING` is set.

## Files

| File | Role |
|------|------|
| `intersection.cppm` | Module interface: `QdIntersectionOptions`, `QdIntersection`, `qd_sc_intersection`, `qd_pc_intersection` |
| `intersection.cpp` | H_eff and R, the shared or three W-form calls, the residual check, s_o and s_c |
| `CMakeLists.txt` | Build configuration |
