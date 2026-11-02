# Gradient

The analytic nuclear gradient of a converged DF-CASSCF on the GPU, of its one
state or of any root of a state average: `mcscf::df_gradient`, every term
returned on its own,

    dE/dR = Σ D dT + Σ D dV − Σ W dS + dE₂ + dV_nn

## Module

`free_breakfast.mcscf.gradient`, target `free_breakfast::mcscf::gradient`,
re-exported by `free_breakfast.mcscf`.

```cpp
Result<mcscf::GradientTerms> mcscf::df_gradient(
    DeviceResources &res, const Molecule &molecule, const BasisSet &basis,
    std::span<const PrimitivePairClass> classes, const BasisSet &aux,
    const SphericalMatrixAO &h, const Buffers::device<double> &b, const double *l,
    const mcscf::Solution &solution, int32_t ncore, int32_t ncas, int32_t root,
    const mcscf::ZVector *multipliers = nullptr, const mcscf::MoAdjoint *extra = nullptr);
Result<mcscf::GradientTerms> mcscf::df_transition_gradient(
    /* df_gradient's arguments through ncas */, std::span<const double> gamma,
    std::span<const double> gamma2, const mcscf::ZVector *multipliers = nullptr,
    const mcscf::MoAdjoint *extra = nullptr);
// g->kinetic, g->potential, g->overlap, g->two_electron,
// g->nuclear_repulsion, g->total: each n_atom x 3, row-major, element 3 A + k
```

`solution` is `mcscf::solve`'s under `DensityFitting::Full` over the same B;
`aux`, B and L are `pipeline.eri`'s `DensityFitted` (`aux`, `b`, `l`), `h` the
core Hamiltonian and `classes` `primitive_pair_classes` of the basis, as
`gradient::df_rhf` takes them. The inputs are explicit, not pipeline types,
so this layer stays below `pipeline`.

**`root` picks the state.** For one root it is 0 and the terms below are
taken at the solution's RDMs. For a state average they are taken at root I's
**Lagrangian**, below: the same terms, at other densities, plus their
response to the orbitals. The weights are not in a `Solution`, so the call
recovers them from it, as the least-squares solution of
Σ_r w_r γ_r = γ with Σ_r w_r = 1 over `root_gammas` and the weighted γ
(`Unsupported` when the roots' 1-RDMs are linearly dependent, `InvalidConfig`
when γ is not their average to 1e-10). That recovery is exported as
`mcscf::sa_gradient_weights`, for `df_nac` at unequal weights. The active electron count, for the CI
strings, is trace(γ).

**The formula is the DF-CASSCF energy's gradient, and only at convergence.**
W stands in for the orbitals' response (stationarity in the orbitals), and the
CI is variational. A solution from `Direct` or `ApproxHessian` converges to
the exact CASSCF, whose gradient this is not; nothing in a `Solution` says
which mode made it, so the call cannot refuse one.

### Unequal weights: the rotations among the roots

At unequal weights the rotations among the averaged roots are parameters of
the SA energy and carry multipliers of their own, Θ (`zvector`'s
`ZVector::theta`, [`../zvector/README.md`](../zvector/README.md) "The
rotations among the roots"); pairs of equal weight stay projected. The call
folds Θ into z_c as its part along the roots (below), so any number of roots
at any weights, partly equal included, is one formula. Projected out, as
before #613, the gradient was off by up to 3.3e-3 at three roots
(0.5, 0.3, 0.2), 7% of a root's gradient (`casscf_hessian.tex` §5.2; the
fixture `test/gtest/data/casscf_hessian/` records it as
`gradient_projected_vs_fd`); for two roots Θ carries nothing (5e-13 there).
PySCF's analytic SA gradient refuses unequal weights.

## The terms

C's columns are core (i, j), active (t, u, v, w), virtual; γ and Γ the
solution's 1- and 2-RDM, E = Σ h_pq D_pq + ½ Σ Γ_pqrs (pq|rs) over MOs.

- **One-electron.** D = 2 C_c C_cᵀ + C_a γ C_aᵀ (a gemm and a `gemmwich`),
  against `kinetic_gradient` and `potential_gradient`; and −Σ W S′ with
  W = C sym(F) Cᵀ, F `response::fock`'s generalized Fock over the fitted
  integrals (F_pq = Σ_r h_pr D_rq + Σ (pr|st) Γ_qrst). The symmetric
  connection C(R) = C (1 − ½ S′_MO) puts −½ S′_pq into U_pq, and
  ∂E/∂U_pq = 2 F_pq, so the term is −Σ F_pq S′_pq; S′ is symmetric, so only
  F's symmetric part counts (F is symmetric only to the solve's convergence).
  These four go through `gradient::one_electron_terms`, exported from
  `free_breakfast.gradient` at any D and −W, which the RHF gradients there
  call too.
- **Two-electron.** Over the fit, (pq|rs) = Σ_PQ (pq|P) M⁻¹_PQ (Q|rs), M = L Lᵀ
  the metric, so any E₂ = ½ Σ Γᴬᴼ_mnls T_mnP M⁻¹ T_lsQ differentiates to
  Σ Γ_mnP T′_mnP + Σ W_PQ M′_PQ. All of E₂ goes into **one** Γ_mnP (in
  `three_center`'s layout, P fastest) and **one** W_PQ, then one
  `three_center_gradient` and one `metric_gradient`: the derivative
  integrals are evaluated once, whatever the number of pieces. E₂ splits as

      E₂ = ½ J[D]·D − ½ J[D_a]·D_a − Σ_i Σ_pq G_pq (ip|iq) + ½ Σ Γ_tuvw (tu|vw),

  D_a = C_a γ C_aᵀ, G = diag(1_core, γ) over the occupied orbitals:

  - **J** (core–core and core–active; the active–active J is in Γ's term).
    ½ J[D_c]·D_c + J[D_c]·D_a = ½ J[D]·D − ½ J[D_a]·D_a. Each density X fits
    to d_X = M⁻¹ T X = L⁻ᵀ (B vec X) (a gemv and a trsv), and
    ½ J[X]·X differentiates as `df_rhf`'s Coulomb, Γ = X ⊗ d_X and
    W = −½ d_X d_Xᵀ: so Γᴶ = D ⊗ d − D_a ⊗ d_a and
    Wᴶ = −½ d dᵀ + ½ d_a d_aᵀ, four rank-one gemms.
  - **K** of the core with itself and with the active space.
    −¼ K[D_c]·D_c − ½ K[D_c]·D_a = −Σ_ij (ij|ij) − Σ_i,tu γ_tu (it|iu)
    = −Σ_i Σ_pq G_pq (ip|iq). With the fitted pairs c_P,ip = (L⁻ᵀ B_ip)_P,
    i core and p occupied (`b_mo`, then a trsm), and V_P,iq = Σ_p c_P,ip G_pq
    (a gemm), Γᴷ_mnP = −2 Σ_i C_mi Σ_q V_P,iq C_nq (a gemm over q and a
    strided-batched gemm over i, as `df_rhf`'s exchange) and
    Wᴷ_PQ = Σ_iq V_P,iq c_Q,iq (a gemm). The core–core block of G is `df_rhf`'s
    exchange exactly; the core–active block is the same MO form with γ in it.
  - **Active–active**, ½ Σ Γ_tuvw (tu|vw). With c_P,tu the fitted active pairs
    and Z_P,vw = Σ_tu c_P,tu Γ_tuvw (one gemm against Γ as the ncas² square
    matrix it is in `response::Fock::gamma2`), Γᵃᵃ_mnP = Σ_vw C_mv C_nw Z_P,vw
    (a gemm and a strided-batched gemm) and Wᵃᵃ = −½ Z cᵀ (a gemm).

  `two_electron` is their sum; PySCF's `df.grad.casscf` reports the same
  sum (J, K, the 2-RDM term and the auxiliary basis's response) as one part.
- **Nuclear repulsion**: `nuclear_repulsion_gradient`, through the shared
  one-electron call.

## The state-averaged Lagrangian

Root I's energy is not stationary in the state-averaged orbitals, so its
gradient is that of the Lagrangian

    L = E_I + z_o · g_o + Σ_r z_c,r · g_c,r,

g_o the SA orbital gradient over the packed pairs and g_c,r = 2 w_r (H₀ − E_r)
c_r the CI's, with z = (z_o, z_c) the Z-vector of
[`../zvector/`](../zvector/README.md): A z = −(g_I, 0), A the SA Hessian. Its
derivative at fixed orbitals (up to the connection below) and CI is the
gradient. The call builds what that needs at the solution's orbitals and
roots: the CI strings, sigma (H₀) and `ci::Rdm`, `response::Hessian` over the
SA Fock, `RelaxedHessian`, root I's RDMs (`rdm12` at weight 1 on root I), g_I
(`response::fock` at them) and `zvector`. Then L's derivative is three pieces,
each one of the terms above at some densities:

- **E_I**: the state-specific formula at root I's γ_I and Γ_I.
- **z_c**: Σ_r z_c,r · 2 w_r (∂H) c_r is the derivative of the energy
  functional Σ_r w_r (⟨z_c,r|H|c_r⟩ + ⟨c_r|H|z_c,r⟩) at the transition RDMs
  γ_T and Γ_T: half the difference of `rdm12` of c + z_c and of c − z_c at the
  weights, the ± polarization `mcscf/solve` takes them by. The rotations
  are z_c's part along the roots, z_c,r + Σ_s Θ_sr c_s (one gemm of C by Θ's
  antisymmetric matrix), whose transition RDMs weigh each pair by
  w_r − w_s: the derivative along Θ. Their overlap is
  zero, so they carry no core energy, and the energy is affine in the active
  RDMs with the core part constant: so this piece and E_I's are the terms
  once, at **γ_I + γ_T and Γ_I + Γ_T**, with F their generalized Fock.
- **z_o**: z_o · g_o(x) is the derivative of E_SA along C → C (1 + ε κ),
  κ = κ(z_o) (`unpack_orbital_pairs`), so z_o · ∂g_o is that derivative of
  the gradient formula itself, at the SA RDMs: every term with each of its
  C's in turn replaced by C̃ = C κ, the product rule. Below.

### The orbital response

Every term is a polynomial in C, so its ε-derivative is the product rule.
C̃ sits beside C in one nao × 2nmo matrix [C | C̃] so that `b_mo` transforms
over either.

- **One-electron.** D_L = D_cL + D_aL, D_cL = 2 (C̃_c C_cᵀ + C_c C̃_cᵀ) and
  D_aL = C̃_a γ C_aᵀ + C_a γ C̃_aᵀ, added to D.
- **Overlap.** W = C sym(F) Cᵀ with F = Cᵀ X(C), so W = sym(C Cᵀ X Cᵀ), and
  C Cᵀ = S⁻¹ is stationary to first order (κ is antisymmetric). So only F's
  inner orbitals move, and W's response is C sym(F_L) Cᵀ:

      F_L = F κᵀ + [(Fᴵ + Fᴬ) κ + V_cL + V_aL] γ_c + [Fᴵ κ + V_cL] γ_a + Fᵃᵃ_L,

  γ_c = 2 on the core and γ_a = γ on the active block (each placing its
  bracket's columns), V_XL = Cᵀ (J − ½ K)[D_XL] C (one `df_coulomb` and one
  two-factor `df_exchange` each, K[D_cL] = 2 (K[C_c C̃_cᵀ] + its transpose)),
  and Fᵃᵃ_L the active 2-RDM term Σ_uvw (pu|vw) Γ_tuvw with u, v, w rotated:
  Σ_svw (ps|vw) R1_svw,t + Σ_suw (pu|sw) R2_suw,t, R1 = κ_act G1 and
  R2 = κ_act G2, G1[u, (v, w, t)] = Γ_tuvw and G2[v, (u, w, t)] =
  Γ_tuvw + Γ_tuwv (permuted on the host), so that each R lies as ppaa's and
  papa's columns do: four gemms. F + F_L replaces F.
- **J.** Of ½ J[D]·D − ½ J[D_a]·D_a at the SA D and D_a: Γᴶ gains
  D_L ⊗ d + D ⊗ d_L − D_aL ⊗ d_a − D_a ⊗ d_aL and Wᴶ gains
  −½ (d_L dᵀ + d d_Lᵀ) + ½ (d_aL d_aᵀ + d_a d_aLᵀ), eight rank-one gemms.
- **K.** With c′_P,ip = c_P,ĩp + c_P,ip̃ (ĩ C̃'s column i; two `b_mo`s over
  [C | C̃], then the trsm) and V_a = c G_a at the SA G: V gains c′ G_a,
  W gains V_a c′ᵀ, and Γᴷ gains −2 Σ_i (C_mi Σ_q V_a,iq C̃_nq +
  C̃_mi Σ_q V_a,iq C_nq) (one more gemm into Y, and one more Y and
  strided-batched gemm over C̃_core).
- **Active–active.** The same with c′ of the active pairs and Z_a = c Γ_a:
  Z gains c′ Γ_a, W gains −½ Z_a c′ᵀ, and Γᵃᵃ gains
  Σ_tu (C_mt C̃_nu + C̃_mt C_nu) Z_a,P,tu.

These are the non-symmetric pairs (D_L ≠ D_R) of the papers' response
densities. They are not factored: they are only more terms in the same Γ_mnP
and W_PQ, so the derivative integrals are still evaluated once,
one `three_center_gradient` and one `metric_gradient` per call.

## A caller's Lagrangian

The last two arguments open the call up to a Lagrangian that is not the
CASSCF's own (NEVPT2's, `nuclear_gradient.tex` Eq. total). Both default to
null, which is the call above unchanged.

**`multipliers`** are (z_o, z_c, Θ) in place of the Z-vector the call would
solve (Θ may be empty, for none), as a `ZVector` (`mcscf::zvector`'s result, or one the caller built).
With them one root is taken at the same Lagrangian as a state average's root,
at weight 1, so a single root with nonzero multipliers gets the transition
RDMs and the orbital response too. **They are in `zvector`'s coordinates**:
z_o over the packed pairs (`response::pack_orbital_pairs`' order,
ncore (nmo − ncore) + ncas (nmo − nocc) doubles), and z_c dim × nroots, root
r's CI row (H₀ − E_r) z_c,r + P H′(z_o) c_r = −y_r, the PySCF `gen_g_hop`
convention, **unweighted** — not the tex's J_cc = 2P(H − E)P. The call
weights root r's transition RDMs by w_r itself, as it does for its own
Z-vector; a caller whose CI multiplier is in another convention rescales it
before the call. Wrong lengths are `InvalidConfig`.

**`extra`** is an MO-basis adjoint (`MoAdjoint`, device pointers, nmo = nao)
of a term Φ(h, B) added to the Lagrangian: h̄ = ∂Φ/∂h (nmo²), Λᴸ = ∂Φ/∂Bᴸ
(naux × nmo², (L, p + nmo q) at L + naux (p + nmo q), as `b_mo` leaves B),
both symmetric in (p, q), and X = 2 h h̄ + 2 Σ_L Bᴸ Λᴸ (nmo², X_xp at
x + nmo p), read only through sym(X). Here Bᴸ is the fitted pair of
(pq|rs) = Σ_L Bᴸ_pq Bᴸ_rs, `b_mo` of `pipeline.eri`'s B. Its pieces (the
tex's Eqs. P–W) go into the same arrays as every term above:

- P = C h̄ Cᵀ into the one-electron D (one `gemmwich`, an axpy);
- ½ C sym(X) Cᵀ into the overlap's W, as ½ X added to F;
- Γ = L⁻ᵀ C Λ Cᵀ: Λ_AO = C Λᴸ Cᵀ by a gemm over q and a strided-batched gemm
  over n (df_rhf's exchange shape) straight into the zeroed Γ_mnP, then one
  trsm in place before any other term is added;
- W_PQ += −½ (L⁻ᵀ Gᵃᵘˣ L⁻¹)_PQ, Gᵃᵘˣ_LL′ = Σ_pq Λᴸ_pq Bᴸ′_pq, taken in
  the AO basis as Λ_AO Bᵀ (a gemm) and two trsms.

So the derivative integrals are still evaluated once: one
`three_center_gradient`, one `metric_gradient`. E_cas's own adjoint
(h̄ = D, Λᴸ = Σ_rs Γ_pqrs Bᴸ_rs over the full-range Γ, X = 2F) reproduces
the terms above exactly, which the test uses.

## A transition density

`df_transition_gradient` takes the same terms at the caller's (γ, Γ), with
the core's own energy switched off: the Hellmann–Feynman half of a coupling
h_IJ = ⟨I|∂H|J⟩ between orthogonal roots (PySCF's `grad_elec_active`), which
`df_nac` builds on. γ and Γ are `ci::Rdm::trans_rdm12`'s symmetrized
transition RDMs (row-major, γ symmetric, Γ_tuvw = Γ_vwtu = Γ_utwv), or any
pair with those symmetries.

**Why the core-core term goes.** At fixed orbitals the CASSCF energy is
affine in the active RDMs,

    E(γ, Γ) = E_nuc + E_cc + Σ_tu h_tu γ_tu + Σ_tu (J − ½K)[D_c]_tu γ_tu + ½ Σ Γ_tuvw (tu|vw),

E_cc = Σ_i 2 h_ii + Σ_ij [2 (ii|jj) − (ij|ij)] the core's own energy and D_c =
2 C_c C_cᵀ. ⟨I|H|J⟩ is that functional at the transition RDMs, but E_nuc and
E_cc multiply ⟨I|J⟩ = 0, so they drop: what is differentiated is

    E_act(γ, Γ) = Σ h D_a + J[D_c]·D_a − ½ K[D_c]·D_a + ½ Σ Γ_tuvw (tu|vw),

D_a = C_a γ C_aᵀ: E(γ, Γ) − E(0, 0). Each term above loses its core-core part
and keeps the rest:

- **One-electron.** D = D_a (no 2 C_c C_cᵀ).
- **Overlap.** The generalized Fock of E_act is F(γ, Γ) − F(0, 0), and
  `response::fock` at zero RDMs is 2 F^I in its core columns and zero
  elsewhere (F^I = Cᵀ (h + 2J − K)[P_c] C includes h): so F loses 2 F^I in
  its core columns, leaving 2 F^A there and the active columns as they were.
  W = C sym(F) Cᵀ over that F. This is PySCF's `gen_g_hop_active` patch,
  which sets h + vhf_c to zero in the core columns.
- **J.** ½ J[D]·D − ½ J[D_a]·D_a less the core-core ½ J[D_c]·D_c is the
  cross term J[D_c]·D_a alone, a bilinear form: with d_c and d_a the two
  fits, Γᴶ = D_c ⊗ d_a + D_a ⊗ d_c and Wᴶ = −½ (d_c d_aᵀ + d_a d_cᵀ), four
  rank-one gemms (no cancellation between large core terms).
- **K.** G = diag(0_core, γ): the core block of G is the core-core exchange
  −Σ_ij (ij|ij), so it is zero, and the core-active −Σ_i,tu γ_tu (it|iu)
  stays.
- **Active–active.** Unchanged, at Γ.
- **Nuclear repulsion**: zero.

**The orbital gradient** of E_act, the Z-vector's right-hand side, is by the
same linearity `response::fock`'s gradient at (γ, Γ) less its gradient at
zero RDMs (g = 2 (F − Fᵀ), linear in F): only the pairs (p, i), i core and
p not, move, each by −4 F^I_pi. No entry point of its own is needed. It
equals PySCF's `gen_g_hop_active` orbital vector element for element
(`gen_g_hop` returns 2 pack(F − Fᵀ), this tree's g), as the test measures.

**Multipliers on top.** Without `multipliers` nothing responds and no
Z-vector is solved, at any number of roots: the call is the Hellmann–Feynman
half alone. With them it is `df_gradient`'s Lagrangian at the transition
density: z_c's transition RDMs (at the solution's weights, which it then
recovers) are added to (γ, Γ), which carry no core energy either, and the
z_o response is the SA gradient formula's derivative at the solution's
weighted RDMs, core included, exactly as for a root. `extra` is unchanged.
So `df_nac` passes its own z = (z_o, z_c) for h_IJ's Lagrange half and gets
h_IJ whole from one call.

## Memory

**Γ is held in full**: naux nao² doubles, the size of B, as in
`gradient::df_rhf`, plus the derivative call's Cartesian copy in
`res.device_workspace()`. Beside it the call holds W (naux²), the two fitted
densities, G (nocc²), the core pairs and V (naux ncore nocc each), one
back-transform (naux max(ncore, ncas) nao), and the active pairs and Z (naux
ncas² each); for a state average (or given multipliers) also four more fitted
densities, a second G, and the response's pairs, their scratch and the SA V
and Z (three more of each size). It refuses (`Unsupported`, naming the bytes)
when those would not fit the device's free memory, before anything is
allocated. A state average's CI objects, Hessian and nao²-sized matrices are
not counted. An extra adjoint adds Λ Cᵀ (naux nao², the size of Γ) and Gᵃᵘˣ
(naux²), both counted; Λ itself is the caller's. `casscf_integrals` and
`response::fock` hold their own nmo² ncas² integrals for the call. naux nao²
must fit `int32_t`.

## Tests

`test/gtest/mcscf_gradient_gpu.cpp`, against
`test/gtest/data/casscf_gradient/`'s state-specific cases (PySCF's
`df.grad.casscf` over a DF-RHF `_scf` sharing the CASSCF's `with_df`, so its W
is fitted as the energy is): H₂O and the skewed H₂O in cc-pVDZ /
cc-pVDZ-JKFIT, CAS(4,4); and against `df.grad.sacasscf`, both roots of H₂CO
SA-2 CAS(4,3) at [0.5, 0.5], whose parts include the multipliers' terms. Each
runs this tree's own `DensityFitting::Full` solve from the exact RHF and the
case's active orbitals to ‖g‖ < 1e-9, then for every root every term and the
total to 1e-8 absolute, the total's sum over atoms below 1e-9, and two
refusals (a root past the solution's, and a solution with a root more than its
per-root RDMs). Then the caller's pieces on the same solution, to 1e-10: for
each state-specific case E_cas's own MO adjoint, built on the host from B, C,
h and the RDMs, fed as `extra` doubles every electronic term and leaves the
nuclear repulsion, so the extra term alone is the state-specific gradient;
for the state average each root's multipliers from `zvector` (over the
CI-relaxed Hessian, as the call builds it) passed as `multipliers` reproduce
the internal path; and multipliers of the wrong length are refused. The gradient does not depend on the orbitals' or CI's signs or
canonicalization, so the solution is compared as it converges.

`test/gtest/mcscf_transition_gradient_gpu.cpp` holds the transition
density, over `test/gtest/data/casscf_nac/` (H₂CO SA-2 and the skewed H₂CO
SA-3, CAS(4,3), cc-pVDZ / cc-pVDZ-JKFIT): this tree's DF-SA-CASSCF put in the
reference's canonical orbitals with its CI re-solved (as
`mcscf_zvector_gpu.cpp` does), then for every ordered pair, at
`trans_rdm12`'s RDMs, the electronic terms' sum against PySCF's
`grad_elec_active` to 1e-9 up to the sign s_I s_J of the two CI phases, and
the orbital gradient (`fock` at the RDMs less at zero) against
`gen_g_hop_active`'s, by (p, q) and in magnitude. With arbitrary
multipliers, each root's `df_gradient` less its `df_transition_gradient` at
its own RDMs is one gradient (the core's own and the nuclear repulsion) for
every root and every z, to 1e-10.
Measured on the RTX 3080: the Hellmann–Feynman terms within 8e-11, the
orbital gradient within 7e-9 (1e-11 at two roots), the multipliers' core
part to 2e-13.

`test/gtest/mcscf_gradient_weights_gpu.cpp` holds the weights, over
`test/gtest/data/casscf_hessian/`'s stretched water (CAS(4,4) / 6-31G /
cc-pVDZ-JKFIT), its orbitals and roots loaded rather than re-converged: every
root of two roots at (0.7, 0.3), of three at equal weights and of three at
(0.5, 0.3, 0.2) against the fixture's exact gradient to 1e-9 (measured on the gfx1200: at
most 7e-12, the three unequal weights included).
