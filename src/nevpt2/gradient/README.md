# `free_breakfast.nevpt2.gradient`

The SC-NEVPT2 correlation energy of one root as a function of the integrals
(h, B) and the CI vector c, and its adjoint: the first piece of the NEVPT2
nuclear gradient (`nuclear_gradient.tex` §5). Then the inactive-rotation
multipliers z^c of its Lagrangian and the term Φ = ⟨z^c, f⟩ they add, with
Φ's adjoints (Algorithm 1, steps 1–2). Then the gradient itself, of a
single-state DF-CASSCF: the Z-vector, the Lagrangian's symmetry gate and
the nuclear term (Algorithm 1 from step 6). And all of it again for
PC-NEVPT2 (`nuclear_gradients.tex` Path 4), the same code over the PC class
energies' reverse pass. Link `free_breakfast::nevpt2::gradient`.

    sc_energy(res, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, vector, batch)
        -> ClassEnergies
    sc_adjoint(res, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, vector, batch)
        -> ScAdjoint
        energies   sc_energy's
        h_bar      dE/dh, nmo², MO basis of c, symmetric
        lambda     dE/dB^L, naux × nmo² ((L, p, q) at L + naux (p + nmo q)), symmetric in (p, q)
        y          dE/dc, at fixed (h, B)
        x          X = 2 h h̄ + 2 Σ_L B^L Λ^L, nmo²
        a          A = X − Xᵀ
        epsilon    ε, nmo: f's core and virtual diagonals at their own indices, 0 on the active
    zc_multipliers(res, nmo, ncore, ncas, a, epsilon) -> ZcMultipliers
        z          z^c, nmo², symmetric, on the core–core and virtual–virtual blocks
        max_z      max |z^c|
        min_gap    the smallest |ε_p − ε_q| over those pairs
    zc_adjoint(res, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, weights, vectors, z)
        -> ZcAdjoint
        phi        Φ = ⟨z, f(γ_SA)⟩ = Σ_pq z_pq f_pq, γ_SA = Σ_r w_r γ_r
        h_bar      dΦ/dh = z
        lambda     dΦ/dB^L, as ScAdjoint's
        y          y^c_r = dΦ/dc_r per root, as ScAdjoint's y, root r's at r ndet
        x, a       X and A of (h̄, Λ) above, as ScAdjoint's
    lagrangian_gradient(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas,
                        corr, zc, phi) -> ScGradient
    sc_gradient(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, batch)
        -> ScGradient
        terms        dE/dR of E_cas + E_corr, mcscf::GradientTerms (n_atom × 3)
        energies     E_corr's class energies
        zc           z^c, max |z^c|, the smallest inactive gap
        multipliers  (Z, z), mcscf::ZVector: zvector's coordinates, its
                     iterations and residual
        asymmetry    max |X^L − X^Lᵀ| per block
    pc_energy(res, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, vector, batch)
        -> PcClassEnergies
    pc_adjoint(res, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, vector, batch)
        -> PcAdjoint     energies (PcClassEnergies), then ScAdjoint's arrays
    pc_gradient(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, batch)
        -> PcGradient    ScGradient's fields, energies PcClassEnergies
    each takes a trailing nfrozen = 0 and pc_floor = 0: PC's conditioning floor
    (EnergyInputs::pc_floor, ../energy/README.md "The conditioning floor", #917),
    read by the energy and its adjoint alike
    qd_sc_energy(res, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, weights, vectors, batch)
        -> std::vector<ClassEnergies>   the SC-QD H_eff's n² correlation elements, [M n + N]
    qd_sc_adjoint(res, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, weights, vectors, w,
                  batch)
        -> QdScAdjoint   elements, then ScAdjoint's arrays of E[W] = Σ_MN W_MN H_MN,
                         y every root's (root r's at r ndet), epsilon the SA Fock's
    qd_pc_energy(res, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, weights, vectors, batch)
        -> std::vector<PcClassEnergies>   the PC-QD H_eff's n² correlation elements
    qd_pc_adjoint(res, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, weights, vectors, w,
                  batch)
        -> QdPcAdjoint   elements (PcClassEnergies), then QdScAdjoint's arrays

    qd_sc_lagrangian(res, molecule, basis, aux, h, b, solution, ncore, ncas, state, batch)
        -> QdScLagrangian
        eigenvalues  H_eff's, ascending
        eigenvectors R, column K at [K n, (K + 1) n), largest component positive
        w, rho       W = u_K u_Kᵀ (n², row-major) and ρ_r = W_rr
        weights      the SA weights, mcscf::sa_gradient_weights'
        corr         qd_sc_adjoint at W plus Φ^SA's adjoint: y = ȳ, a = A^corr + A^{z^c}
        zc           z^c
        pairs        τ, ζ and t (mcscf::RootPairMultipliers)
        zeta         ζ · g and (γ^ζ, Γ^ζ) (mcscf::ZetaTerms)
        multipliers  (Z, z, Θ), mcscf::ZVector
        asymmetry    the gate's residues per block
    qd_sc_lagrangian(res, molecule, basis, aux, h, b, solution, ncore, ncas, w, batch)
        -> QdScLagrangian   the W form: any symmetric W (n², row-major), ρ = diag W,
                            eigenvalues and eigenvectors empty
    qd_sc_gradient(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, state,
                   batch)
        -> QdScGradient
        total        dE_K/dR, n_atom × 3: terms.total + transition.total
        terms        the df_gradient pass (mcscf::GradientTerms)
        transition   the df_transition_gradient pass at ζ's folded RDMs (zero when ζ is)
        lagrangian   qd_sc_lagrangian's QdScLagrangian
        max_zeta     max |ζ|
        root_gap     the smallest |E_r − E_s| over the CASSCF roots
        heff_gap     the smallest |λ_K − λ_L| of H_eff, L ≠ K
    qd_sc_gradient(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, w, batch)
        -> QdScGradient  G(W), linear in W, the state's at u_K u_Kᵀ; heff_gap infinite
    qd_sc_gradient(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, ws, batch)
        -> std::vector<QdScGradient>  G(W) at each of the list ws, one shared solve (below)
    qd_sc_coupling(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, p, q,
                   batch, fold = {})
        -> QdScGradient  h_PQ = R_Pᵀ (∇H_eff) R_Q, G(W^PQ); heff_gap |E_Q − E_P|
    qd_pc_gradient(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, state,
                   batch)
        -> QdPcGradient  QdScGradient's fields, lagrangian PC's (its corr's elements
                         empty), plus elements (PcClassEnergies, qd_pc_energy's)
    qd_pc_gradient(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, w, batch)
        -> QdPcGradient  PC's G(W), linear in W, the state's at u_K u_Kᵀ; heff_gap infinite
    qd_pc_gradient(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, ws, batch)
        -> std::vector<QdPcGradient>  PC's G(W) at each of the list ws, one shared solve
    qd_pc_coupling(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, p, q,
                   batch, fold = {})
        -> QdPcGradient  PC's h_PQ = R_Pᵀ (∇H_eff) R_Q, G(W^PQ); heff_gap |E_Q − E_P|
    qd_inputs(res, molecule, basis, aux, h, b, solution, ncore, ncas, use)
        -> Status        hands use(blocks, pairs) qd_sc_energy's inputs on the solution
    builders_tangent(res, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, vector,
                     count, h_dot, b_dot, c_dot)
        -> BuildersTangent  the builders at the point and their forward tangent along
                            each MO direction (h_dot, B_dot, c_dot); below
    sc_hessian_product(res, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, vector, z,
                       batch, directions, partners = {}, parts = false)
        -> ScHessianProduct  the tangent of sc_adjoint + zc_adjoint at fixed z along
                             each MO direction, the CI one on the unit sphere; below
    pc_hessian_product(res, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, vector, z,
                       batch, directions, parts = false, pc_floor = 0, span = true)
        -> PcHessianProduct  the same of pc_adjoint + zc_adjoint; below

Every call above but `zc_*`, `lagrangian_gradient`, the couplings, `builders_tangent` and
the two Hessian products takes
a trailing `nfrozen`, 0 by default: the frozen core ("The frozen core").

`qd_inputs` is the QD energies' inputs, built on the solution as
`qd_sc_energy` builds them (the weights `mcscf::sa_gradient_weights`
recovers, the active strings, the shared arrays and every ordered pair's
RDMs) and handed to `use` with the MO blocks, for the call's length only: it
is how `../nac/` reaches the PT2 transition density's inputs from a
solution.

`lagrangian_gradient`, `sc_gradient` and `pc_gradient` take
`mcscf::df_gradient`'s arguments (`solution` a
single-state DF-CASSCF, its orbitals core and virtuals canonical, as the
Casscf step leaves them); the others':
`b` is B as `density_fit` leaves it (the AO B, naux × nao², the metric
folded in), `c` the root's orbitals (nao × nmo: core, active, virtual), `h`
the AO core Hamiltonian, `vector` the unit CI vector over the two string
sets. h̄, Λ and X are in the conventions `mcscf::MoAdjoint` takes (#572).

## The functional

Everything `class_energies` reads is built here from (h, B, c), with B in
the MO basis of `c` as `b_mo` would leave it:

    F^I_pq = h_pq + Σ_j [2 (pq|jj) − (pj|jq)]
    ε_p    = F^I_pp + Σ_tu γ_tu [(pp|tu) − ½ (pt|up)]     p core or virtual
    h1e = F^I_aa,  h1e_v_si = F^I_ac,  h1e_v_sir = F^I_vc,
    h1e_v_sr = F^I_va − Σ_u (au|ut)

(pq|rs) = Σ_L B_L,pq B_L,rs. h2e is `nevpt2::h2e`'s, (tu|vw) one einsum over
B_aa, and dm3, f3ca and f3ac `nevpt2::rdms`' over (tu|vw). **The orbitals are
taken as they are**: ε is f's diagonal, so on canonical orbitals (the
Nevpt2 step's per-root ones) it is the step's eigenvalues, and off them it is
not re-canonicalized. That is what makes A's core–core and virtual–virtual
blocks the derivatives #568's z^c needs: SC-NEVPT2 is not invariant under
those rotations, and the pseudocanonical conditions f_ij = 0, f_ab = 0 are
the Lagrangian's, not the functional's.

**ε moves with γ.** It is computed from (h, B, γ) at every point, never an
input, so ē adds a bar on γ (and so on y) and on (h, B) through F. Holding ε
fixed would differentiate a different functional, and a check whose two
sides shared the omission could not see it; the ladders below rebuild ε at
every point.

**dm1 and dm2 are traces of dm3**, dm2[p,q,r,s] = Σ_t dm3[p,q,r,s,t,t] / N and
dm1 the same of dm2 (Σ_t E_tt is the number operator, exact on an N-electron
vector). That is `make_dm123`'s dm2, the Nevpt2 step's Γ + δ_qr γ_ps, to
round-off. So γ̄ and dm2̄ fold into dm3̄ by the adjoint of the trace, and y is
`nevpt2::rdms_adjoint`'s alone: no sigma vector is needed for them, and no
symmetry of their bars is assumed (`ci::Sigma` takes an eightfold-symmetric
(tu|vw), which a 2-RDM's bar is not).

Beyond `nevpt2::mo_blocks`' five blocks, the builders read **B_cc whole**
(b_mo over the core: (ii|jj), (ij|ji), (ti|jj), (tj|ji), (ai|jj), (aj|ji)) and
**B_vv's diagonal** (B_Q,aa from `b_half` over the virtuals: (aa|jj),
(aa|tu)). Nothing reads B_vv off its diagonal.

## The reverse pass

`class_energies_adjoint` takes E back to every `EnergyInputs` field, and
through the slabs onto Λ. Then each builder, listed once as einsum terms
(`fock_builds`), is run backwards, the last first: the orbital energies and
the four F^I blocks onto the MO blocks', B_cc's, B_vv diagonal's, d's, g's
and γ's bars (d_Q = Σ_j B_Q,jj and g_Q = Σ_tu γ_tu B_Q,tu, the core's and the
active space's Coulomb densities); g and d; dm1 and dm2 into dm3̄. Each term's
adjoint is the same einsum with the output and the operand swapping roles,
so no derivative is written by hand; a trace goes through a δ operand, which
takes no bar, so no operand with a bar repeats a label. Then:

- `rdms_adjoint` takes dm3̄, f3cā and f3ac̄ onto y and (tu|vw)'s bar, and the
  (tu|vw) builder's adjoint takes that onto B_aa's;
- `h2e_adjoint` takes h2ē onto Λ, `mo_blocks_adjoint` the five blocks' bars
  and B_vv diagonal's, and **B_cc's bar goes onto Λ's core–core corner whole,
  off its diagonal too**, by one geam here (`nevpt2.density_fitting`'s
  adjoint has only the diagonal; its README, "Not done");
- h̄ is the F^I blocks' bars at their corners of C^T h C and the orbital
  energies' on its diagonal.

h̄ and Λ are symmetrized last (Λ per L, (Λ + Λᵀ) / 2), since h and each B^L
are symmetric and both X and `mcscf::df_gradient` read them so.

**The orbital source term** is `nuclear_gradient.tex` Eq. Xao, an AO
half-transform rather than a probe pair by pair: Λ̃_q = Λ_q Cᵀ per q (one
strided-batched gemm, naux × nao each), D = B_AOᵀ Λ̃ with B_AO read as an
(naux nao) × nao matrix, which its symmetry in the AO pair allows (the DF
exchange's shape), and X = 2 Cᵀ D + 2 h h̄. Along C → C (1 + θ),
dE = Σ θ_xp X_xp; along C → C exp(κ), κ antisymmetric, dE = Σ_{p>q} κ_pq A_pq.
Every block of A has a reader: core–core and virtual–virtual #568's z^c,
active–active the gauge identity, the mixed blocks the Z-vector.

**Memory.** Beside `class_energies_adjoint`'s and `rdms_adjoint`'s own, the
inputs and their bars are buffers of their own (all are held across calls
that take the workspace): two n⁶ (dm3 and the f3s) of each, the MO blocks
twice, and Λ, naux nmo², three times at the end (the raw one, the
symmetrized one, and Λ̃, naux nao nmo). `nevpt2::rdms` and `rdms_adjoint`
each get half the device's free memory at their call, as the Nevpt2 step
gives `rdms`, so that the energy's own buffers find room after the workspace
has grown (it never shrinks).

## The forward tangent of the builders

    builders_tangent(res, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, vector,
                     count, h_dot, b_dot, c_dot) -> BuildersTangent
        value      BuilderOutputs at the point
        tangents   BuilderOutputs per direction: their tangent along (ḣ, Ḃ, ċ)_k
    BuilderOutputs
        blocks     the five MO blocks (MoBlocks)
        eri        (tu|vw)
        h1e … f3ac every EnergyInputs field, packed as the forward holds it; inputs()
        f_core     f_ij over the whole core, ncore², column-major
        f_virt     f_ab over the virtuals, nvirt², column-major

The first piece of the NEVPT2 nuclear Hessian's E_corr + Φ part
(milestone 24, #806): what the class walk's tangent (#805) and the κ_oo
response (#813) read. The arguments are `sc_energy`'s; direction k is ḣ
(nmo², symmetric) at `h_dot + k nmo²`, Ḃ (naux nmo², `lambda`'s layout,
symmetric in (p, q)) at `b_dot + k naux nmo²` and ċ (ndet, any vector, c
raw as for y) at `c_dot + k ndet`, **all in the MO basis of `c`**: the
tangent of a quantity in the MO frame, so that a caller's t_B (#813, the
derivative integrals plus the one-index transform by κ plus c^B) goes in as
it stands.

**Every builder term by its product rule.** `fock_builds`' terms, the ones
the forward and the reverse pass already run, are run once per operand
that varies, that operand swapped for its tangent (the same builds over the
tangent arrays, so every view and stride is the forward's) and the sum onto
the output's tangent; the δ operands are the only constants. So ε's tangent
carries γ̇ (**ε moves with γ** here too) and ḋ, ġ, and nothing is written
twice by hand. h2e's tangent is (tu|vw)'s permuted, h2e[t, u, v, w] =
(tv|uw); the MO blocks', B_cc's and B_vv diagonal's are slices of Ḃ.

**The RDMs.** dm3, f3ca and f3ac are quadratic in c and f3 linear in
(tu|vw), so their tangent is exact in three `nevpt2::rdms` calls: the
bra/ket build at (ċ, c) and at (c, ċ) (`../rdm/README.md`, "Bra and ket":
each is multilinear in the two vectors, and their sum is the symmetrized
one), and the one-vector build at c over (tu|vw)'s tangent, of which only
f3 is kept. dm2 and dm1 are their traces, linear, as in the forward. No
`rdm/` entry point was missing.

**f's inactive blocks** are `zc_adjoint`'s f at one root,

    f_pq = h_pq + Σ_L B_L,pq (2 d_L + g_L) − Σ_j (pj|jq) − ½ Σ_tu γ_tu (pt|uq)

over the core–core and virtual–virtual blocks, whose diagonals are e_core
and e_virt term for term; the virtual block reads B_vv whole (one `b_mo`),
which nothing else here does. f's off-diagonal tangent is what #813's κ_oo^B
= −f^(B)_pq / (ε_p − ε_q) divides. The value's own f is pinned against
`zc_adjoint`'s Φ (Tests).

**All-electron.** It takes no `nfrozen`: the Hessian it feeds refuses a
frozen core (milestone 24's update), and the frozen path's builders are
untouched. One `BuilderOutputs` per direction holds three n⁶ arrays; the
three RDM builds share one more triple (the second and third builds'
outputs, added onto the first's), held across the `rdms` calls and so its
own buffers.

## The SC Hessian-vector product in (h, B, c)

    sc_hessian_product(res, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, vector, z,
                       batch, directions, partners = {}, parts = false) -> ScHessianProduct
        energies    sc_energy's class energies
        products    AdjointTangent per direction: the tangents of h_bar, lambda, y, x, a
        populated   class_energies_hvp's, per direction: the empty perturbers it populates
        defect      class_energies_hvp's weighted ρ_τ with the partner, per direction
        parts       with parts: per direction, the product's five parts (below)
    MoDirections  {count, h, b, c}: builders_tangent's direction layout, MO basis

The E_corr + Φ part of milestone 24's Lagrangian Hessian (#814,
`nuclear_hessian.tex` Eq. Lxx's level 1 and z^c term, and L_Ax's NEVPT2
part): the tangent of `sc_adjoint` + `zc_adjoint` at fixed z^c along each
direction (ḣ, Ḃ, ċ), in the MO basis of `c` as `builders_tangent` takes
them, so #813's t_B goes in as it stands. ⟨t_A, product(t_B)⟩ over the
parameter and integral directions together is then this part's share of
#816's Σ_AB. All-electron: it takes no `nfrozen`. `z` is z^c as
`zc_multipliers` forms it (one root, weight 1).

**Forward over reverse.** The adjoint is (h̄, Λ, y) = Jᵀ Ē, Ē the class
walk's bars at the builders' outputs I(h, B, c) and J the builders'
Jacobian, then X and A from (h̄, Λ). Its tangent has two kinds of term:

- **Jᵀ Ė**: `class_energies_hvp` (#811) takes the class walk's bars to their
  tangent Ė along `builders_tangent`'s outputs (#806) and Ḃ, and Ė runs back
  through the same reverse pass the gradient runs (`fock_builds`' adjoints,
  the traces', the one-vector `rdms_adjoint`, (tu|vw)'s and the folds).
- **J̇ᵀ Ē, the one-slot-dotted terms**: every builder term's adjoint at the
  fixed bars Ē (the gradient's own, kept from `sc_adjoint`'s pass) with one
  other operand at its tangent, the same einsum (`adjoint_slot`) with the
  operand swapped, so no derivative is written by hand; and the reverse
  steps that are not einsums: `rdms_adjoint` is bilinear in its bars and c
  and linear in (tu|vw) through f3, so its share is the bra/ket adjoint at
  (c, u) (the bra's bar) and at (u, c) (the ket's), their (tu|vw) bars
  added, and the one-vector adjoint at (tu|vw)'s tangent with dm3's bar
  zero (y's share; that call's own (tu|vw) bar does not move and is
  dropped); `h2e_adjoint` is linear in B, so it runs once more at Ḃ's
  blocks at the fixed bar. Every one-slot contribution onto an
  intermediate's bar (d, g, γ, the RDMs, (tu|vw), B_aa) runs back through
  the rest of the pass with Ė's. No `rdm/` entry point was missing.

Both are one sweep, `bars_tangent`, the builds the last first as
`builders_adjoint` runs them.

**Φ's share is closed form.** Φ = ⟨z, f⟩ is linear in h, bilinear in B and
quadratic in c, so at fixed z: Λ^c = z g_L + D w_L − ½ (z B_L D + D B_L z)
(g_L = ⟨B_L, D⟩, w_L = ⟨B_L, z⟩) is bilinear in (B, D) and its tangent is
Λ^c(Ḃ, D) + Λ^c(B, Ḋ), Ḋ the active block's γ̇ (`builders_tangent`'s dm1);
y^c = σ(W + Wᵀ) c with W = Σ_L w_L B_L − ½ Σ_L B_L z B_L on the active
block, quadratic in B, so ẏ^c = σ(W + Wᵀ) u + σ(Ẇ + Ẇᵀ) c,
Ẇ = W(Ḃ, B) + W(B, Ḃ); h̄^c = z does not move.

**X and A** are formed in the MO basis, X = 2 h h̄ + 2 Σ_L B_L Λ_L with B
whole (one `b_mo`), which is the AO half-transform's X exactly (Cᵀ B_AO C Λ
= B_MO Λ); its tangent is 2 h ḣ̄ + 2 B Λ̇ + 2 ḣ h̄ + 2 Ḃ Λ, h̄ and Λ the
summed adjoint's (E_corr's and Φ's), and A = X − Xᵀ.

**The CI direction is on the unit sphere: the Rayleigh-normalized
functional.** `nuclear_hessian.tex` ("Which functional of c",
"Normalization") leaves two conventions; this is the second. The product
is the Hessian of F(h, B, c) = (E_corr + Φ)(h, B, c/|c|), degree 0 in c, at
the unit `vector`: h̄, Λ, X and A are the raw ones' tangents along
(ḣ, Ḃ, u), u = ċ − (c·ċ) c, and the c-gradient F's is P y/|c| (P = 1 − c cᵀ,
y = y^corr + y^c), whose tangent is

    P ẏ(ḣ, Ḃ, u) − u (c·y) − c (u·y) − (c·ċ) P y.

The first extra term is the sphere's −(u·v)(∂L/∂c)·c on CI–CI pairs, which
the tangent builders on a linear unnormalized c miss; the others vanish
against directions ⊥ c, and along c itself the product's y is −P y, Euler's
identity for a functional of degree 0. A caller dotting a CI response ⊥ c (#813's
c^B) gets the sphere's element; one along c gets zero, as it should. The
gradient's y is unchanged by the choice: its Z-vector projects y off the root,
P y, which is F's c-gradient. `vector` must be a unit vector (1e-10 on
|c|², else InvalidConfig).

**The off-shell extension is the gradient's**, the rank-reduced one: the
class tables use Ĥ_D|Ψ₀⟩ = E₀|Ψ₀⟩ off the CASCI root too, as
`class_energies` and its adjoint do ("The functional"), and the product is
that functional's second derivative, with no Kollmar residual. A
residual-including CI–CI block against this gradient's z, or the reverse,
would be wrong without a symptom (`nuclear_hessian.tex` "Which functional of
c"); the Hessian built on this product and `sc_gradient`'s multipliers is
consistent because both differentiate the same functional.

**Empty perturbers.** They are `class_energies_hvp`'s: K6 along each
direction, and with `partners` (one per direction, the same layout,
projected the same way) its ρ_τ check on the pair, a pair past 1e-6
refused (Numerical, naming class and perturber). Thresholds are #811's,
untouched here (#914 validates them).

**Parts.** With `parts`, the product is also returned split into what each
separately coded piece contributes, for the visibility rule: (0) Jᵀ Ė of
`class_energies_hvp` along Ḃ alone, the slabs' own B; (1) Jᵀ Ė along the
builders' tangent alone; (2) the one-slot-dotted terms, the X terms
2 ḣ h̄^corr + 2 Ḃ Λ^corr among them; (3) Φ's; (4) the normalization's,
y alone (−c (c·ẏ) − u (c·y) − c (u·y) − (c·ċ) P y). They sum to the product
when no empty perturber is populated (K6's η depends on the direction, so
the class walk's split is not linear there). Two more `class_energies_hvp`
calls and three more sweeps per call: a test's option, not production's.

**Cost and memory.** The base pass is `sc_adjoint`'s plus `zc_adjoint`'s;
per direction, `builders_tangent`'s three `rdms` and one ScArrays (three
n⁶), its share of `class_energies_hvp` (seven reverse passes), one
ScArrays of tangent bars and a Λ (naux nmo²), four `rdms_adjoint` calls
(three with bars fixed), two σ products for Φ, and Φ's Λ tangent (two
naux nmo² scratch einsums). Directions are held together through the class
walk; a caller with many chunks them.

## The PC Hessian-vector product in (h, B, c)

    pc_hessian_product(res, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, vector, z,
                       batch, directions, parts = false, pc_floor = 0, span = true)
        -> PcHessianProduct
        energies    pc_energy's class energies
        products    AdjointTangent per direction, as sc_hessian_product's
        lifts       per direction, each class's Schur lift ℓ per |u|² (below)
        lift_steps  per direction, each class's t*, in the direction's own units (below)
        parts       with parts: per direction, the product's five parts, as SC's

`sc_hessian_product` with `pc_class_energies_adjoint` and
`pc_class_energies_hvp` (`nevpt2.energy`'s README, "The PC Hessian-vector
product") in place of the SC class walk, and nothing else (#815), as
`pc_adjoint` is `sc_adjoint`'s: the builders' tangent, `bars_tangent`, Φ's
closed form, X and A, and the Rayleigh normalization are one function
(`hessian_product` in `hessian_product.cpp`) that both class walks feed.
`pc_floor` is `pc_adjoint`'s; the class walk refuses a class whose kept
metric spectrum reaches below 1e-10 (#917), and a pole, as the energy does.
`span` false leaves the kept span's rotation out, for the tests alone.

**No partners.** PC has no empty perturbers: a tuple whose norm vanishes by
symmetry has b = S x = 0, and moves as a smooth quadratic in the inputs.
What it has instead is the Schur lift, which needs no partner (below).

### The Schur lift

**A null mode can rise at second order.** S is the Gram matrix of the
class's perturbers {O_p |Ψ⟩}, positive semi-definite along every path of
states. Along the CI vector's path on the sphere, c(t) = (c + t u) /
|c + t u| (u ⊥ c), S(t) = S + t S' + ½ t² S'' with S' and S'' the metric
builder at the RDMs' derivatives, γ(c, u) + γ(u, c) and
2 γ(u, u) − 2 |u|² γ(c, c). For a null mode v (S v = 0), vᵀ S' v = 0
(S(t) ⪰ 0), and the null block's eigenvalues rise as ½ t² eig(L), with
the Schur complement

    L = P (S'' − 2 S' S⁺ S') P,   S⁺ = U Uᵀ,   P the null modes' projector,

itself positive semi-definite. The null modes are of three kinds.

- **Structural**, operator identities null for every state: v annihilates
  Σ v_p O_p on any vector, so P S' = 0 and P S'' P = 0, and L = 0.
- **Spin**: null for every singlet (Sr's and Si's quartet modes), so a
  singlet path keeps them; a quintet admixture lifts them. The directions
  the Hessian takes (#813's CP-CASSCF responses) are singlets.
- **Spatial symmetry**: at a symmetric geometry the root's configurations
  lie in one irrep, so {O_p |Ψ⟩} can span less than a generic state's does,
  and a singlet that breaks the symmetry lifts them, L > 0.

**It matters under the current cut.** A lifted mode crosses the cut, at
max(1e-13, 100 ε √d, `pc_floor`) of σ_max, at t* = √(2 cut σ_max / ℓ),
ℓ = eig(L), and is kept beyond, where the energy carries its share:
½ ℓ t² (vᵀ x̃)² / (λ + Δ) for b = S x̃, a quadratic of the Hessian's own
size. At the point the product is the Hessian of the energy with the mode
dropped, which is the energy within |t| < t*; at the default cut t* is
about 1e-5 (3e-4 at a floor of 1e-10). #853's cut, with no gap test, does
not keep the null modes inside the dropped set: it is what lets them out.

**Reported, not refused.** Whether t* is small is a question of the step,
and a step "per unit |u|" means nothing for a CI direction: only the
assembly (#817) knows the physical response c^B a nuclear step moves c by.
So the product stays the exact derivative of the dropped branch and reports,
per direction and class, ℓ (`lifts`: L's largest eigenvalue over σ_max, per
|u|², u the CI part off c) and t* = √(2 cut / (ℓ |u|²)) (`lift_steps`, in
the direction's own units, cut the class's relative cut; infinite where ℓ
is not positive). #817 refuses where any nuclear coordinate's t*_B is below
1e-3 bohr: the surface switches branch within a vibrational or
finite-difference step.

**Real dropped modes are not lifts.** A real mode's σ_n(t) is smooth and
the product differentiates it (the span bar's Daleckii–Krein term); it
crosses the cut at first order, carrying energy of the cut's own size,
which is the conditioning floor's business (#917). The check reads the null
modes alone, σ_n at most 10 ε √d of σ_max (every mode round-off or the
RDMs decide has measured at most 7.4 ε √d), whose own σ the rise outruns.

**The pair.** L is positive semi-definite for every physical direction, so
L_uu = 0 makes L_uv = 0 for every v: a pair lifts only through a direction
that lifts on its own, and each direction's own report is the whole
story. No partners.

**Measured** (per |u|², over σ_max, RTX 3080, `--preset default`):

| case | direction | largest lift |
|---|---|---|
| case | direction | largest ℓ | t* | ladder |
|---|---|---|---|---|
| `h2co_skew_sto3g` (C1), `pc_floor` 1e-6 | two random singlets | 2.3e-15 | | 1.4e-9 |
| PSB3 twist, `pc_floor` 1e-8 | two random singlets | 1.2e-4 (Sr, its round-off nulls) | 2.5e-2 | 5.3e-8, steps ≤ 1e-3 |
| `hcn_sto3g` (C∞v) | h's active block on c (keeps the symmetry), orbital-only | 8.8e-16 | | 5.6e-11 |
| `hcn_sto3g`, default cut | a random singlet (bends) | 2.0e-2 (Sir) | 7e-6 | clean ratios, 1e-2 to 1.5e-1 off |
| `hcn_sto3g`, `pc_floor` 1e-6 | the same | 2.0e-2 | 2.0e-2 | 4.7e-11 |

Along the bend at the default cut, the ladder of PC's adjoint converges
cleanly (ratios 4.000) to the energy with Sir's mode kept, 1e-2 to 1.5e-1
off the product in every field, and its rungs below about t* fall back to
the dropped branch. At `pc_floor` 1e-6, t* is 2.0e-2, and a ladder whose
widest rung (2e-3) is inside it matches the product. A direct probe agrees
with the report: at t = 1e-4 and the default cut, Sir's smallest kept σ is
2.4474e-11 against ½ ℓ |u|² t² = 2.4473e-11, and at the floor Sir drops
both modes at t*/2 and one at 2 t*. A threshold on ℓ alone would have
refused PSB3's twist, whose ladder passes: its t* is past the ladder. The
table is the RTX 3080's; on the gfx1200 (`--preset hip`) the bend's ℓ is
1.7e-1 and t* at the floor 6.9e-3 (the random singlet lands on a
different direction under the CASSCF's run-dependent signs), the probe
agreeing to 1e-5 again, and the twist's ladder misses by 4.1e-8.

**Singlet directions.** Random CI directions shaped as SC's tests make them
(ċ_I = (r_I + 0.3) c_I) mix in quintets, along which the spin-null modes of
Sr and Si rise: every PC test moves c along a
singlet, H_κ c with a random spin-free H_κ.

**The search** (#803's note: `nh3_cs_sto3g` has no PC lifted null mode).
CAS(2,2) cannot lift: its perturbers' N ± k electron spaces are one or two
states, and every symmetric root reaches all of them. A PySCF probe over
symmetric CASSCF(4,4) roots, along a random singlet: H2O and H2CO lift
Sir's (28 → 31 kept), HCN lifts one (30 → 31), NH3 (Cs, every active orbital
A′) and ethylene none. In this tree's DF-CASSCF, H2CO's C2v STO-3G root
converges to a lower solution that breaks the symmetry (−112.4249 against
the symmetric −112.4124), so its metrics carry no null by symmetry; HCN's is
the symmetric root. Hence `test/gtest/data/nevpt2_hessian/hcn_sto3g.json`,
from `tools/nevpt2_hessian/` (its README has what PC's semi-numerical
Hessian does there).

## The inactive multipliers z^c

SC-NEVPT2 is not invariant under core–core and virtual–virtual rotations, so
the Lagrangian carries the pseudocanonical conditions f_pq = 0 there:
L = E + ⟨z^c, f⟩, f the generalized Fock

    f_pq = h_pq + Σ_rs D_rs [(pq|rs) − ½ (pr|sq)],   D = 2 on the core's diagonal, γ on the active block

whose core and virtual diagonals are ε above (F^I plus the active part).
In a state average (QD-NEVPT2) the core and virtuals are canonical over the
state-averaged Fock instead, γ_SA = Σ_r w_r γ_r on the active block, and
that is the f the conditions hold (below, "One root or a state average").
Along κ_pq = −κ_qp = t, p > q in one space, f_pq and f_qp each move by
t (ε_p − ε_q) where f is diagonal, and nothing else z^c reads moves at first
order (D is fixed in the MO indices, and the core and virtual densities are
invariant). So stationarity, A_pq + 2 z_pq (ε_p − ε_q) = 0, is one division
per pair, Park Eqs. 56–57 in this tree's conventions:

    z_pq = z_qp = −A_pq / (2 (ε_p − ε_q))     p ≠ q, both core or both virtual

with ⟨z, f⟩ the full Frobenius sum (each pair counted twice), A = X − Xᵀ
along C exp(κ) as `sc_adjoint` gives it. The −½ against Park's +½ is the
sign of A's κ and the doubled sum, and the stationarity ladder pins both.
**`zc_multipliers` takes A and ε as inputs**, so nothing in it is SC's: the
PC gradient (#564) calls it with PC's A, which vanishes on degenerate pairs
alone.

**Degenerate pairs.** |ε_p − ε_q| < 1e-6 Eh (`kZcGapFloor`) is a division
the state does not determine. When |A_pq| ≤ 1e-8 (`kZcZeroSource`), z is 0:
an abelian subgroup splits a degenerate set across irreps, the energy does
not move along a rotation mixing them, and 0 is the right representative.
Otherwise the pair is refused, `Numerical`, naming both orbitals, the gap
and |A|. Both are the demo's (`PC_GAP_FLOOR`, `PC_ZERO_TOL`;
`nuclear_gradients.tex`, "Degenerate (O2) pairs"). `min_gap` counts the
degenerate pairs too.

**Φ's adjoints** (`zc_adjoint`, at fixed z). B goes to the MO basis whole,
naux nmo² (z spans the virtual–virtual block, which nothing else here reads
off its diagonal), with g_L = Σ B_L,rs D_rs and w_L = Σ B_L,pq z_pq, D over
γ_SA:

    Φ   = ⟨z, h⟩ + Σ_L w_L g_L − ½ Σ_L tr(z B_L D B_L)
    h̄   = z
    Λ_L = z g_L + D w_L − ½ (z B_L D + D B_L z)
    W   = dΦ/dγ = Σ_L w_L B_L,aa − ½ Σ_L (B_L z B_L)_aa
    y^c_r = w_r Σ_tu (W + Wᵀ)_tu (E_tu c_r)
    X   = 2 h z + 2 Σ_L B_L Λ_L,   A = X − Xᵀ

Each left product D B_L and z B_L is one strided-batched gemm over the slow
index (B(·, ·, q) is a packed naux × nmo block), and every sum over (L, r)
one gemm with B read as (naux nmo) × nmo by its symmetry in the pair, as
the source term's half-transform does. Each γ_r is `ci::Sigma::rdm1` of its
root, summed on the host at its weight, and y^c one `ci::Sigma::apply` over
the one-body operator W + Wᵀ with no two-electron part, batched over the
roots, each root's column then scaled by w_r: spin-free, so each y^c_r is in
c_r's spin and symmetry sector.

**y^c is raw, as y is**: the derivative of Φ as a function of an
unnormalized c, not projected (⟨c, y^c⟩ = 2⟨W, γ⟩). `lagrangian_gradient`
passes ½ (y + y^c) to `mcscf::zvector`, which projects it off c; the ½ puts
it in that solver's convention (below). The (h̄, Λ, X, A) of Φ add to
`sc_adjoint`'s field by field.

**One root or a state average.** `zc_multipliers` takes A and ε and nothing
else, so it is state-agnostic: a state average passes its A and the SA
Fock's ε. `zc_adjoint` takes the roots and their weights (at least one,
non-negative, summing to 1 within 1e-12, else `InvalidConfig`), with the
vectors laid out as `ci::Roots::vectors`. One root at weight 1 is the
single-state Φ through the same operations: γ_SA is 0 + 1·γ and y^c's scale
is 1, both exact. It is not bitwise against the single-root `zc_adjoint` it
replaced, and cannot be: that one was not bitwise against itself, its
einsums' atomics reassociating each sum (over five calls on the gfx1200, Φ
apart by up to 1.4e-18, Λ 2.6e-18, y^c 3.5e-18, X and A 5.2e-18), and the
one-root call sits at the same spread (at most 6.9e-18; over 50 repeats,
1.39e-17 on A, one ulp of its largest element). h̄ = z alone is bitwise. `lagrangian_gradient` and
`sc_gradient`, which hold the CASSCF's roots, still refuse a state average
(`Unsupported`), and call `zc_adjoint` at one root, weight 1.

## The frozen core

Every energy and adjoint above, the single-state and QD gradients (every
form but the couplings) and `qd_inputs` take a trailing `nfrozen`, 0 by
default: the lowest `nfrozen` core columns of C are left uncorrelated
(#875; the Nevpt2 step's `frozen_core`, `pipeline/nevpt2/README.md`
"Frozen core"). Past the core is `InvalidConfig`.

**The functional.** The frozen orbitals leave every class with a core
index and stay in the core density, so the split is by reader:

- the builders (F^I's blocks, d, g, the orbital energies) run over the
  **whole** core, unchanged, so h1e, every `h1e_v_*` value and ε are the
  all-electron ones;
- the class walk reads the **correlated** frame: MO blocks over C's columns
  from `nfrozen` on (a second `mo_blocks`, `ScArrays::walk_blocks`), and
  e_core, `h1e_v_si` and `h1e_v_sir` from their correlated columns
  (`ScArrays::inputs`, pointer offsets: the core is each one's slow index).
  That is the Nevpt2 step's construction, so `qd_sc_energy` and
  `qd_pc_energy` give `qd_nevpt2_frozen_core.json`'s eigenvalues.

**The reverse pass is the same split backwards.** The walk adds its bars
onto the same offsets of the builders' bars (`energy_bars`), so the frozen
columns of ē_core, h̄1e_v_si and h̄1e_v_sir stay 0, and its Λ onto a buffer
in its own frame, naux (nmo − n_f)², which `builders_adjoint` first adds
onto the whole Λ's corner at (n_f, n_f) by one geam (`LambdaBars`). From
there h̄, Λ, X, A and ε are the whole frame's, as before.

**z^c is unchanged, and covers the new pairs (Park's "the pseudocanonical
condition also accounts for the frozen core approximation", acs.jctc.9b00919
after Eq. 25).** The core is canonicalized whole and then frozen, so f is
diagonal over the whole core and the pseudocanonical conditions f_ij = 0
are the Lagrangian's for every core pair. What the frozen core changes is A:

- **frozen–correlated** pairs: rotating one into the other moves an orbital
  into or out of the classes, so the energy is no longer invariant and
  A_ij ≠ 0. `zc_multipliers`' division, z_ij = −A_ij / (2 (ε_i − ε_j)), is
  their multiplier with no change: it already runs over every core pair,
  with ε the whole core's (its frozen entries the all-electron ones). The
  stationarity argument above holds as it is: f depends on the orbitals
  through the core and active densities alone, and a core rotation,
  frozen or not, leaves both.
- **frozen–frozen** pairs: the frozen orbitals enter only through the core
  density, which their rotation leaves alone, so A_ij = 0 and z = 0 (to
  round-off: 1.4e-19 and 6.5e-19 against 8.5e-3 on the frozen–correlated
  pairs, SC and PC, skew H2CO, below).
- **correlated–correlated** pairs: as without a frozen core.

So nothing new is written for z^c, Φ, the Z-vector, the gate or the passes:
the frozen core reaches them only through A, h̄, Λ, X and y. The gate's
core–core residue is the check on the division (2.0e-14 at skew H2CO,
1.3e-14 at ethylene Py, below).

**Measured** (`--preset default`, RTX 3080, the card held): directional
central differences of the step's own frozen-core energies, every point's
DF-SA-CASSCF re-converged (`conv_tol_grad` 1e-10, `conv_tol` 1e-13), steps
2e-3, 1e-3 and 5e-4 bohr, one Richardson step over the two finest, along
the analytic gradient and two random unit directions (`default_rng(875)`),
against the bar max(2 × the Richardson bar, 3 σ_E / h_min), σ_E the sd of
5 reference repeats (`../energy/README.md`, "The derivative across a
continuum"). Every check passes:

| case | quantity | σ_E (Eh) | gradient direction (slope; miss / bar) | random0 | random1 |
|---|---|---|---|---|---|
| skew H2CO SA3, cc-pVDZ, n_f 2 | SC-QD E_1 | 6.7e-13 | 0.2341; 1.1e-9 / 4.0e-9 | 7.1e-11 / 4.0e-9 | −9.2e-10 / 4.0e-9 |
| | PC-QD E_1 | 1.7e-12 | 0.2339; −7.1e-11 / 1.0e-8 | −2.1e-9 / 1.0e-8 | 9.6e-10 / 1.0e-8 |
| skew H2CO, one root, n_f 2 | SC E_0 | 2.1e-13 | 0.0281; 6.0e-9 / 2.8e-7 | −1.2e-10 / 7.5e-9 | 3.2e-8 / 2.6e-7 |
| | PC E_0 | 2.4e-13 | 0.0277; −1.3e-10 / 1.4e-9 | 1.5e-10 / 1.4e-9 | −3.8e-10 / 1.4e-9 |
| ethylene, Park's Py, aug-cc-pVDZ / aug-cc-pVDZ-JKFIT, SA3, n_f 2 | PC-QD E_0 + E_1 | 2.8e-13 | 0.0890; −1.1e-10 / 1.7e-9 | −9.7e-10 / 2.5e-9 | −1.7e-10 / 1.7e-9 |
| | PC-QD E_2 | 8.5e-14 | 0.2423; 4.6e-10 / 7.6e-10 | 4.8e-11 / 6.6e-10 | −3.0e-10 / 7.7e-10 |

**SC-QD at ethylene Py misses, frozen or not (#894).** E_0 + E_1 misses by
2.4e-9, 1.8e-8 and 1.5e-8 (bars 2.2e-9, 2.0e-9, 3.9e-9) and E_2 by
−2.3e-8, −1.7e-8 and −1.4e-8 (bars 1.2e-9); with `nfrozen` 0, the same
checks miss by 3.8e-9, 1.9e-8, 1.5e-8 and −2.5e-8, −1.9e-8, −1.5e-8. The
differences are clean (ratios 4.0, the two Richardson values within
1.4e-10), and the frozen core moves no miss by more than 1.4e-9: it is
the all-electron SC-QD gradient's at that geometry, not the frozen core's.
#894 has it; SC's non-smoothness where a perturber's norm is near zero
(the ladders' "Not every ladder is smooth" below) is the candidate. PC
passes there, and SC passes on skew H2CO.

The quantity at ethylene's MECI is E_0 + E_1, not either state: 7 µEh
apart, the two are a cone there, and a step of 5e-4 bohr opens the gap
twenty-fold. Their sum is smooth (tr H_eff less E_2), and its gradient is
the sum of the two states' (the intersection's g_P + g_Q). E_2 is checked
alone beside it.

## SC-QD-NEVPT2: the functional E[W] and its adjoint

`qd_sc_energy` is the SC-QD-NEVPT2 effective Hamiltonian's correlation
part (`nevpt2::sc_effective_hamiltonian`) as a function of (h, B, C) and
the roots c_1 … c_n at their SA weights; `qd_sc_adjoint` is the adjoint of
E[W] = Σ_MN W_MN H_MN along a fixed W, the first piece of the QD-NEVPT2
gradient (#702). The QD gradient's seed is W = u_K u_Kᵀ, u_K state K's
eigenvector of the symmetrized H_eff, so ½ (W + Wᵀ) = W on the
unsymmetrized couplings; the call takes any n × n W.

**The functional mirrors `sc_energy`'s over the shared orbitals.** One set
of MO blocks, B_cc, B_vv's diagonal, C^T h C, h2e, (tu|vw) and F^I serves
every pair (`shared_arrays`). The orbital energies are the **state-averaged**
Fock's diagonal, F^I + F^A(γ_SA), γ_SA = Σ_r w_r γ_r rebuilt from the roots
at every point (the Casscf step's `mo_energies` at its own orbitals, the
canonical ones a state average's NEVPT2 reads), so g and ε read γ_SA, which
sits in the shared arrays' dm1. Each ordered pair (M, N) has its own dm3:
`nevpt2::rdms`' one-vector build with f3ca and f3ac on the diagonal, the
bra/ket one off it (whose f3s are built into scratch and dropped, since QD
reads f3 on the diagonal only and the forward cannot skip them yet, #719);
its dm2 and dm1 are dm3's traces, as one root's are (Σ_t E_tt is N on the
ket). The n² pairs' RDMs are held at once, as
`sc_effective_hamiltonian` reads them. At one root, weight 1, every input
is `sc_energy`'s through the same operations (γ_SA is 1·γ, a copy).

**The reverse pass** is `sc_effective_hamiltonian_adjoint` (#698) at H̄ = W,
then `builders_adjoint` with a QD RDM step in place of one root's:

1. The walk adds the shared inputs' bars onto one set (the one-root pass's
   `EnergyBars`), each root's own RDMs' bars onto that root's set, and
   hands each transition pair's dm1, dm2 and dm3 bars over, pair at a time.
   For each, dm1̄ and dm2̄ fold into dm3̄ by the traces' adjoint, and the
   bra/ket `rdms_adjoint` (#697) with **null f3 bars** takes dm3̄ onto
   ȳ_bra and ȳ_ket before the pair's bars are freed. A transition dm3 does
   not read (tu|vw), so nothing goes onto it.
2. The shared builders run backwards (F^I's blocks, the orbital energies,
   g, d), so γ̄_SA collects in the shared dm1 bar.
3. Each root r's own dm1 bar takes w_r γ̄_SA, then the traces' adjoint and
   the one-vector `rdms_adjoint` take its own dm3̄, f3cā and f3ac̄ onto ȳ_r
   and (tu|vw)'s bar. So **every root takes ε's share through γ_SA**, its
   W_rr zero or not: ȳ_r ∋ w_r σ₁(γ̄_SA + γ̄_SAᵀ) c_r, by the trace and
   `rdms_adjoint` rather than a sigma vector, as the one-root pass does.
4. (tu|vw)'s builder backwards, then h̄, Λ, X, A and ε exactly as
   `sc_adjoint`'s.

ȳ_r is **raw**, at fixed (h, B): not projected, and not yet the
model-space source τ_sr = c_s·ȳ_r − c_r·ȳ_s, which reads ȳ after Φ^SA's
y^c (#701) is added; #704 forms it. `rdms_adjoint` overwrites its outputs,
so each call writes scratch that is added onto ȳ and (tu|vw)'s bar.

**Memory.** Beside one root's (above): the n² pairs' dm1, dm2 and dm3 and
the n diagonal pairs' f3s, as the forward holds them; n own-RDM bar sets
(dm1 … f3ac each); one pair's transition bars at a time, which the walk
allocates; two scratch f3s (n⁶ each) in the forward; and ȳ over every root
plus three scratch vectors. The shared arrays' own dm2, dm3, f3ca and f3ac
(and their bars) are allocated, unused, as `zeroed_arrays` gives every
caller.

## PC-QD-NEVPT2: the functional E[W] and its adjoint

`qd_pc_energy` and `qd_pc_adjoint` are their SC-QD namesakes with
`pc_effective_hamiltonian` and `pc_effective_hamiltonian_adjoint` (#703)
in place of `sc_effective_hamiltonian` and its adjoint, as `pc_adjoint` is
`sc_adjoint`'s (#713). The functional's inputs (the shared builders over
the SA Fock, every pair's RDMs, f3 on the diagonal) and the whole reverse
pass around the walk are one function in `qd.cpp` that both walks feed:
each adds onto the same `EnergyBars` and hands every transition pair's
dm1, dm2 and dm3 bars to the same callback, which takes them through the
traces' adjoint and the bra/ket `rdms_adjoint` onto ȳ_bra and ȳ_ket. PC
hands them over after its walk, all at once in row-major pair order,
where SC hands them over pair at a time; the callback is the same.
Element [M n + N] is ⟨M|V|Ψ_N⁽¹⁾⟩ per class over ket N's own solve, so
the matrix is not symmetric, and W need not be either.

**The dropped rank-projector terms.** PC's finish (design doc Eq.
qd-pcdot) differentiates the restricted inverse (K_N + Δ S_N)⁺ as if S_N's
kept span were fixed. That is exact when b_{M←N} lies in S_N's kept span,
which holds when S_N's dropped modes annihilate |N⟩, so at eigenstates.
#703's ladders move the RDMs along random tangents that carry no
eigenstate structure, so they cannot test it; the ladders here along each
c_r, real states moved along a real direction, are its check.
Since #853 the cut also drops real modes of S_N where they run through it
with no gap, which carry b_{M←N} components of their own small size: at
PSB3 CAS(6,6) (up to 22 modes per class dropped, about half real)
`qd_pc_gradient` matches re-converged central differences to
max(2 × the Richardson bar, 3 σ_E / h_min) on every Cartesian component at
FC and at 90°, and keeping the real modes does not move it beyond that
(`nevpt2.energy`'s README, "The derivative across a continuum"). One
component at 90° did not, until #862 (below).
`Nevpt2Psb3.PcQdGradientMatchesNumerical` holds FC's.

### The kept span's rotation (#862)

**The symptom.** At PSB3 twisted 90° (DF-SA3-CASSCF(6,6)/cc-pVDZ), atom 4
x of the PC-QD gradient missed its re-converged central difference by
9.6e-9, −1.37e-8 and −1.4e-9 Eh/bohr (states 0, 1, 2), cut-independent and
unmoved by the CASSCF's stopping point (`nevpt2.energy`'s README, "The
derivative across a continuum"). SA-CASSCF, SC-QD and single-state SC and
PC passed. The gate's active–active residue tracked it: 3.9e-10, 9.6e-11
and 4.5e-11 for PC-QD against 2.6e-12 to 3.7e-12 for every SC run, and
5.0e-11 for single-state PC.

**Why the active–active block.** PC-NEVPT2 is invariant under an
active–active rotation with the CI vectors counter-rotated, the cut
included: every class's S, K and S₁ transform covariantly in its active
basis (S → R S Rᵀ, R the rotation's representation there), so S's spectrum
and the cut do not move, and the kept span turns with the basis. The
PC-QD elements E_IJ are bilinear in two such rows and invariant the same
way. So the gate's active–active residue is not a missing condition or
multiplier: it is how far the reverse pass is from the derivative of an
invariant function, and the PC Lagrangian can be stationary there
exactly. SC is the check: its residue is the CI response's floor.

**The term.** The PC reverse pass differentiates each ket's solve
E = −c b_Iᵀ A⁺ b_J (A = K + Δ S over S's kept span) holding U, and so the
kept span, still. Under the joint rotation the span turns by
Θ = V_nᵀ G V_k (G the rotation's generator in the class basis), order one,
into the dropped modes V_n. That moves E only through the rows' parts along
V_n, n = V_nᵀ b and a = V_nᵀ A x, and both vanish for an exact null mode:
O_n|J⟩ = 0 makes ⟨J| O_n† S^{(J,I)} = 0 (no part of b_{I←J}), and at an
eigenstate K maps the kept span into itself. Since #853 the cut drops real
modes too, where ‖O_n|J⟩‖ = √σ_n, small but not zero. Then

- the own row's part, V_nᵀ S_J x = σ_n V_nᵀ x, is of order σ_n: negligible;
- K's part, V_nᵀ K U, is of order √σ_n: in single-state PC as well, which
  is its 5.0e-11;
- **the transition row's part, V_nᵀ S^{(J,I)} x = ⟨O_n J | O_x I⟩, is of
  order √σ_n**, and only the QD off-diagonals read a transition row.

A Python probe over the tree's own 90° orbitals (`casscf.molden`, a PySCF
DF-CASCI over them) found the transition rows' parts along the dropped
modes of Sr and Si at 2e-11 to 8e-9 (the norm over tuples) against the own
rows' 3e-15 to 7e-13;
every other class drops no real mode. Along a nuclear direction Θ divides
by σ_l − σ_n, which across a continuum with no gap at the cut is of the
modes' own size, so the term does not shrink with them: hence a miss that
moving the cut did not move.

**The fix** is that term, the eigenvectors' derivative restricted to the
kept span's rotation into the dropped modes (`nevpt2.energy`'s README,
"PC-QD", "The kept span's rotation"): one more bar on each ket's S,
V_n [(V_nᵀ Ū Uᵀ V_k) ⊘ (σ_l − σ_n)] V_kᵀ, Ū the walk bars' derivative with
respect to U. It is zero where nothing real is dropped, so every golden
case, whose dropped modes are null to round-off, moves by round-off. A pair
σ_l − σ_n within σ's round-off, ε √d of S's largest eigenvalue, is
skipped. A floor at the cut's own noise multiple (100 ε √d, 3.3e-13 at
d = 216) is too high: it skips every pair that straddles the cut, and left
state 0's residue at 4.1e-11 and its miss at 8.6e-9.

**Measured** (`--preset default`, RTX 3080, the card held; `conv_tol_grad`
1e-10). The gate's active–active residue and atom 4 x's miss against #900's
central differences (the forward is unchanged, so its energies stand; the
reference energies repeat to ≤ 1.1e-12), after the second Richardson level:

| 90° PSB3, cc-pVDZ | residue before | after | miss before | after | bar |
|---|---|---|---|---|---|
| PC-QD state 0 | 9.6e-11 | 8.1e-12 | 9.58e-9 | 8.98e-10 | 2.4e-9 |
| PC-QD state 1 | 3.9e-10 | 7.8e-12 | −1.37e-8 | 4.13e-10 | 2.2e-9 |
| PC-QD state 2 | 4.9e-11 | 3.9e-12 | −1.36e-9 | 7.45e-10 | 1.1e-8 |
| single-state PC | 5.0e-11 | 7.4e-12 | 5.84e-10 | 9.43e-10 | 1.1e-8 |

The bar is max(2 × the Richardson bar, 3 σ_E / h_min), σ_E from 5 reference
repeats with the fix (3.9e-13 and 3.7e-13 for states 0 and 1; #853's 1.8e-12
for the others, which no repeat here measured). State 0's miss before, 9.6e-9,
was under #853's bar of 1.1e-8 but four times this one. Directional checks at 90°
with the fix (`../energy/README.md`'s method; the gradient direction and
two random unit directions, `default_rng(862)`, steps 2e-3, 1e-3 and
5e-4 bohr, every point re-converged), miss / bar:

| 90° PSB3, cc-pVDZ | σ_E | gradient direction (slope) | random0 | random1 |
|---|---|---|---|---|
| PC-QD state 0 | 3.9e-13 | 0.1985; 2.0e-10 / 2.6e-9 | 7.3e-10 / 2.4e-9 | 4.8e-10 / 2.4e-9 |
| PC-QD state 1 | 3.7e-13 | 0.1310; −4.1e-10 / 2.2e-9 | −1.4e-10 / 2.2e-9 | −7.0e-10 / 2.2e-9 |

Re-checked with the fix by #875's driver and directions
(`default_rng(875)`), against the same bar, every check passes (miss / bar):

| case | quantity | σ_E (Eh) | gradient direction | random0 | random1 |
|---|---|---|---|---|---|
| ethylene, Park's Py, aug-cc-pVDZ, SA3, n_f 2 | PC-QD E_0 + E_1 | 1.4e-13 | 2.4e-10 / 8.5e-10 | −8.3e-11 / 8.5e-10 | 3.4e-10 / 1.1e-9 |
| | PC-QD E_2 | 6.8e-14 | 1.2e-10 / 4.1e-10 | 7.4e-12 / 4.1e-10 | −7.1e-11 / 4.1e-10 |
| skew H2CO SA3, cc-pVDZ, n_f 2 | PC-QD E_1 | 2.7e-13 | 5.5e-10 / 1.6e-9 | −7.4e-10 / 2.3e-9 | −1.6e-10 / 1.6e-9 |
| skew H2CO, one root, n_f 2 | PC E_0 | 6.9e-14 | 4.4e-10 / 1.1e-9 | −3.2e-10 / 8.2e-10 | −4.6e-10 / 1.3e-9 |

`Nevpt2Psb3.PcQdActiveActiveGateAtTheTwist` is the regression test: the
same twist in STO-3G (3 s on the RTX 3080, 14 s on the gfx1200; the
cheapest of its family, `slow_test.h`), PC-QD state 0's residue 3.5e-10 before
(three runs) and 3.3e-12 to 7.3e-12 after (four runs), against a bar of
3e-11; SC-QD there is 8.5e-13.

**Refusals.** A class `pc_effective_hamiltonian` refuses (no positive
eigenvalue of S_N, or a pole: λ + Δ at round-off or an intruder,
|y / (λ + Δ)| > 1; no sign test, `../energy/README.md` "Poles, not signs")
is refused by
`qd_pc_adjoint` with that error, kind and message naming the class and
the state, before any bar is read: never a matrix with the class summed
in as zero. PC's A is the one-root kind: its inactive blocks are not zero
in general, and vanish on a degenerate pair (#564's one-root measurement;
not tested here at n > 1), so `zc_multipliers` takes it as it takes
`pc_adjoint`'s.

**Memory.** SC-QD's (above), plus what `pc_effective_hamiltonian_adjoint`
holds: every ket's solve per class and the n² − n transitions' metric
bars until the walk ends (`nevpt2.energy`'s README, "PC-QD").

## The nuclear gradient

On a converged single-state CASSCF the Lagrangian is (`nuclear_gradient.tex`
Eq. L)

    L = E_cas + E_corr + Z · g_orb + z · g_ci + ⟨z^c, f⟩,

g_orb `response::fock`'s orbital gradient over the packed pairs and
g_ci = 2 (H₀ − E) c, with (Z, z) the Z-vector and z^c above. Everything but
(Z, z) comes in: `lagrangian_gradient` takes E_corr's adjoint (`corr`, an
`ScAdjoint`), z^c and Φ's adjoint, so the same assembly takes any
correlation energy's adjoint in `sc_adjoint`'s shape; `sc_gradient` is
`sc_adjoint`, `zc_multipliers` and `zc_adjoint` at the solution's orbitals
and root, then it.

1. **corr + Φ**, field by field, in place: h̄, Λ, y, X and A. A and y are
   now Eq. cp's sources A^corr + A^{z^c} and y + y^c.
2. **The Z-vector.** The CI-relaxed Hessian at the solution, built once as
   `mcscf::df_gradient` builds it for its own Z-vector: `casscf_integrals`,
   H₀'s sigma, `response::fock` at the root's RDMs, `response::Hessian` over
   it, `ResponseIntegrals` and `RelaxedHessian` at weight 1. Then
   `mcscf::zvector` with the orbital right-hand side g + pack(A) (g the
   Fock's own gradient, below the CASSCF's convergence) and the CI one
   **½ (y + y^c)**. zvector's CI row is (H₀ − E) z + P H′(Z) c = −P y_r, the
   coordinates of PySCF's `gen_g_hop`; the tex's J_cc = 2 P (H − E) P and
   its J_cκ are each twice that row, so for the same (Z, z) its right-hand
   side P(y + y^c) is halved (`src/mcscf/zvector/README.md`, "The CI
   right-hand side"). zvector projects it off c. (Z, z) come out in
   zvector's coordinates, which is what `df_gradient` takes, so nothing is
   mapped back.
3. **The gate**: X^L symmetric in every block. Its antisymmetric part is

       A^L = 2 (F − Fᵀ)(γ + γ_T, Γ + Γ_T) + A^Z + A,

   F the generalized Fock (`response::fock`) at the root's RDMs plus z's
   transition RDMs (E_cas's and z · g_ci's terms, half the difference of
   `rdm12` of c ± z as `df_gradient` takes them), A^Z = 2 (F′ − F′ᵀ) with
   F′ the derivative of the root's F along C (1 + ε κ(Z)) (Z · g_orb's
   term), and A step 1's. A^Z is the orbital Hessian's product with Z over
   *every* pair, the redundant ones included, which `response::Hessian`
   does not give: the Hessian is symmetric at the solution (the full
   gradient, redundant blocks too, vanishes there), so the derivative of
   the gradient along Z is that product. F is a polynomial of degree 4 in C
   (C h C and the DF Coulomb and exchange of the core density, (pu|vw)), so
   the five-point stencil of F at ε = ±s, ±2s, max |s κ| = 0.1, is its
   derivative exactly, to the quotient's round-off: four more
   `casscf_integrals` and Focks, no step to tune. A block past 1e-8 is
   refused, `Numerical`, naming it: on the non-redundant pairs A^L is the
   Z-vector's residual, on the core–core and virtual–virtual pairs z^c's
   division (and F's symmetric core block), on the active–active pairs the
   gauge identity, which only the CI equation's residual moves. So a wrong
   factor in any of the terms names its block.
4. **The nuclear term**: one `mcscf::df_gradient` at root 0 with (Z, z) as
   its `multipliers` and (h̄, Λ, X) as its `extra` MO adjoint: E_cas's, Z's
   and z's terms are the ones it builds at the multipliers, corr's and
   z^c's the extra adjoint. One derivative-integral pass.

### The Z-vector's floor

Every Lagrangian here, single-state, QD state, W form, several W or
coupling, goes through one solve (`lagrangian_zvector`). It runs to a
tolerance, 1e-10 on ‖H̃ z + g‖ (the single-state `lagrangian_gradient`,
`sc_gradient` and `pc_gradient` take it as `zvector_tolerance`, 1e-10 by
default, as the W-form QD ones do; `[hessian]`'s ladder tightens it to
measure what it leaves, #812), in at most 200 products. Its **true**
residual, from one more product, is held to that same tolerance: for one
column as for several, `Numerical` naming the column. Before #864 only the
several-column path checked, and a state gradient accepted 3.8e-10 at PSB3
cc-pVTZ unseen.

**What set the floor (#864).** The PSB3 QD MECI was refused at evaluation 0,
far from any seam, with "column 0 of 3 has true residual 2.546e-10 after 31
products": the recurrence met 1e-10 and the true residual did not follow.
The 200-product cap was not reached, and neither was the CI response's own
200-sigma cap. The cause is the CI response inside each H̃ product. It was
solved to `RelaxedHessian`'s default 1e-8 of its right-hand side, so each
product was off by about 0.04 ‖z‖ × 1e-8, and z with it. Measured on the
RTX 3080 (CUDA), one more product at a response of 1e-13 standing in for
the exact one:

| System | Basis | ‖z‖ | Products | Recurrence | True (1e-8) | Exact |
|---|---|---|---|---|---|---|
| ethylene, Park's SI C3v, PC-QD state 0 | cc-pVDZ | 0.057 | 13 | 2.07e-11 | 2.07e-11 | 2.07e-11 |
| PSB3 twisted 90°, SC-QD MECI, column 0 | cc-pVDZ | 0.79 | 31 | 9.8e-11 | 2.55e-10 | 2.19e-10 |
| PSB3 twisted 90°, SC-QD state 0 | cc-pVTZ (292 bf) | 0.84 | 32 | 8.1e-11 | 3.84e-10 | 2.42e-10 |

The floor grew with ‖z‖ (ethylene 0.06, PSB3 0.8) and so with the system.
At a response tolerance of 1e-10, every true residual equals its recurrence
to under 0.2%, in the same products (cc-pVDZ 9.83e-11, cc-pVTZ 8.09e-11).
At 1e-12 nothing changes again (on PSB3; it is not safe everywhere,
below). The cc-pVTZ run took 37 s at 1e-8 and 38 s
at 1e-10.

**The rule.** `RelaxedHessian` is created with the response at
`mcscf::response_tolerance_for` the Z-vector tolerance, 1e-2 of it and at
least 1e-10 (1e-10 at the default, where the true residual measured within
0.2% of the recurrence at PSB3; tighter breaks on round-off where the roots
nearly span the CI space, `mcscf/intersection/README.md`). The bar itself
stays 1e-10, absolute. The multipliers feed the gradient, off by
at most ‖r‖ ‖H̃⁻¹‖ ‖∂g/∂R‖ ≈ 1e-9 Eh/bohr, and in MECI mode s_o and s_c,
norms of the multipliers, off by about 1e-9 relative. With the floor gone,
nothing measured grows with the system: 31 and 32 products from cc-pVDZ to
cc-pVTZ, and ‖g‖ is 0.07 to 0.5 in every case. So a bar relative to ‖g‖
would buy nothing. `mcscf/intersection/README.md` "The floor under the true
residual" has the CASSCF intersection's side (#858), the same mechanism.

**One parameter.** The W-form and several-W `qd_sc_gradient` /
`qd_pc_gradient` take it as their trailing `zvector_tolerance` (1e-10 by
default), the solve's tolerance and its check's together.
`nevpt2::QdIntersectionOptions::zvector_tolerance` passes it through, and
the intersection holds each W's true residual to the same value. The state
forms, the couplings and the single-state gradient use 1e-10. It is a
runtime parameter only, not a TOML key.

**Measured at PSB3 with it** (`--preset default`, RTX 3080, the card held,
`conv_tol_grad` 1e-10): the SC-QD MECI (cc-pVDZ, from the twisted
start) runs its four allowed steps, gap 1.9e-2 to 8.4e-5. One cc-pVTZ step
passes evaluations 0 and 1. The PC-QD MECI (cc-pVDZ, `conv_tol_grad` 1e-9 per
#867) runs its three allowed steps, gap 2.0e-2 to 7.9e-4.

`ScGradient` carries max |z^c| and the smallest inactive gap (SC's
near-degeneracy hazard), the PCG iterations and residual (the CASSCF
Hessian's conditioning), and the gate's six residues, to report with every
gradient (`nuclear_gradient.tex`, "Hazards").

## PC-NEVPT2

`pc_energy`, `pc_adjoint` and `pc_gradient` are their SC namesakes with
`pc_class_energies` and `pc_class_energies_adjoint` in place of
`class_energies` and `class_energies_adjoint`, and nothing else: the
builders, their reverse pass, `rdms_adjoint`, the density-fitted adjoints,
the symmetrization and the source term are one function (`builders_adjoint`
in `gradient.cpp`) that both class walks feed, since PC's reverse pass adds
onto the same `EnergyBars` (its S, K and S₁ bars go onto SC's block bars,
and SC's sweep takes them; `nevpt2.energy`'s README, "The PC reverse pass").
`pc_gradient` then hands PC's arrays to `lagrangian_gradient` in an
`ScAdjoint` and takes the result's fields into a `PcGradient` beside PC's
class energies. The Z-vector, the gate and the nuclear term are unchanged.

**z^c is the same division as SC's.** PC is invariant to a rotation of two
inactive orbitals only when their orbital energies are equal: every class
sums a quadratic form in its tuples' rows over every ordered core and
virtual index, but each tuple at its own Δ, so a rotation that mixes two
orbitals of different ε moves the energy. With the canonical orbitals'
distinct ε, PC's A on the core–core and virtual–virtual blocks is not zero,
and `zc_multipliers` takes it as it takes SC's (demo
`docs/gradient-theory.md` §4.4: "the z^c divisions, which are the same on
both branches"; #561 measured the class-level derivative at 1e-17 of its
scale with ε forced equal, and not zero with ε distinct). What PC does have
is that its A vanishes on a degenerate pair, where SC's does not: so a
pair that the degenerate-pair rule refuses for SC takes z = 0 for PC
(`Nevpt2PcZc` below).

**Refusals.** A class PC refuses (no positive eigenvalue of the metric, a
pole: λ + Δ at round-off or an intruder, |y / (λ + Δ)| > 1) is refused by
`pc_class_energies_adjoint`
with the energy's error, and `pc_gradient` returns it unchanged: never a
gradient with that class summed in as zero.

## The QD-NEVPT2 gradient

`qd_sc_lagrangian(res, molecule, basis, aux, h, b, solution, ncore, ncas,
state, batch)` is the SC-QD-NEVPT2 **Lagrangian and Z-vector** of QD state
K = `state` (H_eff's eigenvalues ascending) on a DF-SA-CASSCF whose core
and virtuals are canonical over the SA Fock, as the Casscf step leaves them
(`nuclear_gradient.tex` Algorithm qdgrad, steps 1–8). The derivative passes
are `qd_sc_gradient`'s (below); `QdScLagrangian` hands them everything they
read. With W = u_K u_Kᵀ and ρ_r = W_rr = u_rK², the Lagrangian is

    L = E[W] + Z · g_SA + Σ_r 2 w_r z_r · (H − E_r) c_r
        + Σ_{r<s} (2 (w_r − w_s) Θ_sr + ζ_sr) ⟨c_s|H|c_r⟩ + ⟨z^c, f^SA⟩,
    E[W] = Σ_r ρ_r E_r + Σ_MN W_MN C_MN,

(Z, z, Θ) in `mcscf::zvector`'s coordinates: its CI rows are the SA
Hessian's over 2 w_r, its rotation rows over 2 (w_r − w_s), so the
condition ⟨c_s|H|c_r⟩ = 0 of an unequal pair carries 2 (w_r − w_s) Θ_sr,
and an equal pair, which E_SA is blind to, carries ζ_sr of its own.

1. `qd_sc_energy`, then H_eff = diag(E_r) + ½ (C + Cᵀ) (E_r the solution's
   root energies) by `diagonalize`, and W. A QD state within
   `mcscf::kRootGapFloor` of another is refused, `Numerical`, naming both:
   u_K has no derivative there.
2. `qd_sc_adjoint` at W; `zc_multipliers` over its A and the SA Fock's ε;
   `zc_adjoint` at the solution's weights, every root's vector; the two
   added field by field. So ȳ_r = ȳ^corr_r + y^c_r, and A, h̄, Λ, X are
   E^corr[W] + Φ^SA's (`corr`, the `extra` of `qd_sc_gradient`'s `df_gradient`).
3. `mcscf::root_pair_multipliers` on the summed ȳ: τ_sr, ζ on the equal
   pairs, t on the unequal ones; `mcscf::zeta_terms` on ζ: ζ · g and the
   folded (γ^ζ, Γ^ζ) of its `df_transition_gradient`.
4. The Z-vector, over the CI-relaxed Hessian at the solution and its
   weights (built as `lagrangian_gradient` builds it, the Fock at the
   weighted RDMs): orbital right-hand side g^ρ + ζ · g + pack(A), g^ρ
   `response::fock`'s gradient at the ρ-weighted RDMs; CI right-hand side
   **ȳ_r / (2 w_r)** for every root (zvector projects it off every root,
   P ȳ_r / (2 w_r)); rotation right-hand side t. At n = 1 this is
   `lagrangian_gradient`'s ½ (y + y^c).
5. The gate (`lagrangian_gradient`'s, step 3 above), its CAS part at the
   ρ-weighted RDMs plus ζ's, z's transition RDMs at the weights with Θ
   folded into z as its part along the roots (z_r += Θ_sr c_s, as
   `df_gradient` takes them), and A^Z at the weighted RDMs. A block past
   1e-8 is refused.

The weights are the solution's, `mcscf::sa_gradient_weights`', everywhere:
ε and Φ (step 2), τ's split, the Z-vector and the gate. A root of weight
below `mcscf::kRotationWeightTolerance` is refused, `InvalidConfig`: it has
no SA CI row, and step 4 divides by w_r.

**The 1/(2 w_r)** was read off zvector's README (its CI rows are the SA
Hessian's divided by 2 w_r) and not derived against a test before; the
stationarity ladders below are what pin it: along a CI direction off the
roots, ∂L/∂c_r = P ȳ_r + 2 w_r [P H′(Z) c_r + (H − E_r) z_r], zero exactly
when zvector's row holds with ȳ_r / (2 w_r) on its right.

`qd_sc_gradient(res, molecule, basis, classes, aux, h, b, l, solution,
ncore, ncas, state, batch)` is the **nuclear gradient** dE_K/dR (Algorithm
qdgrad's derivative passes): `qd_sc_lagrangian`, then two passes, each over
one evaluation of the derivative integrals.

1. `mcscf::df_gradient` at the mix ρ (Σ_r ρ_r dE_r/dR), with (Z, z, Θ) as
   its multipliers (Θ folded into z along the roots, z's transition RDMs at
   the weights, the orbital response along Z at the weighted RDMs) and
   `corr`'s h̄, Λ and X, E^corr[W] + Φ^SA's, as its extra.
2. `mcscf::df_transition_gradient` at (γ^ζ, Γ^ζ), no multipliers and no
   extra: the Hellmann–Feynman gradient of Σ_{r<s} ζ_sr ⟨c_s|H|c_r⟩, whose
   core part multiplies ⟨c_s|c_r⟩ = 0. Skipped when every ζ is zero (every
   pair of unequal weights); `transition` is then zero.

`total` is their sum. The report is `ScGradient`'s, through `lagrangian`
(max |z^c| and the smallest inactive gap, the PCG iterations and residual,
the gate's residues, H_eff's eigenvalues), plus max |ζ|, the smallest gap
between CASSCF roots (ζ and t divide by it) and H_eff's gap at K (u_K's
derivative divides by it). `qd_sc_lagrangian`'s refusals come back
unchanged.

### At a degenerate CASSCF pair (#892)

Step 3 refuses when two CASSCF roots are within `mcscf::kRootGapFloor`
(`../../mcscf/gradient/README.md`, "Holding the roots apart"). It is the
stop that ended #847's all-electron QD-PC/aug-cc-pVDZ walk to ethylene's
C3v MECI. At evaluation 10 the CASSCF ¹E pair was 7.9e-9 Eh apart, while
H_eff's pair was 6.6e-10 apart. The refusal stays. **The state gradient
dE_K/dR does not exist at a degenerate reference pair.** For SC it diverges.
For PC it is bounded but depends on the direction of approach. The
measurements:

**How.** A PySCF reference over `tools/emit/qd_nevpt2_reference.py`'s
`sc_qd` and `pc_qd` (PySCF has no QD-NEVPT2). It uses the same DF-SA3-CASSCF(2,2)/
aug-cc-pVDZ (aug-cc-pVDZ-JKFIT), its active pair the CH carbon's two
2p⊥ (AVAS), and SA-canonical inactive orbitals. The CASSCF pair is rotated
by θ (c₀′ = cos θ c₀ + sin θ c₁, c₁′ = −sin θ c₀ + cos θ c₁), and H_eff is
rebuilt over the rotated roots: their RDMs, their Dyall denominators, and
the zeroth order Uᵀ diag(E) U. With equal weights the SA-CASSCF, its
orbitals and the SA Fock do not move under θ, so only the state-specific
parts do. τ_K = ∂E_K/∂θ at θ = 0, the CASSCF roots, is
`root_pair_multipliers`' right-hand side for state K, and ζ_K = −τ_K / (E₁ −
E₀). At Park's geometry the reference gives a CASSCF gap of 2.112e-6 and a
PC H_eff gap of 7.154e-7. The tree's evaluation 0 gave 2e-6 and 7.154e-7.

**1. Does the energy depend on the rotation?**

| geometry | CASSCF gap | PC: E₀, E₁ over θ | PC: E₀ + E₁ over θ | SC: E₀, E₁ over θ |
|---|---|---|---|---|
| Park's (SI p. S3) | 2.1e-6 | 4.0e-8 each | 8e-13 | 1.1e-6, 4.4e-6 |
| its exact C3v symmetrization | 1.2e-10 | **3e-15** (gap ≤ 5.1e-11 at every θ) | — | gap 4.1e-7 to 1.2e-5 |

PC is invariant to the rotation at the symmetric pair. A degenerate
reference pair is no defect of PC at a point of symmetry: its two
QD states stay degenerate whatever pair the CASSCF hands it. SC is not
invariant, and it splits the ¹E pair. The exact-C3v SC row is also SC's
non-invariance to the degenerate e core pair (`kZcGapFloor`, above), which
PC does not have. But SC's split survives off the symmetric point: at
1e-4 Å from it, SC's H_eff gap is 6.5e-7 against a CASSCF gap of 4.9e-8.

**2. Does τ vanish with the gap?** Approaching the exact-C3v point along
d₁ (towards Park's geometry) and along d₂ (a fixed random direction, seed
892), with τ by a five-point difference in θ (h = 0.01):

| | CASSCF gap | PC ζ₀ | PC τ₀ + τ₁ | SC ζ₀ |
|---|---|---|---|---|
| d₁, 1e-3 Å | 4.86e-7 | 0.0106 | −3.9e-12 | 5.5 |
| d₁, 1e-4 Å | 4.93e-8 | 0.0113 | 2.7e-12 | 57 |
| d₂, 1e-3 Å | 4.41e-6 | −3.3e-4 | −3.7e-12 | −1.5 |
| d₂, 1e-4 Å | 4.41e-7 | 3.7e-4 | −4.5e-12 | 7.4 |

On a line to a CASSCF gap of 2.2e-9, SC's ζ reaches 1.1e3 and 1.5e3. SC's
τ stays between 1e-6 and 7e-6, so its ζ grows as 1/gap, with no limit.
PC's τ falls with the gap (5.2e-9 down to 1.6e-10), so its ζ is finite.
But its limit is
0.011 along d₁ and under 4e-4 along d₂. ζ_K multiplies the pair's
transition gradient, which at a CASSCF intersection is its interstate
coupling: that difference is a different dE_K/dR. Only PC's τ₀ + τ₁ vanishes
from both directions, to the five-point difference's noise. So the pair's
mean gradient ḡ = ½ (g_P + g_Q) has a limit, and each g_K alone does not.

**3. Park.** His C3v geometry is not exactly C3v. Its three C–H bonds
spread 5.3e-4 Å, its H–C–C angles 0.31°, and the lone H sits 7.5e-4 Å off
the C–C axis (aug-cc-pVTZ: 5.1e-4 Å, 0.04°, 4.5e-4 Å). There the CASSCF
gap is 2.1e-6 Eh and PC's H_eff gap is 7.15e-7, his reported 7e-7. So his
search stopped about 200× above this floor in CASSCF gap, and it never met
the degenerate pair. BAGEL's own treatment of the pair was not inspected.

**What to do instead.** At a geometry where a symmetry makes the CASSCF
roots of the QD pair degenerate (an E pair; the QD intersection is then
pinned to it), take energies there, not gradients: QD single points at a
fixed structure (Park's, or the CASSCF MECI). This is what
`docs/reproduction/park2019/tier1.md` reports for C3v. A walk stopped by
this refusal has reached a QD-degenerate point (#847's: H_eff gap 6.6e-10)
whose forces are not converged. It is not an MECI. Use SC there with
care, if at all: its pair is split by about 1e-6 Eh near the symmetric point.
A PC-only MECI treatment built on ḡ alone (point 2) is not done. It would
need `../intersection/` and `geometry::optimize_meci` to work without
g_Q − g_P and h_PQ, whose own ζ parts depend on the direction of approach.
`../intersection/README.md`, "A MECI inside a degenerate CASSCF pair"
(#902), has the analysis: ḡ's limit, why the plane has none and the walk
does not need one at C3v, and the algorithm and checks it would take.

### At any symmetric W

`qd_sc_lagrangian(…, w, batch)` and `qd_sc_gradient(…, w, batch)` (#745),
the **W form**, take any symmetric W (n² row-major over the solution's n
roots) in place of `state`: the Lagrangian of E[W] and its gradient G(W).
The state form is a thin wrapper over it: step 1 above (`qd_sc_energy`,
H_eff, its eigenvectors and the degenerate-pair refusal, the one
state-specific part) builds W = u_K u_Kᵀ and hands it on, so steps 2–5 and
the passes are one code path. `eigenvalues` is H_eff's from the state form
and empty from the W form, whose `heff_gap` is infinite.

Everything in L is linear in W (`nuclear_gradient.tex`, "The QD-NEVPT2
energy functional") but the core's energy, which E[W] = Σ_r ρ_r E_r + … carries
at tr W. So nothing from H_eff is read, ρ = diag W, and:

- the RDMs at ρ, Σ_r ρ_r (γ_r, Γ_r), are taken root by root (`rdm12` takes
  only weights ≥ 0 summing to 1);
- g^ρ is the Fock's gradient at those RDMs, which is affine in them and
  carries its zero-RDM (core) part once, plus (tr W − 1) more of that part;
  the gate's Fock the same. Within 1e-12 of 1, tr W is taken as 1;
- the first pass is one `mcscf::df_gradient` at the mix ρ when ρ sums to 0
  (its core-off path, #741) or to 1 with no negative entry (a state's ρ =
  u_rK², always). Any other ρ is tr W p + d: p ≥ 0, ρ's positive part
  normalized to 1 (uniform when it has none), one `df_gradient` with zero
  multipliers and no extra (no response) scaled by tr W; and d = ρ − tr W p,
  summing to 0, one `df_gradient` with the multipliers and the extra, so
  that each is counted once.

So G(W₁ + W₂) = G(W₁) + G(W₂), and Σ_K dE_K/dR = G(1). A W not n² long,
or not symmetric to 1e-12 (|W_MN − W_NM|), or not finite, is refused,
`InvalidConfig`. The consumers are the QD couplings' h_PQ at W^PQ, trace 0
(#747, and PC's #748, whose W form is below).

### The interstate coupling h_PQ

`qd_sc_coupling(res, molecule, basis, classes, aux, h, b, l, solution,
ncore, ncas, p, q, batch)` (#747) is the SC-QD-NEVPT2 **interstate
coupling** between QD states P ≠ Q (H_eff's eigenvalues ascending), Park
(2020) Eq. 47's ∂L_PQ/∂X:

    h_PQ = R_Pᵀ (∇H_eff) R_Q = G(W^PQ),   W^PQ = ½ (R_P R_Qᵀ + R_Q R_Pᵀ),

R H_eff's eigenvectors at the reference. It is the W form above at W^PQ,
behind the state form's step 1: `qd_sc_energy`, H_eff, R, then W^PQ. One
file-local function builds both, with the pair (K, K) for a state: at
P = Q, W^PQ is u_K u_Kᵀ bit for bit (½ (x + x) = x), so the state form is
unchanged. tr W^PQ = R_Pᵀ R_Q = 0, so ρ = diag W^PQ sums to 0, g^ρ carries
no core, and the first pass is one `mcscf::df_gradient` on its core-off
path (#741): h_PQ's reference part Σ_r R_rP R_rQ ∇E_r has no core energy and
no nuclear repulsion. d^NEV_PQ = h_PQ / (E_Q − E_P) follows (#749).

- **The fold.** `fold`, empty by default, is called once with H_eff's
  eigenvalues and R after the diagonalization, and returns a `QdFold`:
  active-space RDMs (`gamma`, `gamma2`, ncas² and ncas⁴) and an orbital
  source `a` (nmo² column-major, or empty). They join the Lagrangian as
  they are: the RDMs to ρ's in g^ρ (which is affine in them, so they add
  their transition orbital gradient, the core off) and in the gate, and to
  ζ's folded RDMs of the transition pass (which then runs even at ζ = 0);
  `a` to A = X − Xᵀ, the right-hand side's packed pairs and the gate. One
  Z-vector then solves for h_PQ's right-hand side and theirs together, and
  `total` is h_PQ plus their Hellmann-Feynman and response terms. That is
  Park (2020) Eqs. 51–52's fold of the derivative coupling, which
  `../nac/`'s `qd_sc_nac(…, fold = true)` builds from
  `mcscf::nac_combination_terms`. The fold's RDMs are no state's, so
  their own active-active asymmetry 2 (F − Fᵀ) (the core off), which no
  multiplier removes (it is the CSF term's), is taken out of the gate's
  source in that block alone, by two Fock builds. An empty `fold` runs the
  same operations as before it existed.
- **Signs.** Each column of R is signed so that its largest-magnitude
  component is positive, the Nevpt2 step's convention (`pipeline/nevpt2`),
  in the solution's roots' phases, and `lagrangian.eigenvectors` returns it.
  h_PQ carries R_P's sign times R_Q's, and so the roots' phases only through
  them: a consumer pairing h_PQ with anything else built from R (#749's
  T^PQ) takes R from the same result. W^PQ is symmetric in P and Q element
  by element, so h_QP = h_PQ to the derivative passes' atomic sums.
- **Gaps.** A pair with |E_Q − E_P| under `mcscf::kRootGapFloor` is refused,
  `Numerical`, naming both states, before any adjoint is built: d^NEV
  divides by that gap, as the state's gradient refuses a K degenerate with
  another state. P = Q (a state's, `qd_sc_gradient`) and a state not one of
  the roots' are `InvalidConfig`. The other refusals are the state form's.
- **The result** is `QdScGradient`, as the state's: `total` is h_PQ
  (n_atom × 3), `lagrangian` the Lagrangian at W^PQ with H_eff's eigenvalues
  and R, `heff_gap` |E_Q − E_P|, `root_gap` the CASSCF roots' smallest gap
  (ζ and t divide by it), `max_zeta`, and the Z-vector's iterations and
  residual in `lagrangian.multipliers`. PC's (#748, `qd_pc_coupling`) is
  the same assembly with `qd_pc_energy` and `qd_pc_adjoint` (below).
- **Cost.** One QD gradient: the n² pairs' RDMs and E^corr[W]'s adjoint, one
  Z-vector, and the two derivative passes. The states' gradients and the
  couplings share nothing between calls.

`qd_pc_gradient(…)`, same arguments, is the **PC-QD-NEVPT2** gradient
(#705): the same Lagrangian and the same two passes, one file-local
assembly behind both entry points, with `qd_pc_energy` in step 1 and
`qd_pc_adjoint` (#713) in step 2. Nothing else in it is SC's: z^c is the
same division over PC's A, which vanishes on a degenerate inactive pair
as the single-state PC's does (`Nevpt2PcZc`); τ, ζ, t, the 1/(2 w_r) CI
right-hand side, the gate and the passes read only the arrays. H_eff,
W and u_K are PC's. The result is `QdPcGradient`: `QdScGradient`'s fields,
its `lagrangian` in `QdScLagrangian`'s fields (`corr` qd_pc_adjoint's
arrays plus Φ^SA's, its `elements` empty), and PC's own elements beside
it. A class PC refuses is `qd_pc_energy`'s error, returned unchanged before
any adjoint is built: never a gradient with the class zeroed. The
Lagrangian's own refusals (a root of zero weight, a degenerate pair of
PC's H_eff, the gate) are SC's, named `nevpt2::qd_pc_gradient`. The
restricted inverse's dropped rank-projector terms are #713's, checked
there by the c_r ladders.

### PC at any symmetric W, and PC's interstate coupling

`qd_pc_gradient(…, w, batch)` (#748) is PC's **W form**, G(W) of PC's
E[W]: SC's W form with `qd_pc_adjoint` at W in step 2, the rest the same
file-local code (ρ = diag W, the core at tr W, the one or two
`df_gradient` passes). There is no `qd_pc_lagrangian`; the Lagrangian is
the result's `lagrangian`, and `elements` are `qd_pc_adjoint`'s. The state
form is a thin wrapper over it, as SC's is: step 1 (`qd_pc_energy`, PC's
H_eff, its eigenvectors and the degenerate-pair refusal) builds
W = u_K u_Kᵀ and hands it to the same assembly. So G(W₁ + W₂) = G(W₁) +
G(W₂), and Σ_K dE_K/dR = G(1) over PC's states. A refused class is
`qd_pc_adjoint`'s error at any W; the other refusals (W's size, symmetry,
finiteness) are SC's W form's, named `nevpt2::qd_pc_gradient`.

`qd_pc_coupling(res, molecule, basis, classes, aux, h, b, l, solution,
ncore, ncas, p, q, batch)` (#748) is the **PC-QD-NEVPT2 interstate
coupling**, h_PQ = R_Pᵀ (∇H_eff) R_Q = G(W^PQ) with H_eff, R and G PC's:
`qd_sc_coupling` with the file-local pair function given PC's energy and
adjoint, R signed the same way (largest-magnitude component positive), the
same core-off pass at tr W^PQ = 0. The result is `QdPcGradient`: `total`
h_PQ, `lagrangian` at W^PQ with PC's eigenvalues and R, `heff_gap`
|E_Q − E_P|, and PC's elements. A class PC refuses refuses the coupling
with `qd_pc_energy`'s error, before any adjoint is built, as it refuses the
gradient; P = Q, a state not one of the roots', and a pair within
`mcscf::kRootGapFloor` are SC's refusals, named `nevpt2::qd_pc_coupling`.

### Several W's at once

`qd_sc_gradient(…, ws, batch)` and `qd_pc_gradient(…, ws, batch)` (#787)
take a list of k symmetric W's (`std::span<const std::vector<double>>`, each
n² row-major) and return one G(W) per W, in order: what
`nevpt2::qd_*_intersection` takes at u_P u_Pᵀ, u_Q u_Qᵀ and W^PQ. They share
everything that does not depend on W:

- `qd_setup` (the weights and the active strings), once;
- the adjoint's forward arrays: the shared builders, every ordered pair's
  RDMs (dm3 by `nevpt2::rdms`) and γ_SA's builders, once, the reverse pass
  (the class walk at H̄ = W, the RDMs' adjoints, the builders backwards)
  per W over them. The forward arrays are only read, so each W's adjoint is
  the one-W call's;
- `casscf_integrals`, the state-averaged Fock, and the CI-relaxed Hessian
  (H₀'s sigma, `response::Hessian`, `mcscf::RelaxedHessian`), once;
- one `mcscf::zvectors` (Schur) over the k columns of right-hand sides
  (orbital g^ρ + ζ · g + A, CI ȳ_r / 2 w_r, and t), in place of k
  `mcscf::zvector` solves. Each column's true residual, from one more
  product (`ZVectors::residuals`), must be at or below the solve's
  tolerance (`zvector_tolerance`, 1e-10 by default), or the call is
  `Numerical` naming the column: the block conjugate gradients stop on
  their recurrence (#786, and `mcscf::df_intersection`'s check, #791). The
  one-column path holds its true residual to the same bar ("The Z-vector's
  floor").

Per W, as the one-W form: z^c and Φ^SA, τ split into ζ and t, ζ's terms,
the gate on the W's own column, and the derivative passes. One W in the list
is the one-W form's path exactly (`mcscf::zvector`, not `zvectors`), which
is what the one-W form calls. The first W with an error returns it.

**The derivative passes stay one per W.** Stacking them, so the three-center
and metric derivative contractions run once over several density sets,
would need a multi-density derivative kernel: #773 (PR #776) measured one
`geri3` kernel at ~50 ms of each ~57 ms `df_gradient` call, so a stacked
pass that still launches that kernel per density set buys nothing. A
multi-density FelizJueves kernel is out of scope here.

## Not done

- The per-root canonicalization's own derivative: the caller passes
  canonical orbitals and z^c takes the constraint.
- Λ lives at naux nmo² whole; a virtual–virtual block that only its diagonal
  reaches is still stored whole, as `mcscf::df_gradient`'s extra adjoint
  takes it.
- The frozen core in the interstate couplings (`qd_sc_coupling`,
  `qd_pc_coupling`) and so in the derivative coupling (`../nac/`): they
  take no `nfrozen`, and `[nevpt2_nac]` refuses `frozen_core`
  (`../nac/README.md`).
- A state average's root of the single-state NEVPT2 (`lagrangian_gradient`,
  `sc_gradient`, `pc_gradient`) is refused; the SA reference is
  `qd_sc_gradient`'s and `qd_pc_gradient`'s.
- The single-state gradient's torque (the rotational sum) is not checked;
  its translational one is. The QD gradient's both are.

## Tests

`test/gtest/nevpt2_gradient_gpu.cpp`, on this tree's own DF-CASSCF of
`data/casscf/casscf_df.json`'s `h2o_ccpvdz` (CAS(4,4), one root, so the
CASSCF's orbitals are the root's canonical ones), run through the registry
once per process.

- `EnergyMatchesPyscf`: sc_energy's E_corr against PySCF's DF-NEVPT2
  (`data/nevpt2/nevpt2_df.json`) at the pipeline test's 1e-9, and
  sc_adjoint's energy against sc_energy's.
- Each adjoint against a Richardson ladder of sc_energy itself, every input
  rebuilt at each point: central differences at t = 1e-2 / 2^k, ten rungs,
  successive changes falling by a factor in (3.2, 4.8) over at least three
  halvings, and the extrapolant at that run's finest pair within 1e-7
  relative (`nevpt2_energy_gpu.cpp`'s ladder):
  - `CiAdjointMatchesLadder`: ⟨u, y⟩ along c(t) = (c + t u) / ‖c + t u‖, u =
    P H_r c normalized, H_r a random combination of the active block of
    C^T h C, its square and the active (tu|vw): spin-free, so u is in c's
    spin, and built from operators that commute with the point group, so u
    is in its symmetry sector. A raw random u breaks the hole-side metrics
    and puts poles in the SC finish.
  - `Nevpt2GradientRotation`: Σ_{p>q} κ_pq A_pq along C exp(t κ) for a random
    κ on each block alone, core–core, active–active, virtual–virtual and the
    three mixed ones, not re-canonicalized.
  - `Nevpt2GradientIntegrals`: ⟨h̄, Cᵀ ḣ C⟩ along a seeded symmetric AO ḣ,
    and ⟨Λ, Cᵀ Ḃ C⟩ along a seeded Ḃ symmetric in its AO pair. **Ḃ is a
    tenth the length of the others**: at full length (uniform in [−1, 1) on
    every element) its ratios had not settled by t = 2e-5 and the
    extrapolant missed by 3.0e-6; at a tenth they reach 4.000 and the miss
    is 3.6e-10, the prediction exactly a tenth of the full one's. The
    tolerance is unchanged.
  - **The state is put in one gauge first** (`fix_gauge`): each orbital's
    leading coefficient positive, the CI vector carried along, then its
    own. The CASSCF's eigensolvers leave run-dependent signs, so unfixed,
    one seed's κ was a different rotation from run to run, and one such
    active–core ladder had not reached the t² regime by its last rung. In
    one gauge the directions are the same every run.
  - Measured on an RTX 3080 over five runs, worst of each: ci 7.2e-10;
    core–core 7.0e-9, active–active 1.6e-8, virtual–virtual 2.2e-9,
    active–core 3.9e-9, virtual–core 4.5e-9, virtual–active 1.1e-9; h
    6.5e-10, B 5.9e-10 relative. Each ladder is well under a second; the
    CASSCF setup dominates.
- `SourceTermShapes`: A is X − Xᵀ, h̄ and every Λ^L symmetric.
- z^c, from `sc_adjoint`'s A and ε at the same state (max |z^c| and the
  smallest inactive gap printed with each):
  - `Nevpt2ZcStationarity`: E_corr + ⟨z^c, f⟩ along C exp(t κ), κ random on
    the core–core and on the virtual–virtual block, z^c fixed and every
    input rebuilt from C(t), ε included, not re-canonicalized: the ladder's
    extrapolant within 1e-7 of the slope Σ κ A it cancels, and at the point
    Σ κ (A + A^{z^c}) the same.
  - `Nevpt2ZcRotation`, `Nevpt2Zc.CiSourceMatchesLadder`,
    `Nevpt2ZcIntegrals`: each of Φ's adjoints against a ladder of Φ at fixed
    z^c along the sc_adjoint ladders' directions (every block's κ, the same
    u, the same ḣ and Ḃ), at the same rule and 1e-7. **Along h and B the
    path is bent**, (t + t³/t0²) along the direction: Φ is linear in h and
    quadratic in B, so along a straight line its central difference is exact
    and the ladder's changes are round-off, with no t² regime for the ratio
    rule to find. The bend keeps the tangent, so the prediction, and makes
    D(t) = D(0) (1 + t²/t0²) exactly.
  - `Nevpt2Zc.DegeneratePairRule`: the first virtual and a partner with
    |A| > 1e-6 forced 1e-7 Eh apart: refused `Numerical`, the message naming
    both; with A_pq set to 1e-12, z_pq = 0 and every pair without the moved
    orbital the unforced division to the bit.
  - `Nevpt2Zc.Shapes`: z^c symmetric, zero off its two blocks and on the
    diagonal; h̄^{z^c} = z^c, A^{z^c} = X − Xᵀ, Λ^L symmetric; y^c symmetric
    under the swap of its two strings, as the singlet c is.
  - `Nevpt2ZcStateAverage`: Φ = ⟨z^c, f(γ_SA)⟩ over two roots, the state's
    c and the u above (in c's sector, perpendicular to it; Φ reads γ_SA
    alone, so the roots need not be eigenvectors), at weights equal (½, ½)
    and unequal (0.7, 0.3), z^c fixed: each adjoint against a ladder of Φ
    at the same rule and 1e-7, along h and B (the bent path), along
    C exp(t κ) with every block's κ at once, and along each root alone,
    c_r(t) = (c_r + t d_r)/|·|, d_0 = u and d_1 = c, against ⟨d_r, y^c_r⟩.
  - `Nevpt2Zc.OneRootIsTheSingleStatePhi`: the one-root call at weight 1
    against a second call and against weights (1, 0) over (c, u): h̄ to the
    bit, root 1's y^c exactly 0, Λ, y^c, X and A within 32 ε of each one's
    largest element, and Φ (itself round-off, ~1e-17) within 5e-17
    absolute: the run-to-run floor above, where 50 repeats reached 10.7 ε
    (A) and 6.2e-18 (Φ). Not 1e-17 absolute: two calls of the same
    function have come out 1.39e-17 apart on A.
  - `Nevpt2Zc.RefusesBadWeights`: no weights, a negative one, and a sum of
    0.9 each `InvalidConfig`.
  - Measured on an RTX 3080, one run: max |z^c| 2.4e-3 and the smallest
    inactive gap 1.43e-3 (the two lowest virtuals, of different C₂ᵥ
    symmetry; `test/gtest/data/README.md`, "Not symmetric water");
    stationarity core–core 8.4e-9, virtual–virtual 2.0e-9 of the slope, and
    Σ κ (A + A^{z^c}) at round-off; Φ's ladders ci 2.3e-11, the six blocks
    4.6e-11 to 3.8e-10, h 2.3e-11, B 4.5e-11 relative. Each z^c case is well
    under a second.
- The nuclear gradient, on `data/nevpt2_gradient/nevpt2_gradient.json`'s
  cases (`test/gtest/data/README.md`): skew H₂O CAS(4,4) and H₂CO CAS(6,6),
  cc-pVDZ / cc-pVDZ-JKFIT, each this tree's own DF-CASSCF from the case's
  start through the registry (the Casscf step's canonical core and
  virtuals), put in one gauge and the `mcscf::Solution` rebuilt from it
  (its RDMs from the gauge-fixed vector):
  - `Nevpt2NuclearGradient.MatchesNumerical`: `sc_gradient`'s total against
    the numerical gradient (`gradients.sc.richardson`) per component to
    1e-6 Eh/a0, the sum over atoms below 1e-9, X^L's six residues below
    1e-9, the CASSCF energy against the reference's to 1e-8 and E_corr to
    1e-9; and a state average refused `Unsupported`. Each prints every
    component, max |z^c|, the smallest inactive gap and the PCG iterations.
  - `Nevpt2NuclearGradient.MultipliersVanishWithoutCorrelation`: the same
    `lagrangian_gradient` with E_corr's adjoint and z^c zeroed. The
    right-hand side is then the CASSCF's own orbital gradient g (< 1e-9), so
    the multipliers are that small (below 1e-8), and the gradient is the
    DF-CASSCF's: PySCF's analytic one (`casscf_gradient`) to 1e-8, and
    `df_gradient`'s own single-root one to 1e-9, the bar first order in g
    that `mcscf_gradient_gpu.cpp` takes, since the own formula drops Z.
    That isolates the Lagrangian plumbing from the NEVPT2 adjoint.
  - `Nevpt2NuclearGradient.LagrangianIsStationary` (skew H₂O): the full L
    (E_cas and g_orb from `response::fock`, g_ci from H₀'s sigma, E_corr,
    Φ), the multipliers fixed and every input rebuilt, along C exp(t κ), κ
    random on the three non-redundant blocks at once, and along the CI
    ladders' u ⊥ c: the derivative within 1e-7 of the slope of E_corr + Φ
    it cancels. E_corr and the rest are laddered apart and their
    extrapolants summed, each where its extrapolants are steadiest: E_corr
    reaches its O(t²) regime only at t ~ 3e-4, where the rest has reached
    E_cas's rounding, so one ladder over L saw three ratios in (3.2, 4.8)
    only by chance (#640).
  - Measured on an RTX 3080: every component within 3.4e-9 (skew H₂O) and
    5.4e-9 (H₂CO) of the numerical gradient, each about twice its error
    bar; translation 1e-14 and 6e-13. max |z^c| 1.9e-3 and 4.4e-3, the
    smallest inactive gaps 3.0e-3 and 1.6e-2, PCG 18 iterations each,
    residual 1.6e-10 and 5.3e-11. X^L worst on the active–active block,
    7.6e-10 and 1.8e-10 (the CI response's tolerance through the gauge
    identity), the mixed blocks ≤ 9e-11, core–core and virtual–virtual
    ≤ 1e-14. Zeroed: max |Z| 4.1e-10 (skew H₂O; H₂CO exactly 0), against
    df_gradient 1.5e-10 and 6e-14, against PySCF 1.9e-11 and 8.3e-12.
    Stationarity, worst of 50 runs on the gfx1200: orbitals 1.4e-8, CI
    2.5e-9 of the slope (20 on the 3080: 1.7e-8, 1.2e-9). Each case
    is under a second, the setup included, so none is a slow test.
- PC-NEVPT2:
  - `Nevpt2NuclearGradient.PcMatchesNumerical`: `pc_gradient`'s total
    against `gradients.pc.richardson` (block2's `WickICNEVPT2` over PySCF's
    fitted integrals) per component to 1e-6 Eh/a0, the sum over atoms below
    1e-9, X^L's six residues below 1e-9, PC's E_corr against `e_corr.pc` to
    1e-9, and a state average refused `Unsupported`. PC's z^c is not
    asserted to vanish: it does not (above).
  - `Nevpt2PcZc.DegeneratePairTakesTheZeroBranch` (`h2o_ccpvdz`): in each of
    the core and the virtual space, the pair with |A^SC| > 1e-6 closest in ε,
    the upper orbital moved to 1e-9 Eh above the lower through h alone (h +
    δ v vᵀ, v = C^−T e_j, so only that orbital's diagonal of C^T h C moves
    and f stays diagonal): PC's A_pq at most 1e-8 and 1e3 below SC's, its
    z_pq exactly 0, and SC's pair refused `Numerical`, naming both orbitals.
    The port of `nevpt2_energy_gpu.cpp`'s degenerate-ε check to the
    gradient's inputs.
  - `Nevpt2PcGradient.RefusesWhatTheEnergyRefuses` (skew H₂O): a zero CI
    vector (Sr's metric zero, "no positive eigenvalue") and an intruder,
    every virtual shifted through h to put Sr's strongest mode at a quarter
    of its coupling, |t| = 4 ("has an intruder"): `pc_gradient` fails with `pc_energy`'s error, kind and message.
  - Measured on the gfx1200: every component within 3.5e-9 (skew H₂O) and
    5.9e-9 (H₂CO) of the numerical gradient, each about its error bar;
    translation 1.4e-14 and 5.0e-13; E_corr −3.2e-12 and −1.1e-12 from
    block2's. max |z^c| 1.9e-3 and 4.5e-3, PCG 18 and 21 iterations. X^L
    worst on the active–active block, 6.1e-10 and 4.2e-10, the mixed blocks
    ≤ 7.7e-11, core–core and virtual–virtual ≤ 1.3e-14. At the real state PC's
    A on the core pair (1, 0) is 7.4e-3 (SC's 7.3e-3), and on the virtual
    pair (22, 21) −4.5e-5 (SC's −4.6e-5); forced to 1e-9 apart, PC's are
    6.6e-14 and −1.5e-13, SC's −1.5e-7 and 7.4e-8. Each case is under three
    seconds, the setup included.

- SC-QD-NEVPT2 (`qd_sc_energy`, `qd_sc_adjoint`), on `data/qd_nevpt2/`'s
  six cases (#670), each this tree's own DF-SA-CASSCF from the case's start
  through the registry (as `pipeline_nevpt2_gpu.cpp`'s `PipelineQdNevpt2`
  runs it, without the Nevpt2 step), every root's vector carried through
  `fix_gauge`:
  - `Nevpt2QdAdjoint.AdjointsMatchLadders`: E[W] at two W, u_K u_Kᵀ (K the
    highest state of our own symmetrized H_eff, by Jacobi on the host) and a
    seeded random symmetric W, each adjoint against a ladder of E[W] at the
    gradient ladders' rule and 1e-7, every input rebuilt at each point, the
    SA ε included: along h and B, along C exp(t κ) with every block's κ at
    once, and along each root alone, c_r(t) = (c_r + t d_r)/‖·‖, d_r
    `ci_direction`'s at c_r, against ⟨d_r, ȳ_r⟩. Both seeds read the same
    forwards. Also ε against the CASSCF's `mo_energies` (core and virtual)
    to 1e-10, measured ≤ 1.4e-13; H_eff's eigenvalues are printed beside the
    goldens' (the h2co cases and `h8_ring_sto3g_sa3` agree to 4e-10; the
    other hydrogen ones, whose SA-CASSCFs the goldens did not polish, to
    at most 5.1e-8).
  - **Not every ladder is smooth at the ladder's scale, so not every one
    runs** (`kQdSmooth`, as #724 found along raw directions). Measured over
    five runs on the RTX 3080, ratios repeating to three decimals: every
    path at both seeds on `h8_sto3g_sa3`; h, B, C, c_0, c_1 (eigen) and c_0,
    c_1 (random) on `h2co_ccpvdz_sa2`; h, B, c_0, c_1 and c_0, c_1 on
    `h2co_ccpvdz_sa2_weighted`; B, c_0 and c_0 on `h2co_skew_ccpvdz_sa3`;
    B, C, c_0 at both on `h6_631g_sa2`; h, c_0, c_1, c_2 at both on
    `h8_ring_sto3g_sa3`. Worst relative miss per case, 2.1e-9 to 7.9e-8
    (`h6_631g_sa2` along B at the eigen seed, the same 7.89e-8 each run).
    The rest end in a kink or never settle. On the two C₂ᵥ h2co cases it is
    symmetry: the roots are 1A₁ and 1A₂, so every coupling and every
    perturber whose own norm is zero by symmetry (dropped at
    |N| ≤ 1e-14) vanish at the point; along h, B or C, which break C₂ᵥ, such
    a perturber's N and H grow as t² with their ratio finite, so its
    coupling T/(Δ + H/N) grows as t, a first-order change no derivative at
    the point sees. The random W reads those couplings (its ladders along h
    and B find a ratio run, then break, and miss by 2.6e-4 and 8.0e-4); the
    eigenvector seed, diagonal there, does not. Elsewhere it is #724's
    small own norm moved toward the pole Δ + H/N = 0 (`h6_631g_sa2` along h:
    ratios … 0.000 −1.000 −13037, a step); a random diagonal W was not
    smooth along those paths either. Every case's c_0 ladders at both seeds.
  - `Nevpt2Qd.OneRootIsScAdjoint` (`h2o_ccpvdz`): one root, weight 1,
    W = 1 against `sc_adjoint`: the energy to 1e-12, and h̄, Λ, ȳ, X, A
    and ε within 4096 ε of each one's largest element. Not bitwise:
    `sc_adjoint` is not bitwise against itself (its walk's and the einsums'
    atomics), and over 35 runs two of its calls came 2476 ε apart (A), the
    one-root QD call 2532 ε from it; the bound is that floor with room.
  - `Nevpt2Qd.RefusesBadInputs`: no weights, a negative one, a sum of 0.9
    (both entry points), and a W not n² long, each `InvalidConfig`.
  - Each case is about a second, its SA-CASSCF included.

- PC-QD-NEVPT2 (`qd_pc_energy`, `qd_pc_adjoint`), on #671's three cases
  (`h2co_skew_ccpvdz_sa3`, `h6_631g_sa2`, `h8_ring_sto3g_sa3`), the same
  SA-CASSCFs:
  - `Nevpt2QdPcAdjoint.AdjointsMatchLadders`: the SC-QD ladders above
    (`expect_qd_ladders`) with PC's elements, the eigenvector seed u_K of
    PC's own symmetrized H_eff; H_eff's eigenvalues are printed beside the
    `pc` goldens'. The c_r ladders are the check of the dropped
    rank-projector terms. Measured over 24 runs on the RTX 3080, every path
    at both seeds finds its ratio run, and every one is asserted
    (`kQdPcSmooth`) but three eigen-seed c_r ladders at E[W]'s round-off:
    `h6_631g_sa2` c_0 and `h8_ring_sto3g_sa3` c_0 and c_2, slopes 8.1e-6 to
    1.4e-4, missing by ≤ 2.7e-12 and ≤ 1.8e-11 absolute with the sign
    changing run to run, the forwards' own spread over 2t at the finest
    rung and the random seed's absolute miss along the same c_r; so 1e-7
    relative fails in 15, 1 and 3 runs of 24. The same c_r at the random
    seed is asserted on every root of every case. State 1's PC is refused
    at the coarse rungs on `h2co_skew_ccpvdz_sa3` (h, B, C, c_1) and
    `h8_ring_sto3g_sa3` (c_1), out to t = 1.25e-3, as #729 found; those
    rungs are NaN and the run is read below them. The ladder moves a
    denominator onto a pole there, and the intruder rule (#874, no sign
    test) still refuses there: on `h2co_skew_ccpvdz_sa3`, 13 points, Sijr
    and Srs, |t| from 1.07 to 130 at |λ + Δ| from 4.0e-5 to 1.3e-2 (RTX
    3080). Every ladder passes with its bars and rungs unchanged.
    `h8_ring_sto3g_sa3` eigen c_1 is not asserted either (#736): near that
    pole its ratios are still settling at the finest pair, whose
    extrapolant reads the forwards' rounding at a slope of only 6.1e-3 (it
    missed 1e-7 on a push, at 1.09e-7), and the pair before is 5.5e-8 off
    in truncation; starting finer only reads more rounding. Its random seed
    is asserted (≤ 3e-10). The PC-QD ladders take the steadiest extrapolant
    (where two neighbours agree best), not the finest pair's, which rounding
    took past 1e-7 on `h8_ring_sto3g_sa3` eigen C too (1.08e-7). Over 20
    runs every asserted one then missed by at most 2.7e-8
    (`h2co_skew_ccpvdz_sa3` random h).
  - `Nevpt2QdPc.OneRootIsPcAdjoint` (`h2o_ccpvdz`): one root, weight 1,
    W = 1 against `pc_adjoint`: the energy to 1e-12, every array within
    16384 ε of its largest element. Over 38 runs two `pc_adjoint` calls came
    6841 ε apart (y), the one-root call 5772 ε from it; the bound is about
    2.4 times that floor.
  - `Nevpt2QdPc.RefusesWhatTheEnergyRefuses` (`h8_ring_sto3g_sa3`): root 1's
    vector zero ("state 1: Sr's metric has no positive eigenvalue") and the
    virtuals shifted through h to make state 0's strongest Sr mode an
    intruder ("state 0: Sr has an intruder"): `qd_pc_adjoint` returns `qd_pc_energy`'s error, kind and
    message.
- The SC-QD-NEVPT2 Lagrangian (`qd_sc_lagrangian`), on `h8_sto3g_sa3`'s
  geometry (its E[W] smooth along every path), its DF-SA-CASSCF at three
  weightings, state K = 2 (the highest):
  - `Nevpt2QdLagrangian.IsStationary/{equal,unequal,mixed}`: weights
    (⅓, ⅓, ⅓) (every pair a ζ), (0.5, 0.3, 0.2) (every pair a t) and
    (0.4, 0.4, 0.2) (ζ and t in one solve). L, its multipliers fixed and
    every input rebuilt from (C, c), is laddered along a random orbital
    rotation over every block but the active–active one (core–core and
    virtual–virtual with κ_nr), a random CI direction off every root on
    every root at once, and a random rotation among the roots; E^corr[W]
    and the rest each have a ladder, and the summed steadiest extrapolant
    is held to 1e-7 of the slope it cancels. On the RTX 3080 the worst
    was 3.9e-8 (equal, orbitals); the CI ladders, which pin the
    1/(2 w_r), within 3.8e-8. The gate's residues are printed and held to
    1e-9. The unequal-weight fixtures of `data/qd_nevpt2/` are not used:
    `h2co_ccpvdz_sa2_weighted` is C₂ᵥ, τ = 0 and E[W] even in Θ, so its Θ
    ladder is round-off alone; `h2co_skew_ccpvdz_sa3_weighted`'s E^corr[W]
    is not smooth along the orbitals at the ladder's scale (#724's
    mechanism; the test's comment has the numbers), though its gate holds
    to 8e-11.
  - `Nevpt2QdLagrangian.Refuses` (`h2co_ccpvdz_sa2`): the weighted γ set to
    root 0's, so that the recovered weights are (1, 0), is `InvalidConfig`
    naming root 1; root 1's CASSCF energy moved to make H_eff's diagonal
    equal (its off-diagonal vanishes by C₂ᵥ symmetry) is `Numerical`
    naming states 0 and 1, for either state.
- The SC-QD-NEVPT2 nuclear gradient (`qd_sc_gradient`), on every case of
  `data/qd_nevpt2_gradient/` (#696), each this tree's DF-SA-CASSCF at the
  generator's bars (‖g‖ < 1e-10, |ΔE| < 1e-13):
  - `Nevpt2QdGradient.MatchesNumerical/<case>`: every state K, g_K · u
    along each stored direction and every committed full-3N component,
    within 2× that number's committed error bar; the translational and
    rotational sums, absolute, within 2× the case's largest committed bar
    (rotational invariance holds only to screening and convergence:
    `mcscf::df_gradient`'s own root gradients carry a torque of up to
    ~1e-8 relative on skew formaldehyde). Each separately coded term's own
    nuclear contribution (ζ's pass; Θ, z^c and the correlation arrays alone
    through `df_transition_gradient` at zero RDMs) is printed against the
    checked numbers' bars and held above 10 bars on one case each. The sum
    rule Σ_K g_K is printed.
  - `Nevpt2QdGradient.RefusesWhatTheLagrangianRefuses` (`h2co_ccpvdz_sa2`):
    `qd_sc_lagrangian`'s two refusals, kind and message unchanged.
  - `Nevpt2QdGradient.OneRootAgainstScGradient` (`h2o_skew_ccpvdz`): the
    one-root call against `sc_gradient`, printed.
  - `Nevpt2QdGradient.WFormIsLinear` (`h8_sto3g_sa3` at weights
    (0.4, 0.4, 0.2), ζ and t both): G(W₁ + W₂) against G(W₁) + G(W₂) to
    1e-10 relative to the largest element compared, W₁ random symmetric of
    trace 1 and W₂ of trace 0, every off-diagonal non-zero. The sum rule
    Σ_K dE_K/dR against G(1) is printed. The state form's check is
    `MatchesNumerical`: it is the W form at u_K u_Kᵀ.
  - `Nevpt2QdGradient.WFormRefusesBadW`: a W one short, one of (n + 1)²,
    one off-diagonal moved 1e-6 and one NaN, each `InvalidConfig` from both
    W forms.
- The SC-QD-NEVPT2 interstate coupling (`qd_sc_coupling`), on every case
  of `data/qd_nevpt2_nac/coupling/` (#743), each this tree's DF-SA-CASSCF
  at #696's bars:
  - `Nevpt2QdCoupling.MatchesNumerical/<case>`: every pair P < Q the data
    writes, h_PQ · u along each stored direction and every committed full-3N
    component (skew H₂CO), within 2× that number's committed `sc_h` error
    bar; h_QP from the swapped call against h_PQ to 1e-10 relative to the
    largest summand of the two derivative passes' terms (their W is
    bitwise identical, so the check sees only atomic-sum noise, which
    follows the largest summand, not the cancelled h: #748's ruling); the
    translational and rotational sums, absolute, within 2× the case's
    largest committed `sc_h` bar. The data is in its
    own run's root phases, so each pair's sign s_m(P) s_m(Q) is read off
    H_eff's element (m(P), m(Q)), ours against the data's `sc_heff`
    (m(P) R_P's largest component), never off h itself. Printed, not held:
    the reference part (`mcscf::df_gradient` at the mix ρ = diag W^PQ) and
    the PT2 part (h_PQ less it), each against the data's `sc_h_reference`
    and `sc_h_pt2` and its largest contribution in bars. On the RTX 3080
    the worst |h − reference| was 0.019 bars (`h8_ring_sto3g_sa3`), 0.73
    (`h2co_skew_ccpvdz_sa3`, a full-3N component of pair (0, 2)) and 0.036
    (`h2co_skew_ccpvdz_sa3_weighted`); h_QP within 3.5e-11 relative; the
    rigid sums at most 1.8e-11 against bars of 7e-8 and up (#747's run,
    h_QP then against max |h|); each part at least 342 bars on every pair,
    and each within 0.73 bars of its own family. About 2 s a case.
  - `Nevpt2QdCoupling.Refuses` (`h2co_ccpvdz_sa2`): E_0 = E_1 forced as in
    `Nevpt2QdLagrangian.Refuses` is `Numerical` naming both states, in either
    order; P = Q and a state out of range are `InvalidConfig`.
- The PC-QD-NEVPT2 nuclear gradient (`qd_pc_gradient`), the same way, on
  every case with PC numbers (all but `h8_sto3g_sa3`):
  - `Nevpt2QdPcGradient.MatchesNumerical/<case>`: SC's checks against the
    data's `pc` numbers and `pc_energies`, one shared test body. Visibility
    adds PC's own term: its correlation arrays less SC's at the same W
    (`qd_sc_adjoint` at PC's u_K u_Kᵀ), through `df_transition_gradient`
    at zero RDMs. Each term is held above 10 bars on one case.
  - `Nevpt2QdPcGradient.RefusesWhatTheEnergyRefuses` (`h8_ring_sto3g_sa3`):
    state 0's Sr intruder (the virtuals shifted through h), and root 1's
    vector zero; each is
    `qd_pc_energy`'s error, kind and message unchanged.
  - `Nevpt2QdPcGradient.RefusesWhatTheLagrangianRefuses`
    (`h2co_ccpvdz_sa2`): a root of weight 0 and a degenerate pair of PC's
    H_eff, named `nevpt2::qd_pc_gradient`.
  - `Nevpt2QdPcGradient.OneRootAgainstPcGradient` (`h2o_skew_ccpvdz`): the
    one-root call against `pc_gradient`, printed.
  - `Nevpt2QdPcGradient.WFormIsLinear` (`h2co_skew_ccpvdz_sa3_weighted` at
    its own (0.4, 0.4, 0.2), ζ and t both): `Nevpt2QdGradient.WFormIsLinear`'s
    body through PC's W form, the same seeds and the same 1e-10; the sum
    rule over PC's states printed. `h8_sto3g_sa3` is not used: its root 0
    passes PC's cuts only over round-off. On the RTX 3080: 2.9e-11
    relative, the sum rule 3.3e-11; about 2 s.
- The PC-QD-NEVPT2 interstate coupling (`qd_pc_coupling`), on every case
  of `data/qd_nevpt2_nac/coupling/` (PC accepts all three):
  - `Nevpt2QdPcCoupling.MatchesNumerical/<case>`:
    `Nevpt2QdCoupling.MatchesNumerical`'s body against the data's PC
    numbers (`pc_h`, its sign from `pc_heff` and `pc_heff_vectors`,
    `pc_h_reference`, `pc_h_pt2`): every pair's directional and full-3N
    components within 2× each `pc_h` bar, h_QP to 1e-10 of the largest
    summand, the rigid sums within 2× the case's largest `pc_h` bar. On
    the RTX 3080 the worst |h − reference| was 0.006 bars
    (`h8_ring_sto3g_sa3`), 0.31 (`h2co_skew_ccpvdz_sa3`, a full-3N
    component of pair (1, 2)) and 0.73 (`h2co_skew_ccpvdz_sa3_weighted`);
    h_QP within 2.4e-11 of the largest summand (9.2e-10 of max |h| on a
    small-coupling pair, the reason for the summand scale); the
    rigid sums at most 2.9e-10 against bars of 6.7e-8 and up. About 2 s a
    case.
  - `Nevpt2QdPcCoupling.RefusesWhatTheEnergyRefuses` (`h8_ring_sto3g_sa3`):
    `Nevpt2QdPcGradient.RefusesWhatTheEnergyRefuses`' two refused classes
    refuse the pair (0, 1) and (1, 0) with `qd_pc_energy`'s error, kind and
    message unchanged; P = Q and a state out of range are `InvalidConfig`,
    named `nevpt2::qd_pc_coupling`.
- **The frozen core** (#875):
  - `Nevpt2QdFrozenCore.EnergyMatchesReference`: `qd_sc_energy`'s and
    `qd_pc_energy`'s H_eff eigenvalues at every count
    `data/qd_nevpt2/qd_nevpt2_frozen_core.json` has (`h2co_ccpvdz_sa2` at
    2, `h2co_skew_ccpvdz_sa3` at 2 and 1) against #859's references to
    1e-8; worst 9.5e-12 on the RTX 3080.
  - `Nevpt2QdFrozenCoreAdjoint.AdjointsMatchLadders`
    (`h2co_skew_ccpvdz_sa3`, n_f 2, the eigenvector seed): the ladders
    above at the frozen count along h, B, C exp(t κ) with κ on the
    frozen–correlated pairs alone, and each root; and A's frozen–frozen
    block within 1e-10 of the frozen–correlated one's largest (measured
    1.4e-19 and 6.5e-19 against 8.5e-3). PC runs every path. SC runs B, C
    and c_0: h, c_1 and c_2 are not smooth at the ladder's scale on this
    case all-electron either ("Not every ladder is smooth" above; the h
    ladder runs the all-electron ratios; #894). Worst relative miss 2.2e-8
    (SC, B).

`test/gtest/nevpt2_builders_tangent_gpu.cpp`, `Nevpt2BuildersTangent`, on
a synthetic point: the builders are polynomials in (h, B, c), with no
denominator and no stationarity, so a random one (naux 24, two core, CAS(4,4)
over random CI coefficients, five virtuals; AO h and B over a random
non-orthogonal C = 1 + 0.1 R) tests them as well as a converged CASSCF and
runs in under a second.

- `MatchesLadder`: one block call along two random directions (ḣ symmetric,
  Ḃ symmetric in its pair at a tenth the length, ċ = 0.3 c plus a unit
  vector across c), each against a ladder of the builders' value, run in the
  MO basis itself (C = 1, Cᵀ h C and Cᵀ B C moved along the direction), so
  the directions are checked to be read in the MO basis of C. Every field:
  the five blocks, (tu|vw), each `EnergyInputs` field, f_core and f_virt.
  Central differences at t = 1e-2 / 2^k, ten rungs, ratios of successive
  changes (max norms) in (3.2, 4.8) over at least three halvings, the
  steadiest extrapolant of that run, within 1e-7 of the tangent relative to
  its largest element. **A field quadratic along the direction** (the blocks'
  copies, (tu|vw), h2e, the F^I blocks, the RDMs) has exact central
  differences: its changes sit at rounding, below 1e-9 of the field, its
  ratios are noise, and its steadiest pair over every rung meets the same
  bar. Measured on the gfx1200 (`--preset hip`): worst 3.3e-12 (e_core),
  every flat field under 1.8e-12, f3ca and f3ac 2.3e-13. Freezing γ in ε's
  tangent fails it.
- `FockBlocksAreZcAdjointsFock`: the value's f against `zc_adjoint`'s,
  which the ladder cannot see (it checks the tangent of whatever f the
  builders form): Σ z_pq f_pq over a random symmetric z on the two blocks
  against Φ at 1e-12 of the summands' sum (6.3e-17), and f's diagonals
  against e_core and e_virt.

`test/gtest/nevpt2_hessian_product_gpu.cpp`, `Nevpt2HessianProduct`, on
this tree's DF-CASSCF of `data/nevpt2_hessian/`'s `h2co_skew_sto3g` (C1,
CAS(4,4), no SC perturber empty: #803's geometry for SC's smoothness), z^c
its own `zc_multipliers`. The directions are AO perturbations of h (uniform,
symmetric) and B (a tenth that, symmetric in the AO pair), taken to the MO
basis of C in the test, and ċ_I = (r_I + 0.3) c_I (r symmetric in the two
strings), covariant with the CI vector's signs, so no gauge is fixed.

- `MatchesLadder`: one call along two such directions, every field of the
  product (h̄, Λ, y, X, A) against a ladder of `sc_adjoint` + `zc_adjoint`
  at fixed z, at h + t ḣ, B + t Ḃ and c(t) = (c + t ċ)/|c + t ċ|, y as
  P(c(t)) (y + y^c)/|c + t ċ|: the Rayleigh-normalized functional's
  gradient, so the normalization's terms are inside what the ladder sees.
  `nevpt2_builders_tangent_gpu.cpp`'s ladder, at fifteen rungs (t = 1e-2 to
  6.1e-7): X's and A's ladders along the second direction break their run
  near t = 2e-4 (a max-norm change passing between elements; X_ao = X_mo
  to 3e-15 at every rung, so X's ladder is h̄'s and Λ's) and settle only past
  the tenth rung. Then `parts`: each part above 10 bars (1e-7 of the field's
  largest element) in some field, and their sum against the product.
  Measured on the gfx1200 (`--preset hip`) and the RTX 3080 (`default`):
  worst relative miss 2.9e-10 and 3.1e-10 (A); the smallest part, Φ's,
  3.2e4 bars (in A, second direction); per direction at least: the slabs'
  1.2e6 (A), the builders' 1.0e7 (X), the one-slot terms' 2.1e5 and the
  normalization's 3.1e5 (y); the parts' sum off by 1.4e-13 and 6.0e-14
  (bar 1e-9). About 1 s on the 3080, 10 s on the gfx1200.
- `PartnersChangeNothingWithoutEmptyPerturbers`: each direction partnered
  with the other; with nothing empty, the products as without, to 1e-12.
- `Refuses`: InvalidConfig for one partner for two directions, a null
  partner array, a null z, and a vector of norm 1.5.

`Nevpt2PcHessianProduct`, the same file, over `pc_adjoint` + `zc_adjoint`
at PC's own z^c. Its CI directions are singlets, ċ = H_κ c + 0.3 c with
H_κ a random spin-free Hamiltonian over the active orbitals (one- and
two-body) and the part off c 0.5 long: SC's covariant ċ mixes in quintets,
which lift the metrics' spin-null modes (above). PC's ladder starts at
t = 2e-3 (thirteen rungs): at 1e-2 the moved states meet intruders, which
PC refuses.

- `MatchesLadder/h2co_skew_sto3g` (`pc_floor` 1e-6, which drops a real
  mode of Srs and of Sij beside the structural ones; every kept spectrum
  gapped, the smallest 1.7e-6): every field of two directions' products
  against the ladder at 1e-7, the null-space rotation terms (the product
  less its `span`-false twin) above ten bars in some field of each, the
  parts summing to the product at 1e-9. Measured: worst 1.1e-9 to 1.5e-9;
  the rotation at least 8.8e4 bars (y); lifts ≤ 2.4e-14.
- `MatchesLadder/psb3_twist_sto3g` (slow tier): the PSB3 twist,
  DF-SA3-CASSCF(6,6)/STO-3G (state 0), at `pc_floor` 1e-8: Sr and Si drop
  43 and 42 modes, about 30 of them real, and keep 1.2e-8. h and B move a
  tenth as far, and the ladder runs from 1e-3 over twelve rungs: intruders
  sit within 2.5e-4 along the full-length h. Measured: worst 2.3e-8 to
  5.3e-8 over three runs; the rotation 2.3e5 to 2.8e5 bars. At `pc_floor` 1e-9 the worst was 6.7e-8 in one run
  and 1.2e-7 in another, the products repeating to 3e-9 (8e-10 at 1e-8).
- `SymmetricLiftReported` (`hcn_sto3g`): above, "The Schur lift": the
  report against the probe (1e-2), the bend's product against a ladder
  inside t*, the symmetric and orbital-only directions lifting nothing and
  passing the ladder.
- `DegenerateInactivePair` (`h2co_skew_sto3g`, the default cut): its two
  virtuals and its closest core pair brought to 1e-9 Eh apart through h
  alone (`Nevpt2PcZc`'s construction); z^c is zero on both pairs, and the
  product passes the ladder.

## Files

| File | Role |
|------|------|
| `gradient.cppm` | Module interface: `sc_energy`, `sc_adjoint`, `ScAdjoint`; `pc_energy`, `pc_adjoint`, `PcAdjoint`; `qd_sc_energy`, `qd_sc_adjoint`, `QdScAdjoint`; `qd_pc_energy`, `qd_pc_adjoint`, `QdPcAdjoint`; `qd_sc_lagrangian`, `QdScLagrangian`; `qd_sc_gradient`, `qd_sc_coupling`, `QdScGradient`; `qd_pc_gradient`, `qd_pc_coupling`, `QdPcGradient`, `QdFold`; `qd_inputs`; `builders_tangent`, `BuildersTangent`, `BuilderOutputs`; `sc_hessian_product`, `ScHessianProduct`, `pc_hessian_product`, `PcHessianProduct`, `AdjointTangent`, `MoDirections`; `zc_multipliers`, `ZcMultipliers`, `zc_adjoint`, `ZcAdjoint`; `lagrangian_gradient`, `sc_gradient`, `ScGradient`, `pc_gradient`, `PcGradient` |
| `operands.cppm` | Internal partition `:operands`: the column-major einsum operands and contractions, zeroed buffers, block adds, host copies, and the count and weight checks the units share |
| `builders.cppm` | Internal partition `:builders`: `ScArrays`, the builders as einsum terms, their forward and reverse runs, `shared_arrays`, `sc_arrays`, `builders_adjoint`, `adjoint_slot`, `fold_bars` and `root_rdms_adjoint`, declared for `gradient.cpp` (which defines them), `qd.cpp`, `tangent.cpp` and `hessian_product.cpp`; `direction_arrays` and `b_block`, `tangent.cpp`'s, for `hessian_product.cpp`; and `with_qd_inputs`, `qd.cpp`'s, for `lagrangian.cpp` |
| `gradient.cpp` | The builders as einsum terms, forward and backwards, around the SC or PC class walk, and the source term's half-transform |
| `qd.cpp` | `qd_sc_energy`, `qd_pc_energy`, `qd_sc_adjoint` and `qd_pc_adjoint`: every pair's RDMs over the shared builders, γ_SA, and the reverse pass through `sc_effective_hamiltonian_adjoint` or `pc_effective_hamiltonian_adjoint`, the transition and one-vector `rdms_adjoint` and `builders_adjoint`; `with_qd_inputs`, the same inputs handed to a callback |
| `tangent.cpp` | `builders_tangent` and `direction_arrays`: the builders' terms by the product rule over the tangent arrays, the RDMs' through the bra/ket `nevpt2::rdms`, and f's core–core and virtual–virtual blocks |
| `hessian_product.cpp` | `sc_hessian_product` and `pc_hessian_product` over one body (`hessian_product`): the class walk's tangent bars and the one-slot-dotted terms back through the reverse pass (`bars_tangent`), Φ's closed-form tangent, X and A in the MO basis, and the Rayleigh normalization; PC's Schur lift per direction |
| `zc.cpp` | z^c's divisions and the degenerate-pair rule; Φ = ⟨z^c, f⟩ and its adjoints over B whole in the MO basis |
| `lagrangian.cpp` | `lagrangian_gradient`, `sc_gradient` and `pc_gradient`: the Z-vector over the CI-relaxed Hessian, the X^L gate (the Lagrangian Fock and the stencil of its derivative along Z), and the one `df_gradient` pass; `qd_sc_lagrangian` over the same Z-vector and gate at the average's weights, at any symmetric W (the state form a wrapper at u_K u_Kᵀ), and `qd_sc_gradient`'s two passes over it; `qd_sc_coupling`, the same at an interstate pair's W^PQ; `qd_pc_gradient` (state and W forms) and `qd_pc_coupling`, the same Lagrangian and passes over `qd_pc_energy` and `qd_pc_adjoint`; `qd_inputs`, `with_qd_inputs` on the solution's weights and strings |
| `CMakeLists.txt` | The module `free_breakfast.nevpt2.gradient` |
