# `free_breakfast.nevpt2.gradient`

The SC-NEVPT2 correlation energy of one root as a function of the integrals
(h, B) and the CI vector c, and its adjoint: the first piece of the NEVPT2
nuclear gradient (`nuclear_gradient.tex` §5). Then the inactive-rotation
multipliers z^c of its Lagrangian and the term Φ = ⟨z^c, f⟩ they add, with
Φ's adjoints (Algorithm 1, steps 1–2). Link
`free_breakfast::nevpt2::gradient`.

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
    zc_adjoint(res, alpha, beta, naux, nao, nmo, b, c, h, ncore, ncas, vector, z)
        -> ZcAdjoint
        phi        Φ = ⟨z, f⟩ = Σ_pq z_pq f_pq
        h_bar      dΦ/dh = z
        lambda     dΦ/dB^L, as ScAdjoint's
        y          y^c = dΦ/dc, as ScAdjoint's y
        x, a       X and A of (h̄, Λ) above, as ScAdjoint's

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

## The inactive multipliers z^c

SC-NEVPT2 is not invariant under core–core and virtual–virtual rotations, so
the Lagrangian carries the pseudocanonical conditions f_pq = 0 there:
L = E + ⟨z^c, f⟩, f the generalized Fock

    f_pq = h_pq + Σ_rs D_rs [(pq|rs) − ½ (pr|sq)],   D = 2 on the core's diagonal, γ on the active block

whose core and virtual diagonals are ε above (F^I plus the active part).
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
off its diagonal), with g_L = Σ B_L,rs D_rs and w_L = Σ B_L,pq z_pq:

    Φ   = ⟨z, h⟩ + Σ_L w_L g_L − ½ Σ_L tr(z B_L D B_L)
    h̄   = z
    Λ_L = z g_L + D w_L − ½ (z B_L D + D B_L z)
    W   = dΦ/dγ = Σ_L w_L B_L,aa − ½ Σ_L (B_L z B_L)_aa
    y^c = Σ_tu (W + Wᵀ)_tu (E_tu c)
    X   = 2 h z + 2 Σ_L B_L Λ_L,   A = X − Xᵀ

Each left product D B_L and z B_L is one strided-batched gemm over the slow
index (B(·, ·, q) is a packed naux × nmo block), and every sum over (L, r)
one gemm with B read as (naux nmo) × nmo by its symmetry in the pair, as
the source term's half-transform does. γ is `ci::Sigma::rdm1` of the vector,
and y^c one `ci::Sigma::apply` over the one-body operator W + Wᵀ with no
two-electron part: spin-free, so y^c is in c's spin and symmetry sector.

**y^c is raw, as y is**: the derivative of Φ as a function of an
unnormalized c, not projected (⟨c, y^c⟩ = 2⟨W, γ⟩). #563 passes
½ P_c(y + y^c) to `mcscf::zvector`, which is where it is made ⊥ c and put
in that solver's convention (`src/mcscf/zvector/README.md`). The (h̄, Λ, X,
A) of Φ add to `sc_adjoint`'s field by field.

**One root.** Both entry points take one vector, so a state average cannot
reach them; #563, which holds the CASSCF's roots, refuses one
(`Unsupported`).

## Not done

- The PC adjoint (Eq. pcdot), and the per-root canonicalization's own
  derivative: the caller passes canonical orbitals and z^c takes the
  constraint.
- Λ lives at naux nmo² whole; a virtual–virtual block that only its diagonal
  reaches is still stored whole, as `mcscf::df_gradient`'s extra adjoint
  takes it.

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
  - Measured on an RTX 3080, one run: max |z^c| 2.4e-3 and the smallest
    inactive gap 1.43e-3 (the two lowest virtuals, of different C₂ᵥ
    symmetry; `test/gtest/data/README.md`, "Not symmetric water");
    stationarity core–core 8.4e-9, virtual–virtual 2.0e-9 of the slope, and
    Σ κ (A + A^{z^c}) at round-off; Φ's ladders ci 2.3e-11, the six blocks
    4.6e-11 to 3.8e-10, h 2.3e-11, B 4.5e-11 relative. Each z^c case is well
    under a second.

## Files

| File | Role |
|------|------|
| `gradient.cppm` | Module interface: `sc_energy`, `sc_adjoint`, `ScAdjoint`; `zc_multipliers`, `ZcMultipliers`, `zc_adjoint`, `ZcAdjoint` |
| `operands.cppm` | Internal partition `:operands`: the column-major einsum operands and contractions, zeroed buffers, block adds, host copies and the count check both units share |
| `gradient.cpp` | The builders as einsum terms, forward and backwards, and the source term's half-transform |
| `zc.cpp` | z^c's divisions and the degenerate-pair rule; Φ = ⟨z^c, f⟩ and its adjoints over B whole in the MO basis |
| `CMakeLists.txt` | The module `free_breakfast.nevpt2.gradient` |
