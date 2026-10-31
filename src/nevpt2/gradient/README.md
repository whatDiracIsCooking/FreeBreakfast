# `free_breakfast.nevpt2.gradient`

The SC-NEVPT2 correlation energy of one root as a function of the integrals
(h, B) and the CI vector c, and its adjoint: the first piece of the NEVPT2
nuclear gradient (`nuclear_gradient.tex` §5). Link
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

## Not done

- The PC adjoint (Eq. pcdot), and the per-root canonicalization's own
  derivative: the caller passes canonical orbitals and the z^c of #568 take
  the constraint.
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
  - Measured on an RTX 3080: ci 3.3e-10; core–core 7.6e-9, active–active
    5.1e-9, virtual–virtual 9.5e-10, active–core 6.6e-9, virtual–core
    1.3e-9, virtual–active 2.1e-10; h 2.6e-10, B 3.6e-10 relative. Each
    ladder is well under a second; the CASSCF setup dominates.
- `SourceTermShapes`: A is X − Xᵀ, h̄ and every Λ^L symmetric.

## Files

| File | Role |
|------|------|
| `gradient.cppm` | Module interface: `sc_energy`, `sc_adjoint`, `ScAdjoint` |
| `gradient.cpp` | The builders as einsum terms, forward and backwards, and the source term's half-transform |
| `CMakeLists.txt` | The module `free_breakfast.nevpt2.gradient` |
