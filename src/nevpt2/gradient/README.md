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
        w, rho       W = u_K u_Kᵀ (n², row-major) and ρ_r = W_rr
        weights      the SA weights, mcscf::sa_gradient_weights'
        corr         qd_sc_adjoint at W plus Φ^SA's adjoint: y = ȳ, a = A^corr + A^{z^c}
        zc           z^c
        pairs        τ, ζ and t (mcscf::RootPairMultipliers)
        zeta         ζ · g and (γ^ζ, Γ^ζ) (mcscf::ZetaTerms)
        multipliers  (Z, z, Θ), mcscf::ZVector
        asymmetry    the gate's residues per block

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

**Refusals.** A class `pc_effective_hamiltonian` refuses (no positive
eigenvalue of S_N, no gap at the cut, λ + Δ not positive) is refused by
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

**Refusals.** A class PC refuses (no positive eigenvalue of the metric, no
gap at the cut, a non-positive λ + Δ) is refused by `pc_class_energies_adjoint`
with the energy's error, and `pc_gradient` returns it unchanged: never a
gradient with that class summed in as zero.

## The QD-NEVPT2 gradient

`qd_sc_lagrangian(res, molecule, basis, aux, h, b, solution, ncore, ncas,
state, batch)` is the SC-QD-NEVPT2 **Lagrangian and Z-vector** of QD state
K = `state` (H_eff's eigenvalues ascending) on a DF-SA-CASSCF whose core
and virtuals are canonical over the SA Fock, as the Casscf step leaves them
(`nuclear_gradient.tex` Algorithm qdgrad, steps 1–8). The derivative passes
are #728's; `QdScLagrangian` hands them everything they read. With
W = u_K u_Kᵀ and ρ_r = W_rr = u_rK², the Lagrangian is

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
   E^corr[W] + Φ^SA's (`corr`, the `extra` of #728's `df_gradient`).
3. `mcscf::root_pair_multipliers` on the summed ȳ: τ_sr, ζ on the equal
   pairs, t on the unequal ones; `mcscf::zeta_terms` on ζ: ζ · g and the
   folded (γ^ζ, Γ^ζ) of #728's `df_transition_gradient`.
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

## Not done

- The per-root canonicalization's own derivative: the caller passes
  canonical orbitals and z^c takes the constraint.
- Λ lives at naux nmo² whole; a virtual–virtual block that only its diagonal
  reaches is still stored whole, as `mcscf::df_gradient`'s extra adjoint
  takes it.
- A state average's root of the single-state NEVPT2 (`lagrangian_gradient`,
  `sc_gradient`, `pc_gradient`) is refused; the SA reference is
  `qd_sc_lagrangian`'s, and its derivative passes are #728's.
- PC-QD's Lagrangian.
- The torque (the rotational sum) is not checked; the translational one is.

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
    vector (Sr's metric zero, "no positive eigenvalue") and every virtual
    lowered 1000 Eh through h (Sr's λ + ε_r negative, "not positive"):
    `pc_gradient` fails with `pc_energy`'s error, kind and message.
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
    `h8_ring_sto3g_sa3` (c_1), λ + Δ not positive out to t = 1.25e-3, as
    #729 found; those rungs are NaN and the run is read below them.
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
    virtuals lowered 1000 Eh through h ("state 0: Sr has a denominator …
    not positive"): `qd_pc_adjoint` returns `qd_pc_energy`'s error, kind and
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

## Files

| File | Role |
|------|------|
| `gradient.cppm` | Module interface: `sc_energy`, `sc_adjoint`, `ScAdjoint`; `pc_energy`, `pc_adjoint`, `PcAdjoint`; `qd_sc_energy`, `qd_sc_adjoint`, `QdScAdjoint`; `qd_pc_energy`, `qd_pc_adjoint`, `QdPcAdjoint`; `qd_sc_lagrangian`, `QdScLagrangian`; `zc_multipliers`, `ZcMultipliers`, `zc_adjoint`, `ZcAdjoint`; `lagrangian_gradient`, `sc_gradient`, `ScGradient`, `pc_gradient`, `PcGradient` |
| `operands.cppm` | Internal partition `:operands`: the column-major einsum operands and contractions, zeroed buffers, block adds, host copies, and the count and weight checks the units share |
| `builders.cppm` | Internal partition `:builders`: `ScArrays`, the builders as einsum terms, their forward and reverse runs, `shared_arrays` and `builders_adjoint`, declared for `gradient.cpp` (which defines them) and `qd.cpp` |
| `gradient.cpp` | The builders as einsum terms, forward and backwards, around the SC or PC class walk, and the source term's half-transform |
| `qd.cpp` | `qd_sc_energy`, `qd_pc_energy`, `qd_sc_adjoint` and `qd_pc_adjoint`: every pair's RDMs over the shared builders, γ_SA, and the reverse pass through `sc_effective_hamiltonian_adjoint` or `pc_effective_hamiltonian_adjoint`, the transition and one-vector `rdms_adjoint` and `builders_adjoint` |
| `zc.cpp` | z^c's divisions and the degenerate-pair rule; Φ = ⟨z^c, f⟩ and its adjoints over B whole in the MO basis |
| `lagrangian.cpp` | `lagrangian_gradient`, `sc_gradient` and `pc_gradient`: the Z-vector over the CI-relaxed Hessian, the X^L gate (the Lagrangian Fock and the stencil of its derivative along Z), and the one `df_gradient` pass; `qd_sc_lagrangian` over the same Z-vector and gate at the average's weights |
| `CMakeLists.txt` | The module `free_breakfast.nevpt2.gradient` |
