# `free_breakfast.nevpt2.energy`

The eight strongly contracted NEVPT2 class energies of one state on the GPU,
the SC-QD-NEVPT2 effective Hamiltonian of several
([SC-QD-NEVPT2](#sc-qd-nevpt2)) and its reverse pass
([The SC-QD reverse pass](#the-sc-qd-reverse-pass)), and the eight partially contracted ones
and their QD effective Hamiltonian ([PC-NEVPT2](#pc-nevpt2),
[PC-QD-NEVPT2](#pc-qd-nevpt2)) and its reverse pass
([The PC reverse pass](#the-pc-reverse-pass)), and SC's forward tangent
([The SC forward tangent](#the-sc-forward-tangent)) and Hessian-vector
product ([The SC Hessian-vector product](#the-sc-hessian-vector-product)),
and PC's forward tangent ([The PC forward tangent](#the-pc-forward-tangent)),
and SC-QD's ([The SC-QD forward tangent](#the-sc-qd-forward-tangent)) and
Hessian-vector product
([The SC-QD Hessian-vector product](#the-sc-qd-hessian-vector-product)),
each class walking its
external index in batches over the density-fitted slabs; the two effective
Hamiltonians also hand over their amplitudes
([The amplitude hand-over](#the-amplitude-hand-over)). Link `free_breakfast::nevpt2::energy`.

    class_energies(res, blocks, in, batch) -> ClassEnergies
        norms[8], energies[8]   Sr, Si, Sijrs, Sijr, Srsi, Srs, Sij, Sir
        total                   Σ energies, the correlation energy
    class_energies_footprint(ncore, ncas, nvirt, nelecas, batch)
    kBatchFromBudget            batch: each class's widest that the budget holds

    footprints of the other calls, below: sc_effective_hamiltonian_footprint,
    pc_class_energies_footprint, pc_effective_hamiltonian_footprint,
    class_energies_adjoint_footprint, sc_effective_hamiltonian_adjoint_footprint,
    pc_class_energies_adjoint_footprint, pc_effective_hamiltonian_adjoint_footprint,
    class_energies_tangent_footprint, sc_effective_hamiltonian_tangent_footprint,
    pc_class_energies_tangent_footprint, pc_effective_hamiltonian_tangent_footprint,
    pc_class_energies_hvp_footprint, pc_effective_hamiltonian_hvp_footprint,
    pc_metric_lifts_footprint, class_energies_hvp_footprint,
    sc_effective_hamiltonian_hvp_footprint

## The port

A transcription of the NEVPT2 demo's `src/energy/energy.cpp` and
`energy_sc_classes.cppm` (nevpt2-gpu-demo, commit `7c5a674`), which are
themselves the NumPy reference's `np.einsum` calls. Each `deviceEinsum*` and
`deviceTransposeAccum` site is a
[`recipes.einsum`](../../recipes/einsum/README.md) `einsum` or `transpose`
with **the same subscript string, axes and scale**, so a line here is checked
against the demo by reading the two side by side. The demo's other pieces
come with it:

- the three slice assignments the reference does with NumPy indexing rather
  than an einsum (`a16[:, i, :, :, :, i] += fdm2` and a22's two), as
  `nevpt2::device::diag_a16`, `diag_a22a` and `diag_a22b`;
- the Sijrs class, which reads no RDM, as one reduction kernel per slab
  (`nevpt2::device::sijrs`): a grid-stride loop, a `block_reduce` per block,
  an atomicAdd per block;
- PySCF's `_norm_to_energy`, verbatim, on the host, its vanishing-norm guard
  (`NUMERICAL_ZERO`, 1e-14) included: −Σ_k N_k / (Δ_k + H_k / N_k) over the
  perturbers whose norm does not vanish. Sijrs has no guard, as in PySCF;
  nor, since #949, has a class whose perturbers span one function each
  ([One-function classes](#one-function-classes-949)), which keeps every
  perturber at −N_k / (Δ_k + η).

Each class builds its RDM-only intermediates (`make_a16`, …) once, then for
each batch has nevpt2::slab write its external block into a plain device
buffer and contracts that against them. A slab's norm and H vectors are read
back and finished on the host; the finish is elementwise, so splitting it by
slab only reassociates the sum. Srsi fills its whole (v, v, c) tables first,
because its symmetrization pairs (r, s) with (s, r) across slabs.

The demo's `IntegralSource`, `FullBlockSource` and `ActiveIntegralsDevice`
are not ported: the slabs come from `nevpt2::slab` directly
([`density_fitting`](../density_fitting/README.md) "No `IntegralSource`
interface"), and the per-state arrays are `EnergyInputs`.

## Layouts

- **Integrals column-major**, first index fastest, as
  `nevpt2.density_fitting` writes them: `h1e`, `h2e` (nevpt2::h2e's
  physicist (tv|uw)), the slabs, `h1e_v_sr` (v × a), `h1e_v_si` (a × c),
  `h1e_v_sir` (v × c). A slab, or a run of h1e_v's rows, enters an einsum
  through its strides, so nothing is transposed to feed it.
- **RDMs row-major**, PySCF's layout: `dm1`, `dm2`, `dm3` (`make_dm123`'s, not
  reordered), `f3ca`, `f3ac`, as `nevpt2::rdms` writes the last three.
- **Every intermediate row-major**, because `einsum` writes its output packed
  row-major. That is the demo's layout, so its subscripts, the diagonal
  slices' index arithmetic and its host finish all carry over as written.

## Frozen core

The layer has no frozen-core switch: a frozen core is what the caller hands
it (#859, `pipeline/nevpt2` "Frozen core"). With the lowest n_f canonical
core orbitals frozen, they leave every class that carries a core index, six
of the eight:

| class | core indices | frozen orbitals leave |
|---|---|---|
| Si, S(+1)′ | i | its i |
| Sijrs, S(0) | i, j | its i and j |
| Sijr, S(+1) | i, j | its i and j |
| Srsi, S(−1) | i | its i |
| Sij, S(+2) | i, j | its i and j |
| Sir, S(0)′ | i | its i |
| Sr, S(−1)′; Srs, S(−2) | none | nothing: unchanged |

So the caller builds `MoBlocks` over the orbitals from n_f on (`mo_blocks`
over C's columns n_f…, its `ncore` the correlated core alone), and gives
`e_core`, `h1e_v_si` and `h1e_v_sir` over the correlated core. **The frozen
orbitals stay in the core density**: h1e, every `h1e_v_*` value and so
Dyall's Hamiltonian are the inactive Fock's over the whole core, unchanged.
Sr and Srs then come out exactly as without freezing, which the reference
checks (`tools/emit/nevpt2_reference.py --frozen-core`). Everything indexed
by "core" below (`ScAmplitudes::si`, a slab's batch over i, …) is the
correlated core.

## Memory

The intermediates and the slab buffers are `Buffers::device` of their own,
freed when their class returns. The largest are the n_act⁶ ones (a16, a22,
dm3_h, hdm3, rm3): at most two at once. `fdm2` (make_a16's one, make_a22's
two), n_act⁴ each, is the workspace's: fixed sizes, independent of the batch.

**The batch.** A walk's buffers are its slabs (two for Sir and Srsi), the
gemm product `nevpt2::slab` holds while it builds one (its own buffer, never
the workspace), and the slab's norm and H tables (PC: its y rows), all
allocated once before the walk at the batch's width. 0 is each class's whole
extent; `kBatchFromBudget` (any negative batch) sizes each class's batch
when its walk starts, its intermediates built and its PC solve done: the
widest whose walk fits `res.memory_budget()` less the pool's 64 MiB
fragmentation headroom (`pipeline/footprint`'s README), up to the whole
extent, and `Unsupported` when a batch of one does not fit. The walk is the
class's last allocation, so it may take the rest. The adjoints, tangents and
HVPs take a fixed batch only: their callers pass 0.

**The footprint.** `class_energies_footprint(ncore, ncas, nvirt, nelecas,
batch)` is the call's peak, class by class in its order: each make_* and its
temporaries, then the walk at its widest (`batch_step`'s step, or one for
`kBatchFromBudget`: the least the call runs with); its workspace the fdm2s.
It is persistent-free (the result is on the host). `nelecas` matters only at
two active electrons (Srs) or two holes (Sij), whose one-function η builds
the class's blocks once more. `test/gtest/nevpt2_footprint_gpu.cpp` holds it
sound and tight at CAS(4,4) and CAS(8,8), at a batch of one and whole; has
the batch from the budget take more than one and stay under a pool cap; and
the SC and PC energies at the least (rdms at one determinant, a batch of one)
equal the unconstrained ones to 1e-10 Eh.

**The other footprints** are mirrors of their calls the same way, each
allocation taken and given back in its order:

- `sc_effective_hamiltonian_footprint(ncore, ncas, nvirt, nelecas, n,
  diagonal, transitions, batch)`: n states, the diagonal given or not and the
  first `transitions` off-diagonal entries (the call shapes the step and the
  gradient make: the diagonal alone, a pair beside its transpose, every
  entry); each entry's identity, then a class at a time. `class_energies`'
  is one state's.
- `pc_effective_hamiltonian_footprint(ncore, ncas, nvirt, n, diagonal,
  transitions, batch, syevd_lwork, amplitudes, dropped)`: per own state its
  matrices and solve, per transition its T and T₁, then the walk; with
  `amplitudes` every class's U_J is the call's product, the kets handed over.
  Its workspace is the fdm2s and diagonalize's at each solve's two orders, d
  and m, read from `syevd_lwork` (the pipeline's `FootprintDimensions::
  syevd_lwork`). **The kept rank m is the data's**: `dropped[J][k]` is d - m
  of state J's class k, `PcClassEnergies::dropped`; empty, every class is at
  its whole basis, the most the forward holds. `pc_class_energies_footprint`
  is one state's, no kets kept.
- `sc_effective_hamiltonian_adjoint_footprint(naux, ncore, ncas, nvirt,
  nelecas, n, batch, transition)` and `class_energies_adjoint_footprint`
  (one state): the block bars, each class's reverse walk (its blocks, slabs
  and their bars, each slab's tables and multipliers), the sweeps, Sijrs and
  the orbital energies' upload; over n > 1 the states' evaluation walks and
  each pair's metric bars and transition RDM bars, `transition` the footprint
  of what the caller's transition does. Every bar is wanted, as the
  gradient's; the workspace adds the slabs' adjoints'.
- `pc_effective_hamiltonian_adjoint_footprint(naux, ncore, ncas, nvirt, n,
  batch, syevd_lwork, dropped, transition)` and
  `pc_class_energies_adjoint_footprint`: the kets with their dropped modes
  kept, the walk bars, the slabs and their rows, the finish (the span's bar,
  each pair's matrix bars and their way back). A solve that drops modes keeps
  their V (d²) and leaks, so the most is not always at m = d: with `dropped`
  empty each class is at the rank, m from 1 to d, that holds the most.
- `class_energies_tangent_footprint(naux, ncore, ncas, nvirt, nelecas, batch,
  directions)` and `sc_effective_hamiltonian_tangent_footprint(..., nelecas,
  n, batch, directions)` (#1182): the SC forward tangents, the reverse walks'
  mirror (one `ScWalkBytes` in `reverse.cpp`, the adjoints' too) in tangent
  mode. Per class its blocks and each direction's tangent of each, held to
  the class's end, each built beside the shifted inputs and its builder's own
  peak; at the widest slab the tables and their tangents and each slab's
  tangent buffer; B + s Ḃ and slab_tangent's two builds, held across classes
  at the widest slab yet; a one-function class's η̇; Sijrs's walk. Over n
  states, each state's own tangent and each pair's walk along both sides'
  directions. Only which fields a direction gives is read: a field given
  counts as moving (a step of zero is a direction that is zero).
- `pc_effective_hamiltonian_tangent_footprint(naux, ncore, ncas, nvirt, n,
  batch, syevd_lwork, directions, dropped)` and
  `pc_class_energies_tangent_footprint` (#1183): the PC forward tangents, the
  adjoint's walk mirror (one `PcWalkBytes` in `reverse.cpp`) in tangent mode.
  Per class the kets as the adjoint keeps them, M and M_Δ on every entry,
  then `pc_solve_tangent`'s: per ket and direction Ṫ, ṫ1, K̇' and Ṡ', and
  per transition Ṫ and ṫ1, held to the class's end; each polarization's
  three matrices beside the shifted inputs and the builder's own peak; Θ'
  where the ket drops modes. At the widest slab the rows and their tangents
  and each slab's tangent buffer; B + s Ḃ and slab_tangent's two builds held
  across classes at the widest slab yet; then Sijrs's walk. `dropped` reads
  as the adjoint's, and a direction is read as the SC tangents read it.
- `class_energies_hvp_footprint(..., nelecas, batch, directions, partners)`
  and `sc_effective_hamiltonian_hvp_footprint(..., nelecas, n, batch, hbar,
  directions, partners, transition)` (#1180): the SC Hessian-vector
  products over those tangents and the adjoints' walk. Per direction the
  empty perturbers' second tables (each rung's shifted inputs beside a
  tangent, and over n states once per pair beside `pair_tables`), the fixed
  walk at the multipliers' tangent, the bars at fixed multipliers
  differenced (`line_derivative`'s scratch, each rung's shifted inputs beside
  a fixed walk, each pair's with its transition bars), and Sijrs's
  (`sijrs_hvp`, its scaled buffers where Σ H̄_JJ is not 1). **At the most**:
  the second tables are counted as though every ket has an empty tuple a
  direction populates, empty for every bra too, which N₂'s symmetry gives;
  without one the footprint is sound and over by them.

- `pc_effective_hamiltonian_hvp_footprint(naux, ncore, ncas, nvirt, n,
  batch, syevd_lwork, directions, dropped, transition)` and
  `pc_class_energies_hvp_footprint` (#1184): the same mirror's other two
  passes. The base walk is the tangent's with `pc_hvp_slab`'s rows and each
  ket's record kept to the call's end (U, S's modes, M and M_Δ and their
  tangents, and where the span rotates U̇ and the span bar's derivative,
  `pc_hvp_keep`'s); each fixed walk is the adjoint's with the kets rebuilt
  (`pc_fixed_kets`). Per direction: the transitions' bars, the multipliers'
  fixed walk, U's difference and the inputs' (each beside
  `line_derivative`'s scratch bars, the latter beside every pair's shifted
  inputs), and `sijrs_hvp`. Every bar is wanted, as the gradient's; at more
  than one state Sijrs's weight Σ H̄_JJ is taken as not one.
- `pc_metric_lifts_footprint(ncore, ncas, nvirt, n, syevd_lwork, dropped)`,
  both `pc_metric_lifts` (n = 1 the one-state overload): per class the solve
  with its modes kept, the path's metrics, L and its eigensolve, and per bra
  the rows' parts. The null modes are a prefix of the dropped ones; every
  dropped mode is counted null, the most.


`test/gtest/nevpt2_footprint_gpu.cpp` holds `pc_class_energies`, both
reverse passes, the SC and PC tangents, the SC and PC products (PC's at the
Hessian's floor, 1e-3) and the PC metric lifts sound and tight at CAS(4,4)
and CAS(8,8), at a batch of one and whole, at the run's own ranks (the
tangents along a direction moving every field and B beside one moving the
RDMs alone, the effective Hamiltonians' over two states, SC's one-function
classes at CAS(4,4), and SC's products with and without partners); the
effective Hamiltonians' call shapes are in
`test/gtest/pipeline_step_footprint_gpu.cpp`'s
`Nevpt2ReaderFootprint.EffectiveHamiltonians`.

## One-function classes (#949)

**Two classes can span one function per perturber.** Srs at two active
electrons: E_rp E_sq |Ψ⟩ removes both, which leaves the active vacuum, so
pair (r, s)'s perturber is one amplitude A_rs = Σ_pq W_pq ⟨vac|a_q a_p|Ψ⟩
(W the external integrals) times a fixed function. Sij at two active holes
is the same: filling both leaves the closed shell. So N_t ∝ A_t², H_t =
η N_t with η that one function's Dyall energy, one number per state and
class, and c_t = −1 / (Δ_t + η) does not depend on A_t. Every CAS(2,2) has
both.

**PySCF's guard drops a perturber the energy needs.** A_t is a smooth signed
function of geometry, so it vanishes on a surface, and near it N_t falls
under 1e-14 and `_norm_to_energy` drops the perturber. Its couplings
T^t_IJ ∝ A^I_t A^J_t are linear in A_t, so H_eff's off-diagonal moved at
first order through a perturber the derivative at a dropped point did not
see: 3e-8 to 6e-8 Eh/bohr of SC-QD gradient at ethylene's Py, within 1e-5
bohr of such a surface (`../gradient/README.md`, "What the SC-QD miss at Py
was", #894). Lowering the guard is not the fix: where symmetry zeroes A_t,
N_t and H_t are round-off, and H_t / N_t is noise next to a pole.

**So a one-function class keeps every perturber, at η from its blocks.**
`single_function_class` tells one by tr dm1 (the active electrons, rounded,
`ActiveOperands::nelecas`): Srs at 2, Sij at 2 n_act − 2.
`single_function_blocks` reads the class's Dyall block X (a7, a9) and metric
Y (rm2 read `pqba`, hdm2) in the pairing the H and norm einsums contract
with W_pq W_ab, and η = ⟨X, Y⟩ / ⟨Y, Y⟩ (`single_function_ratio`). On a
state's own RDMs X = η Y to round-off (‖X − ηY‖/‖X‖ 1e-15 to 6e-15 on
h6_631g_sa2's Srs and h2co_ccpvdz_sa2's Sij), so η is every perturber's
H_t / N_t, read with no amplitude and no division by a small N. The
per-perturber quotient is not: at h2co_ccpvdz_sa2's 1A2 root moved 1e-6 off
C₂ᵥ, its Sij perturbers' H_t / N_t (N_t ~ 1e-12) ran from 0.620 to 0.703
against η 0.70291. A class every one of whose amplitudes symmetry zeroes
(that root's whole Sij at C₂ᵥ) is no longer special either: η is defined
there, and its couplings grow linearly off it. A zero Y (no such state)
falls back to the guard.

Every pass reads that η:

- **The energy, the QD walk and `ScAmplitudes`**: c_t = −1 / (Δ_t + η^J)
  for every perturber, N and H of the class read by nothing but T.
- **The reverse pass** (`sc_couplings_adjoint`): H̄_JJ c_t on N_t, nothing
  on H_t, D̄_t = H̄_JJ N_t c_t² + A^J_t on Δ_t, and η̄ = Σ_t D̄_t onto the
  blocks after the walk: η̄ Y / S on X and η̄ (X − 2ηY) / S on Y, S = ⟨Y, Y⟩
  (`single_function_eta_bars`), which the sweep takes to the RDMs and
  integrals. Their Eq. 40 share is η̄ (η + η − 2η) = 0.
- **The forward tangent**: Ė = Σ_t c_t Ṅ_t + N_t c_t² (Δ̇_t + η̇), η̇ =
  (⟨Ẋ, Y⟩ + ⟨X, Ẏ⟩ − 2η ⟨Y, Ẏ⟩) / S from the blocks' central polarization,
  exact as `block_tangent`'s.
- **The Hessian-vector product**: P = c_t, Q = 0 and R = N_t c_t², the block
  bars as fixed multipliers of `fixed_adjoint` and their tangent through η̄̇
  and (Ẋ, Ẏ). Such a class is never empty, so never takes K6.

Off the RDMs' rank-one structure (the tests' raw directions) X is not η Y,
and η is their least-squares ratio: one smooth function, which every pass
differentiates exactly. PC is unchanged: its solve has no guard of this
kind.

**Measured** (A100, `--preset default`, card held): at ethylene's Py, every
SC-QD check that missed now passes, and the energies move by at most
6.5e-13 (`../gradient/README.md` has the table). The tests:
`Nevpt2EnergyOneFunction` (below).

## SC-QD-NEVPT2

    sc_effective_hamiltonian(res, blocks, pairs, batch) -> vector<ClassEnergies>
        pairs        n x n EnergyInputs, row-major, [I n + J] bra I, ket J
        [I n + J]    norms[k] Σ_t T^t_IJ, energies[k] ⟨I|V|Ψ_J⁽¹⁾⟩ of class k,
                     total Σ_k; the diagonal is each state's class_energies

`qd.cpp` checks the inputs; the walk is `class_energies`' own
(`sc_couplings`, `energy.cpp`), which runs it at n = 1. The states share
their orbitals, so every pair carries the same integrals, orbital energies
and h1e_v pointers (checked), and its own transition dm1, dm2, dm3 in
`nevpt2::rdms`' bra/ket convention (dm1[p, q] = ⟨I|E_pq|J⟩). f3ca and f3ac
are read on the diagonal only. The conventions, and what the brute force
found, are `test/gtest/data/qd_nevpt2/README.md`'s:

- **The coupling** of class k is −Σ_t T^t_IJ / (Δ_t + H^J_t / N^J_t), over
  the perturbers with |N^J_t| > 1e-14 (every perturber, at H/N = η^J, in a
  [one-function class](#one-function-classes-949)): the denominator is the ket's, as
  `class_energies` forms it, and the norm the transition one. At I = J it is
  `_norm_to_energy` itself, so the diagonal is `class_energies`' arithmetic,
  einsum for einsum.
- **T^t_IJ** is the class's norm einsum over the transition RDMs' norm
  intermediates (dm3_h, dm2_h, dm1_h, hdm1, rm2, hdm2, or the RDMs), with
  the terms that read no RDM dropped: they are ⟨I|J⟩ = 0 times a number
  (`make_hdm1`'s and `make_hdm2`'s `overlap`, Sir's h1e_v·h1e_v, all of
  Sijrs). Then **the mean over (I, J) and (J, I)**: PySCF's `* 2.0` in Sr's,
  Si's and Sir's cross terms folds two Hermitian conjugates into one, which
  holds only for a state's own RDMs. The other five are symmetric already,
  and the mean leaves them as they are.
- **The intermediates.** The Dyall blocks (a16, a17, …) are read only for
  H^J, so they are built from each state's own RDMs, as before. The norm
  intermediates are the only builders that see a transition RDM, and each
  is PySCF's einsum as written, assuming no symmetry of its RDM.

Each class walks its slabs **once** for every pair. Per slab its einsums
accumulate into one table per given pair and one per given state
(`pair_table_slots`), one value per perturber: pair (I, J)'s norm einsum, so
state J's own N^J on the diagonal, and state J's H^J. The host finish
(`add_pair_shares`) then forms every coupling from them. The memory is a set
of Dyall blocks per given state and of norm intermediates per given pair,
all held across the walk (dm3_h is n_act⁶ per pair), and the slab tables
grow with the given entries; Srsi's whole (v, v, c) tables the same way.

**Omitted entries** (#689). An entry whose dm1 is null is not built, walked
or read, and its element is zero; `qd.cpp` checks that every diagonal is
given or none, and each off-diagonal beside its transpose. With the diagonal
omitted, the kets come from the `ScAmplitudes` a call with those diagonals
filled ([The amplitude hand-over](#the-amplitude-hand-over)): element (I, J)
is Σ_t T^t_IJ c^J_t, the coupling above with −1 / (Δ_t + H^J_t / N^J_t)
divided out once, by that call, and Sijrs (diagonal only) is not walked. So
n states take one call over the n diagonals, which walks each state's classes
once, then one per pair over (I, J) and (J, I) alone, which walks only the
transition norms: the pipeline's QD step, holding two transition sets at a
time and never n². It agrees with the whole n × n call to round-off: the
division moves, and the atomics reorder
(`Nevpt2QdEnergyPairs.DiagonalOnceMatchesPerPairPath`, 1e-10 of the pair's
largest diagonal class energy; at most 2e-13 on the ten cases).

The matrix is returned unsymmetrized; its symmetrization and
diagonalization are the caller's.

## PC-NEVPT2

    pc_class_energies(res, blocks, in, batch) -> PcClassEnergies
        energies[8], dropped[8]   the same eight classes, and each metric's cut
        cut[8], kept[8]           the largest dropped σ and the smallest kept one,
                                  over S's largest (#917)
        total                     Σ energies
    in.pc_floor: a relative floor on the cut, 0 by default (#917)

`pc.cpp`, a transcription of the demo's `energy_pc_solve.cppm`,
`energy_pc_classes.cppm` and `energy_pc.cpp` (commit `7c5a674`, unchanged
since). The demo's `docs/pc-nevpt2.md` has the derivation and every measured
figure; in short, **PC is SC without the final contraction.** Each SC class
takes two quadratic forms of a slab row x_t over the same two matrices,
N_t = x_tᵀ S x_t (the metric: dm3, hdm2, rm2, ...) and H_t = x_tᵀ K x_t (the
Dyall Hamiltonian: a16, a9, a7, ...), and contracts them into one function.
PC keeps the span:

    E_t = −b_tᵀ (K + Δ_t S)⁺ b_t,   b_t = S x_t + S₁ h1_t

S and K are the same for every tuple, so each class is **solved once**
(`solve_class`): S and K symmetrized (K is symmetric only to the CI vector's
convergence), S = V σ Vᵀ by `recipes::diagonalize`, the cut, X = V_kept
σ_kept^−½, K′ = Xᵀ K X by `gemmwich`, K′ = W λ Wᵀ, U = X W, T = S U and
t1 = S₁ᵀ U. Each slab is then one einsum into that eigenbasis,
y_t = Tᵀ x_t + t1ᵀ h1_t, finished on the host as
E_t = −Σ_k y_tk² / (λ_k + Δ_t). No new intermediate: every S and K is a
`make_*` SC already has, which is why those live in the `:shared` partition
(declared there, defined in `energy.cpp`). SC's one-body a17, a19, a23 and
a25 are not read: the one-body part enters only through S₁ h1_t. Sijrs has
no active index, so it is SC's `energy_sijrs`.

| class | basis, d | S | K | S₁ | Δ_t, factor |
|---|---|---|---|---|---|
| Sr | (p, q, r), n³ | dm3[r,p,q,b,a,c] | a16 | dm2[r,p,q,a] | ε_r, 1 |
| Si | (p, q, r), n³ | dm3_h[r,p,q,b,a,c] | a22 | dm2_h[r,p,q,a] | −ε_i, 1 |
| Sijr | p, n | hdm1 | a3 | | ε_r − ε_j − ε_i, 2 direct − exchange |
| Srsi | p, n | dm1 | k27 | | ε_r + ε_s − ε_i, 2 direct − exchange |
| Srs | (p, q), n² | rm2[p,q,b,a] | a7 | | ε_r + ε_s, ½ |
| Sij | (p, q), n² | hdm2 | a9 | | −ε_i − ε_j, ½ |
| Sir | (p, q) ⊕ (p, q), 2n² | the two spin couplings' 2 × 2 blocks of dm1/dm2 | of a12, a13 | [2 dm1; −dm1] | ε_r − ε_i, 1 |

**Layouts.** S and K are row-major d × d, as every intermediate here is.
Symmetrized, a row-major matrix is its column-major self, so the solve runs
column-major in BLAS and the solver unchanged. T and U come out column-major
d × m, and enter einsum through their strides: T's basis index row-major at
unit stride, k at stride d.

**The cut, and refusals.** S is singular by construction (Sr and Si drop 44
of 64 modes at CAS(4,4), 10 of 1000 at CAS(10,10); Sir 1 or 2). Its modes at
or below **max(1e-13, 100 ε √d)** of its largest eigenvalue are dropped
(ε = 2.2e-16, d the class's basis size), or at `EnergyInputs::pc_floor` when
that is higher ([The conditioning floor](#the-conditioning-floor)): 1e-13 up
to d = 20, then 3.3e-13 at
d = 216, 7.0e-13 at d = 1000 and 9.2e-13 at d = 1728, still more than a
decade below the first real mode on cc-pVTZ (2.1e-11). A class is a
**Numerical error, naming it**, when S has no positive eigenvalue, or when a
ket meets a pole ([Poles, not signs](#poles-not-signs), #874): a
denominator λ_k + Δ_t at round-off, or an intruder, a mode whose
first-order amplitude |y_tk / (λ_k + Δ_t)| exceeds 1. There is no sign
test and no gap test (#853, below). The demo reported a refused class and
carried on; here the call fails, since a total without that class is not
PC-NEVPT2 (the pipeline step then drops that root's PC and keeps its SC).

**The floor: clear of the noise.** What the cut must not keep is a mode
that round-off or the RDMs' error decides: its σ is noise, so its λ in
K′ = Xᵀ K X is noise over noise, and its denominator anything. Measured, the
largest such mode is 7.4 ε √d (6.6e-15, `h8_ring_sto3g_sa3`'s Sij, 15 of 16
dropped, on the QD goldens; 1.2e-15 on the single-state ones). On PSB3
(below) the null cluster tops out at 0.21 ε √d, and the modes where this
layer's spectrum and an exact fp64 one (PySCF's dense RDMs, the same
orbitals) disagree, so round-off decides them, reach 0.53 ε √d (1.75e-15
exact against 2.0e-16 here). 100 ε √d sits at least 13 times above every
one of them, at every d; 1e-13 alone was 30 ε √d at d = 216 and would be
under 10 ε √d past d = 2000. Below the noise the hazard is real: cut at
1e-16 instead, PSB3 FC root 1 keeps a null mode of Si and its denominator
comes out λ + Δ = −4.4, refused.

**No gap test (#853).** Until #853 a class was also refused when its
smallest kept eigenvalue was under 1e3 times its largest dropped one (or
ε √d of the largest, nothing dropped), on the argument that the rank, and
so the energy, would depend on where the cut fell. That refused every PSB3
geometry at SA3-CASSCF(6,6)/cc-pVDZ (cc-pVDZ-JKFIT; the ground-state
minimum and C13=C14 twisted to 90°), every root, on Sr or Si at ratios 1.4
to 630, and HF/cc-pVDZ CAS(10,7) with its F 1s active at 1.7. The premise
does not hold there: the modes at the cut are **real** (this layer's
spectrum and the exact one agree mode for mode from 1e-14 up), they run
continuously through it with no gap anywhere, and they carry **no energy**.

Each class's metric, relative eigenvalues (RTX 3080, `--preset default`;
PSB3 FC from the CASSCF minimum, 90° the same geometry with C13=C14 twisted
rigidly at its FC length):

| state, class (d) | modes under 1e-15 (largest) | the next | modes in 1e-15 … 1e-9 | dropped at 1e-13 / at the cut |
|---|---|---|---|---|
| PSB3 FC root 0, Sr (216) | 6 (2.7e-17) | 6.5e-15 | 19 | 9 / 10 |
| root 0, Si | 7 (2.0e-16) | 1.1e-14 | 18 | 10 / 11 |
| root 1, Sr | 7 (3.9e-17) | 2.1e-13 | 14 | 7 / 8 |
| root 1, Si | 7 (4.4e-16) | 2.1e-14 | 15 | 8 / 9 |
| root 2, Sr | 7 (4.4e-17) | 1.4e-13 | 14 | 7 / 8 |
| root 2, Si | 7 (1.2e-16) | 3.1e-14 | 14 | 8 / 9 |
| PSB3 90° root 1, Sr (216) | 11 (5.2e-16) | 2.6e-15 | 45 | 18 / 22 |
| root 1, Si | 12 (6.8e-16) | 2.3e-15 | 42 | 19 / 22 |
| HF CAS(10,7), Sr (343) | 14 (4.6e-17) | 5.8e-14 | 67 | 16 / 21 |

The energy against the cut, everything else this layer's (E_corr, and the
PC-QD state 0 energy, Eh; a probe build that took the cut from the
environment and skipped the gap test):

| cut | FC root 0 | FC root 1 | FC root 2 | FC QD state 0 | 90° QD state 0 | HF E_corr |
|---|---|---|---|---|---|---|
| 1e-15 | −0.7833187827 | −0.8103104954 | −0.7862713113 | −249.0621766021 | −248.9730127974 | −0.1415607189 |
| 1e-14 | −0.7833187827 | −0.8103104954 | −0.7862713113 | −249.0621766021 | | −0.1415607189 |
| 1e-13 | −0.7833187827 | −0.8103104954 | −0.7862713113 | −249.0621766021 | −248.9730127972 | −0.1415607187 |
| 1e-12 | −0.7833187827 | −0.8103104952 | −0.7862713113 | −249.0621766021 | | −0.1415607163 |
| 1e-11 | −0.7833187826 | −0.8103104949 | −0.7862713112 | −249.0621766021 | −248.9730127962 | −0.1415606856 |
| 1e-10 | −0.7833187821 | −0.8103104929 | −0.7862713108 | −249.0621766021 | | −0.1415605477 |
| 1e-9 | −0.7833187811 | −0.8103104912 | −0.7862713086 | −249.0621766008 | −248.9730127825 | −0.1415594819 |

From 1e-15 to 1e-12, 3 to 7 more modes of each Sr and Si dropped, PSB3
moves by at most 3e-10 Eh and HF by 2.6e-9; FC's PC-QD state 0, the
energy #848 needs, by none in ten decimals from 1e-15 to 1e-10. A real mode
of small σ carries energy of its own order: b = S x + S₁ h1 has a component
along it of that order, so dropping it or keeping it is a σ-sized change,
which is why the cut may fall anywhere in the continuum. The exact fp64
solve in Python on PSB3 FC's Sr says the same: root 0's E(Sr) moves 2e-8 Eh
across 1e-15 … 1e-8, while 28 modes leave. Park's own PC (BAGEL's SMITH,
"CAS/A ... equivalent to partially contracted NEVPT2"; `src/smith/denom.cc`
orthonormalizes each metric by `Matrix::inverse_half(thresh_overlap)`)
keeps eigenvalues above an **absolute** 1e-9 (`smith_info.cc`'s default),
about 4e-11 of the largest here, and has no gap test: the rule his PSB3
QD-NEVPT2 numbers (JCTC 2020, acs.jctc.9b00919, SI S8–S11) came through.

`h8_sto3g_sa3` root 0's Srs and Sij (d = 16) keep real modes at 2.3e-12 of
their largest, and nothing below; the cut, 1e-13 there, leaves them all
(#687 had accepted it against the gap rule's ε √d floor). Every golden
test passes unchanged under the new cut, at its own bar (#853's tier and
slow-tier runs). The
generators cut the same way (`tools/emit/nevpt2_reference.py`'s `pc_cut`,
`qd_nevpt2_reference.py`'s `pc_solve` and brute-force spans) and gate on no
gap; `qd_nevpt2.json`'s `min_gap` is that ratio, recorded, not a gate.

At (2e,2o) nothing comes near the cut along a whole PC-QD MECI walk:
ethylene's SA3-CASSCF(2,2)/cc-pVDZ, PC-QD states 0 and 1, converged in 83
evaluations from a twisted, pyramidalized start with no class refused at
any of them (`docs/reproduction/park2019/README.md`, "Tier 0", #843).

### Poles, not signs

Until #874 a class was also refused when any denominator λ_k + Δ_t was not
positive. That refused ethylene's S0 minimum at Park's SI geometry
(SA3-CASSCF(2,2), aug-cc-pV{D,T}Z / aug JKFIT), root 2, the π*² (Z) state,
whose PC-QD energies Park reports (acs.jctc.9b00919 SI p. S2). For a ground
state Dyall's denominators are positive; for an excited reference a
perturber may lie below it, and λ + Δ is then negative. Here they are
Sr's π¹ r¹ configurations, r a diffuse virtual, below the Z state, as the
π → Rydberg states lie below it. The term −y² / (λ + Δ) is as well defined
there as above zero; it fails only at a pole.

- **BAGEL** (qsimulate-open/bagel, ad8c7c8), whose SMITH CAS/A is Park's
  PC, refuses nothing. `src/smith/denom.cc:515-517`: each metric is
  `inverse_half(thresh_)` then `diagonalize(denom_)`, no sign read.
  `src/smith/spinfreebase_update.cpp:132` (and 200, 264, 324, 394, 452,
  517, 582): `interm[iall] /= min(-0.1, e0_ - (denom + eig...))`, which is
  `update_amplitude`, CAS/A's **preconditioner** only
  (`src/smith/casa/CASA.cc:182, 229`); the amplitudes come from `LinearRM`
  (`CASA.cc:185-215`), unclamped. Park's Eq. 12 divides with no condition;
  his gradient is an in-house program (CAS/A has none, `CASA.cc:256`).
- **SC already takes them.** `norm_to_energy` (PySCF's `_norm_to_energy`)
  guards only |N| > 1e-14. At ethylene S0, SC's state 2 Sr has one
  perturber below zero, carrying exactly PC's negative-denominator energy.
- **Park's numbers include them.** Measured, RTX 3080, `--preset default`
  (PC with the sign test removed, then this rule; the energies are the same):

| ethylene S0 | PC-QD state 2's Sr: below zero, the lowest | smallest \|λ + Δ\| (its \|t\|) | their energy | SC state 2's Sr below zero |
|---|---|---|---|---|
| aug-cc-pVDZ | 7, −0.1223 | 9.6e-3 (5e-13) | +3.227654e-3 | 1, −0.0431, +3.227654e-3 |
| aug-cc-pVTZ | 10, −0.1388 | 1.9e-3 (7e-7) | +8.342916e-4 | 1, −0.0754, +8.342916e-4 |

  PC-QD then gives, against Park's (his frozen core, so Δ = E − E_Park is
  the core offset `docs/reproduction/park2019/single_points/README.md`
  measured at the MECIs, #860):

| | E₀, E₁, E₂ (Eh) | Δ₀, Δ₁, Δ₂ (mEh) | the MECIs' mean Δ (spread) |
|---|---|---|---|
| aug-cc-pVDZ | −78.339968077678, −78.063572639838, −77.857138939295 | −5.515, −5.505, −5.497 | −5.417, −5.400, −5.465 (≤ 0.18) |
| aug-cc-pVTZ | −78.446937242316, −78.165961397939, −77.954122569329 | −35.548, −35.446, −35.374 | −35.117, −35.105, −35.169 (≤ 0.71) |

  Without the negative terms Δ₂ at aug-cc-pVDZ would be −8.7 mEh, 3.2 mEh
  off the band its three neighbours sit in to 0.1.

**The rule** (`pc_intruder_check`, one check for the energy, the effective
Hamiltonian and the reverse pass, per class and ket):

- **an intruder**: a mode whose amplitude t = y_tk / (λ_k + Δ_t), the
  weight of the normalized perturber in Ψ⁽¹⁾, has |t| > 1. A perturber that
  outweighs its reference is where perturbation theory fails, and it is
  what a pole does: a denominator near zero refuses only through a mode it
  couples. One it does not couple is harmless: ethylene S0's nearest
  denominators (−1.6e-3 at aug-cc-pVDZ root 2 alone, −1.7e-3 at
  aug-cc-pVTZ) carry |t| of 3e-12 and 7e-7.
- **a denominator at round-off**: |λ + Δ| at most 100 ε √d of the class's
  largest |λ + Δ| (the cut's noise multiple, d its basis size). λ is an
  eigenvalue of K′, known to about ε ‖K′‖; below that its sign is noise,
  and a mode y = 0 there would pass |t| as 0 / 0.

The margin, the largest |t| of any class over Park's SI geometries
(`[nevpt2] quasi_degenerate`, every root and the QD kets): ethylene
aug-cc-pVDZ S0 0.274 (Sr, λ + Δ = −0.043), aug-cc-pVTZ S0 0.197, the Py,
Et and Hm MECIs 0.05–0.08; PSB3 cc-pVDZ S0, S1, CI(11–12) and CI(13–14)
0.04–0.08. Every golden passes unchanged. The message names the class, the
state, and the mode's amplitude and denominator
(`state 0: Sr has an intruder: amplitude y / (λ + Δ) = 4.000e+00 at
λ + Δ = …, beyond 1`). Each element's `PcClassEnergies::strongest` and
`nearest` report the two modes the check reads.

**The reverse pass** needs nothing for the sign: Ė = −c (2 xᵀ ḃ − xᵀ Ȧ x)
with x = A⁺ b holds for any invertible A on the kept span, and its
multipliers g = y / (λ + Δ) read no sign. Measured at ethylene S0
aug-cc-pVDZ, by "The derivative across a continuum"'s method (below:
directional central differences of `[nevpt2_gradient] quasi_degenerate,
variant = "pc"`'s energies at 2e-3, 1e-3 and 5e-4 bohr, one Richardson
step, the bar max(2 × its bar, 3 σ_E / h_min)), RTX 3080, `--preset
default`, card held: σ_E = 2.5e-13 Eh over 5 reference repeats, so
3 σ_E / h_min = 1.5e-9. Along each state's gradient and two random unit
directions (`default_rng(874)`), state 2 (its own ket's Sr negative) and
state 0 (through H_eff), all six pass: state 2 misses by 4.4e-10, 2.0e-10
and 4.9e-10 at slopes 0.474, 0.061 and 0.026; state 0 by 4.6e-10, 3.6e-10
and 5.0e-10 at 2.2e-3, 5.2e-4 and −1.1e-4, every bar 1.5e-9 to 2.5e-9.

**The derivative across a continuum.** The reverse pass (below)
differentiates the solve over S's kept span, and since #862 it also takes
that span's rotation into the dropped modes, which past #853 are real.
Measured at PSB3 before #862, holding the span still did not move the
derivative past this bar but for one component: the analytic PC-QD state 0
gradient (`[nevpt2_gradient] quasi_degenerate, variant = "pc"`) against
central differences of the same energy, every displaced point re-converged
(`conv_tol_grad` 1e-10), steps 2e-3, 1e-3 and 5e-4 bohr, one Richardson
step over the two finest and its error bar (its distance from the step over
the two coarser); RTX 3080, `--preset default`, card held.

- **The bar** is max(2 × the Richardson bar, 3 σ_E / h_min). σ_E, the
  energies' own noise, is measured: 8 repeats of the 90° reference give an
  sd of 1.3e-12 Eh at `conv_tol_grad` 2e-10 and at 1e-8 alike (atomics, not
  convergence), and 18 displaced points rerun moved by 2.6e-12 rms, so
  σ_E ≈ 1.8e-12. The Richardson value carries 0.95 σ_E / h_min of it (its
  weights at 1e-3 and 5e-4), and 3 σ_E / h_min = 1.1e-8 Eh/bohr is three of
  those. The 2× rule alone assumes the bar measures truncation; where the
  derivative is small or zero by symmetry the bar is noise, and was 1.7e-10
  on FC's atom 6 z, whose 5.3e-10 miss is that noise.
- **FC**, 42 of 42 components pass (largest gradient component 2.4e-2;
  largest miss 8.7e-9, worst in-plane miss / bar 1.9). The data is
  `test/gtest/data/nevpt2_psb3/psb3_fc.json`, held by
  `Nevpt2Psb3.PcQdGradientMatchesNumerical` (slow tier).
- **90°** (22 modes each of root 1's Sr and Si dropped, 11 and 10 of them real), 42 of
  42 pass; under the plain 2× rule 9 did not. Along the gradient direction
  and two random unit directions (`default_rng(853)`), at `conv_tol_grad`
  2e-10 and 1e-8, all six directional checks pass, the gradient direction's
  miss at most 4.7e-10 of a slope of 0.198. Rerunning the worst Cartesian
  components' points moved atom 8 x's Richardson value by 7.2e-9 (its miss
  fell from 8.0e-9 to 7.9e-10) and atom 3 z's by 2.7e-9: noise.
- **One residual, cut-independent:** atom 4 x at 90° (gradient 0.0313)
  reproduces its Richardson value to 1e-9 on a rerun, yet misses by 5.7e-9
  and 6.8e-9 (mean 6.3e-9, 2e-7 relative), and by 6.9e-9 with the cut at
  1e-15, every real mode kept: not the cut's. It passes the bar.
- **It is analytic, not the CASSCF's stopping (#862).** Those points were
  converged against a CI held to 2.5e-10 absolute, which also stalled
  `conv_tol_grad` 1e-12 at |g| 2.4e-10 (#867). Since #867, atom 4 x's
  seven points were rerun at five settings (RTX 3080, `--preset default`,
  card held). The settings were `conv_tol_grad` 1e-8, 1e-10 and 1e-11, and
  1e-10 and 1e-11 again from a perturbed start: the SCF stopped at 8
  iterations (`commutator_tolerance` 3e-3), not 20. The `[casscf]` table
  has no perturbed-start key. The points stopped at |g| 2.3e-11 to
  3.4e-11, 1.3e-11 to 1.6e-11 and 2.2e-12 to 3.5e-12. The miss held at
  8.5e-9, 9.5e-9, 9.1e-9, 8.8e-9 and 7.9e-9. A stopping bias would scale
  with |g|, and |g| fell 15×. The three steps are clean h²: the ladder is
  3.98 to 4.03, and a second Richardson level (h⁴ removed) moves each miss
  by under 1e-10.
- **It is PC-QD's alone, at the same 90° geometry** (`conv_tol_grad` 1e-10,
  atom 4 x, miss after the second Richardson level). SA-CASSCF root 0's own
  gradient misses by 8.5e-10. SC-QD's states 0, 1 and 2 miss by −1.2e-9,
  −5.2e-10 and 1.6e-9. Single-state SC and PC over a state-specific CASSCF
  miss by −3e-11 and 5.8e-10. PC-QD's states 0, 1 and 2 miss by 9.6e-9,
  −1.37e-8 and −1.4e-9. States 0 and 1 are roots 1 and 0, ρ 0.9999, so the
  miss is opposite in sign across the pair that mixes. It tracks the gate's
  active-active residue (`asymmetry.active_active` in
  `nevpt2_gradient.json`). PC-QD's is 3.9e-10, 9.6e-11 and 4.5e-11 at
  states 1, 0 and 2, and single-state PC's is 5.0e-11. Every SC run's is
  2.6e-12 to 3.7e-12. Their other blocks agree to within 3×. The PC
  Lagrangian is the one not stationary in active-active rotations, by
  about 100× more than SC's. The per-term split (`terms`) cannot localize
  it: between the stopping points the terms move by up to 3.6e-5 and
  cancel in the total.
- **Fixed (#862): the kept span's rotation.** Holding the span still drops
  the transition rows' share of the real dropped modes, which is of order
  √σ, not σ ([PC-QD](#pc-qd) below; `../gradient/README.md`, "The kept
  span's rotation", has the argument and the numbers). With it taken, the
  four gate residues above fall to 3.9e-12 to 8.1e-12 and atom 4 x's misses
  to 9.0e-10, 4.1e-10, 7.4e-10 (states 0, 1, 2) and 9.4e-10 (single-state).

Keeping the real modes instead (the cut at 1e-15) moves the analytic gradient
by up to 1.3e-8 per component and fixes none of the misses, which is what a
derivative that does not depend on the cut predicts.

### The conditioning floor

    EnergyInputs::pc_floor     S cut at max(1e-13, 100 ε √d, pc_floor) of σ_max; 0 by default
    PcClassEnergies::cut[k]    the largest σ the cut dropped, over σ_max (0: none)
    PcClassEnergies::kept[k]   the smallest σ it kept, over σ_max

**The library default stays 0 (#923)**, the round-off cut alone, and every
energy, gradient and golden is bitwise what it was before the floor
existed. A global 1e-10 was tried and measured on the gfx1200: beyond
PSB3's 1.2e-10 / 5.8e-9 Eh and HF CAS(10,7)'s 1.7e-7 (the table below),
it changes the dropped counts of the QD goldens and moves validated
results. The PC-QD H_eff elements of `h8_sto3g_sa3` and
`h8_ring12_ccpvdz_cas66_sa2` moved by up to 2.5e-8 Eh (Sir's dropped count
1 → 4, Sr's 6 → 8), and the PC-QD NACs of `h8_sto3g_sa3` moved by up to
1600 bars from #744's numerical references (8e-7 on a coupling of 9.3).
Those are real modes the round-off cut keeps, and the goldens were made at
it. So the 1e-10 floor is applied where it is needed: to the PC Hessian's
chain in the pipeline, `[hessian] pc_floor`, 1e-10 by default
(`../../pipeline/hessian/README.md`, "The PC floor"), the smallest
relative cut at which PSB3's second-order ladder is clean throughout
(the ladder table below), above BAGEL's absolute 1e-9 (about 3.5e-11 of
PSB3's σ_max). The floor is
one field of the inputs, so `pc_class_energies`, its reverse pass and its
tangent read the same value through the one `solve_class`, and each is the
derivative of the energy the others report. The gradient's entry points
(`pc_energy`, `pc_adjoint`, `pc_gradient`) take it as a trailing
`pc_floor = 0`. Its one TOML key is `[hessian] pc_floor` (#923), which a
PC `[nevpt2_gradient]` beside a `[hessian]` also takes.
The QD paths (`pc_effective_hamiltonian` and its reverse pass) read it from
each state's own inputs, which the PC-QD entry points (`qd_pc_energy`,
`qd_pc_adjoint`, `qd_pc_gradient`, `qd_pc_coupling`) set from their own
trailing `pc_floor = 0` (#1020); nothing in the pipeline sets it there. `cut` and `kept`
are reported per class, and per ket in an effective Hamiltonian; both are 0
for Sijrs.

Why it exists (#917, measured on the RTX 3080, `--preset default`, card
held): at PSB3 the solve's conditioning is harmless to first order but not
to second. The points are state-specific DF-CASSCF(6,6)/cc-pVDZ
(cc-pVDZ-JKFIT) at the FC geometry and twisted to 90°. Along each of the 42
physical directions t_x of `hessian::df_nevpt2_tangents` (#813), PC's inputs
move as the nuclear Hessian moves them.

- **First order is harmless.** Over five reverse passes, Σ ⟨bar, t_x⟩
  repeats to 4.9e-15 Eh/bohr at FC and to 1.5e-12 at the twist, where the
  largest component is 3.4e-2 and 2.5e-2: under 1e-10 of it, and under
  2e-13 of the largest summand. Along random directions the same bars
  repeat only to 5e-7 to 1e-5 relative, from Sr's and Si's kept modes at about
  5e-13 of σ_max. The physical directions do not reach those modes; random
  ones do. Raising the cut leaves t_x's spread as it is (the twist's
  worst σ / |Σ| moves from 3.9e-9 to 2.3e-9 at a cut of 1e-11), so what
  remains is not the cut's.
- **The atomics (#916).** The warp pre-reduction in `recipes::einsum` cut
  the random directions' spread about 4×. In the SA3/STO-3G twist fixture
  of `SpanTermVisibleAtPsb3Twist` it fell from 1.4e-5 to 2.4e-5 relative
  before to 2e-6 to 7e-6. The sums are still not deterministic.
  Deterministic reductions would not help what follows: the ladder below
  fails with each rung reproducible to 1e-10 to 1e-5 of its largest
  element, while its rungs disagree by factors of 1e2 to 1e6.
- **Second order is not.** The ladder #815 would gate on: central
  differences of `pc_adjoint`'s first derivatives along t_A, at
  (h, B, c) + t t_B (c renormalized, C the identity in the MO basis), at
  t = 4e-4 / 2^k over five rungs. SC's `sc_adjoint` on the same path is
  clean (ratios 3.89 to 4.00, H_BB about 5e2 to 1e3). PC's, along t_0x
  and t_1y (the ratios of the two finest rung pairs; H_BB's limit):

| relative floor | FC t_0x | FC t_1y | 90° t_0x | 90° t_1y |
|---|---|---|---|---|
| 0 (the default cut, about 3e-13) | 0.55, 7.65: none | 141.8, 4.81: none | 0.13, 1223: none | 12.3, 4.16: none |
| 1e-11 | 7.47, 4.15 | 4.20, 4.02 (−1036) | 5.10, 4.04 | 4.93, 4.03 |
| 1e-10 | 4.16, 4.01 (−569) | 4.02, 4.00 (−1036) | 4.09, 4.01 | 4.13, 4.01 |
| 1e-9 | 4.03, 4.00 (−569) | 4.01, 4.00 (−1036) | 4.02, 4.00 | 4.04, 4.00 |

  At the default floor and at 1e-11, the path at t = 4e-3 meets intruders
  (|t| from 1.1 to 6.9, in Sr, Si, Sij, Srs and Sir). At 1e-10 the twist's
  coarsest rung (4e-4) still breaks. Out of the FC plane (t_4z,
  symmetry-breaking), H_BB settles neither at the default nor at 1e-11. The near-cut kept modes are what moves: K stops
  annihilating them off the eigenstate, and their λ swings as 1/σ through
  −Δ.
- **What the floor costs** (E_corr, Eh; the state-specific references
  above):

| relative floor | FC E_corr | its shift | 90° E_corr | its shift |
|---|---|---|---|---|
| 0 | −0.7622369939151 | | −0.7723305932308 | |
| 1e-12 | −0.7622369939122 | 2.9e-12 | −0.7723305928627 | 3.7e-10 |
| 1e-11 | −0.7622369938649 | 5.0e-11 | −0.7723305916378 | 1.6e-9 |
| 1e-10 | −0.7622369937948 | 1.2e-10 | −0.7723305874084 | 5.8e-9 |
| 1e-9 | −0.7622369923425 | 1.6e-9 | −0.7723305725053 | 2.1e-8 |

  Elsewhere ([the cut table above](#pc-nevpt2)): HF CAS(10,7) moves 1.7e-7
  at 1e-10 and 1.2e-6 at 1e-9; PSB3 FC's PC-QD state 0, which
  `Nevpt2Psb3.PcQdGradientMatchesNumerical` holds to 1e-9, does not move at
  1e-10 and moves 1.3e-9 at 1e-9. BAGEL's absolute 1e-9 is about 3.5e-11 of
  PSB3's σ_max, between the 1e-11 and 1e-10 rows.

**What a PC Hessian should refuse on.** `kept`, not `kept / cut`. Where real
modes run through the cut (PSB3), the ratio is near 1 at any floor: 1.2 to
3.5 at the default (`kept` 3.4e-13 to 7.6e-13), and 1.1 at 1e-10
(`kept` 1.08e-10), where the ladder is clean. A gapped golden's ratio is
1e11 or more at the default (`n2_ccpvdz_cas44_df`: `kept` 1.6e-5 over a cut
at 8e-17). The ladder's behaviour follows `kept` alone: it fails at 5e-13,
is partial at 1e-11, and converges at 1e-10 for steps to 2e-4 and at 1e-9
for steps to 4e-4. So #815 and #817 should refuse when a class's `kept` is
below about 1e-10, a factor of 300 to 1000 above the default cut. Their
ladders should gate either on fixtures whose kept spectra are gapped, or on
PSB3 with the floor raised to 1e-10 or more.

**Memory.** The solve's d × d matrices (S, K, their symmetrized copies, V,
and the d × m X, U, T and gemmwich's work) are buffers of their own, freed
when the class returns; at n = 12, d = n³ is 24 MB per matrix, below the n⁶
intermediates. The slab's y is a buffer of the walk's, sized to its widest
slab before the walk.

## PC-QD-NEVPT2

    pc_effective_hamiltonian(res, blocks, pairs, batch) -> vector<PcClassEnergies>
        pairs        sc_effective_hamiltonian's: n x n, [I n + J] bra I, ket J
        [I n + J]    energies[k] ⟨I|V|Ψ_J⁽¹⁾⟩ of class k, dropped[k] ket J's
                     cut count, total Σ_k; the diagonal is each state's
                     pc_class_energies

`qd.cpp` checks the inputs as for SC; the walk is `pc_class_energies`' own
(`pc_couplings`, `pc.cpp`), which runs it at n = 1. Per class and ket J,
S_J, K_J and S₁_J are J's own and are cut and solved once (`solve_class`,
its cut and refusals unchanged), n solves per class; a refusal names
the class **and the state**
(`nevpt2::pc_effective_hamiltonian: state 1: Sr's metric has no positive eigenvalue`).
Then, the conventions being `test/gtest/data/qd_nevpt2/README.md`'s,

    ⟨I|V|Ψ_J⁽¹⁾⟩_t = −c b_{I←J,t}ᵀ (K_J + Δ_t S_J)⁺ b_{J,t}
                   = −c Σ_k y_{I←J,tk} y_{J,tk} / (λ_{J,k} + Δ_t)
    b_{I←J,t} = S^{(J,I)} x_t + S₁^{(J,I)} h1_t

- **S^{(J,I)} and S₁^{(J,I)}** are the class's S and S₁ formulas (the table
  above) over the transition RDMs of **pair (J, I)**, bra J and ket I,
  unsymmetrized. For S either orientation would do (S^{(J,I)} = S^{(I,J)}ᵀ),
  but S₁ must be (J, I)'s. Their terms that read no RDM carry ⟨J|I⟩ = 0 and
  are dropped: hdm1's 2δ (Sijr) and hdm2's δδ (Sij), `make_hdm1` and
  `make_hdm2`'s `overlap`, as in SC. A transition builds no K.
- **One einsum per pair.** y_{I←J,t} = T_{I←J}ᵀ x_t + t1_{I←J}ᵀ h1_t, with
  T_{I←J} = S^{(J,I)ᵀ} U_J (one `gemm`, the row-major S^{(J,I)} read
  column-major) and t1_{I←J} = S₁^{(J,I)ᵀ} U_J, built once per pair before
  the walk (`pc_kets`); at I = J they are the solve's own T and t1, so the
  diagonal is `pc_class_energies`' arithmetic, einsum for einsum. Each
  class's S, K and S₁ come from one builder (`pc_sr_matrices`, …), over a
  state's own RDMs or a transition's, which the reverse pass reads too.
- **Sijr and Srsi**: "2 direct − exchange" becomes the bilinear
  Σ_t (2 y_{I←J,t} y_{J,t} − y_{I←J,t} y_{J,t′}) / (λ + Δ_t), t′ the
  exchange partner, whose Δ is the same. Srs and Sij keep ½, the rest 1.
- **No (I, J) / (J, I) mean**, unlike SC's norms, so the matrix is not
  symmetric. **Sijrs is SC's**: the diagonal `energy_sijrs`, zero off it.

Each class walks its slabs **once** for every pair: per slab, n² einsums into
one buffer, pair (I, J)'s y block (tuples × m_J) at
`PcKets::offsets`, read back and finished on the host, each ket's smallest
denominator checked as `pc_class_energies` does. The memory over one class
is n solves (λ, T, U, and t1) and n² − n transition T and t1, d × m_J each,
held across the walk; the slab's y grows as n².

Entries may be omitted as SC's. With the diagonal omitted no state is
solved: `pc_kets` takes each U_J from the `PcAmplitudes` a call with the
diagonals filled, builds only the given transitions' T and t1, and the
finish reads that call's g: (I, J) is −c Σ_k y_{I←J,tk} g_{J,tk}, Sijr's and
Srsi's y_{I←J} (2 g_t − g_{t′}), the partner t′ anywhere in the cached
(r, s, i) table. Its dropped counts are d − m_J. Nothing is refused that
the diagonal call did not refuse.

## The amplitude hand-over

    sc_effective_hamiltonian(res, blocks, pairs, batch, &sc_amplitudes)
    pc_effective_hamiltonian(res, blocks, pairs, batch, &pc_amplitudes)

Both take an optional last argument that receives each state's first-order
wave function in the seven classes with an active index. Sr's, Si's and
Sir's are what the QD-NEVPT2 PT2 transition density
([`../nac/`](../nac/README.md)) contracts; all seven are the kets a later
call with its diagonal omitted reads instead of walking them again, when
the same pointer is passed back ([SC-QD-NEVPT2](#sc-qd-nevpt2)). Nothing is
re-solved: the walks already form these and only keep them. The elements
are the same with and without it
(`Nevpt2Pt2Density.AmplitudeOutputLeavesHeffUnchanged`). On the host they
are one double per perturber per state for SC, and tuples × m_J per class
and state for PC: the largest Srsi's v² c n_act and Srs's v² m.

- **`ScAmplitudes`**: per state J and perturber t, c_t = −1 / (Δ_t +
  H^J_t / N^J_t), J's own denominator from its N and H tables, and 0 where
  `_norm_to_energy`'s guard drops the perturber; −1 / (Δ_t + η^J) for every
  perturber of a one-function class. Ψ_J⁽¹⁾ holds c_t P_t V|J⟩.
  Sr over r, Si over i, Sijr over (r, j ≤ i) and Srsi over (r ≤ s, i) as
  their tables pack them, Srs over (r, s), Sij over (i, j), Sir over (r, i)
  at r ncore + i (its tables run (i, r), and are reordered). A class with
  no perturber holds n empty lists.
- **`PcAmplitudes`**: per class and state J, the solve's U_J (moved out once
  the class is finished and checked) and g_tk = y_{J,tk} / (λ_k + Δ_t) from
  J's own rows of each slab, tuples × m_J on the host. Ψ_J⁽¹⁾ holds
  −Σ_k g_tk Σ_μ U_J[μ,k] φ^J_μ. A refusal leaves it unspecified.

## The SC reverse pass

    class_energies_adjoint(res, blocks, in, batch, bars) -> ClassEnergiesAdjoint
        energies     class_energies' norms, energies and total
        eq40[8]      Park Eq. 40, Σ_t (P_t N_t + Q_t H_t), from the bars
    bars (EnergyBars, each added onto, a null one not formed):
        h1e, h2e, dm1, dm2, dm3, f3ca, f3ac             in their inputs' layouts
        h1e_v_sr, h1e_v_si, h1e_v_sir, e_core, e_virt   in their inputs' layouts
        lambda                                          Λ, nevpt2::slab_adjoint's

`reverse.cpp`, `docs/nevpt2/nevpt2.tex`, "The reverse passes" (demo `docs/gradients.md`
§5.1–5.3): from the energy back to every `EnergyInputs` field and, through
Λ, to B. It runs in two parts. The walk takes the energy back to the
class-level inputs and the block bars M_G at fixed h1e, h2e and RDMs; the
[sweep](#the-sweep) then takes the M_G back through the builders onto h1e,
h2e and the RDMs. Each class walks its slabs as `class_energies` does, and lists its
norm and H terms **once**, as `ClassTerm`s with energy.cpp's subscripts,
coefficients and operands, plus the bar each operand's adjoint goes into.
Per slab the list is evaluated, the norms and H read back, and the finish's
multipliers taken on the host (Park Eqs. 36–38):

    T = −1 / (Δ + H/N),  P = ∂E/∂N = 2T + ΔT²,  Q = ∂E/∂H = T²,  R = ∂E/∂Δ = N T²

zero for a perturber whose norm `_norm_to_energy` drops. P and Q go back
up as the out-bars of the norm and H terms, and every term is differentiated
by `einsum_adjoint`. Then `slab_adjoint` takes the slab's bar onto Λ, and the
bar is dead before the next slab is built: no class holds more than its own
slab's. R goes onto the orbital energies each perturber's Δ reads, summed on
the host and added once at the end.

**`einsum_adjoint`** (`:shared`) is the one derivative rule. It takes the
forward call's subscripts, operands, out_dims and α **unchanged**, plus one
bar per operand (null: none) in that operand's own layout, and writes operand
k's bar as the same einsum with the output and operand k swapping roles. So
no derivative is written by hand.

- A row- or column-major operand's bar is written in place, the output labels
  ordered so that einsum's packed row-major output is that layout.
- An operand that repeats a label (Sir's `raai`; the builders' `jbij`,
  `mjjn`, …) takes its bar over its distinct labels into workspace, then
  scatters it onto the diagonal against `delta`.
- A two-axis view of a larger column-major array (h1e_v_sr's and
  h1e_v_sir's rows, strides {1, ld}) takes its bar packed, then one in-place
  geam adds it at the view.

**The hand-written pieces** are the finish above and the cases the demo
names. Sijrs reads no block, so its slab bar and its orbital energies' are a
kernel of their own (`device::sijrs_adjoint`): ḡ = 2θ/D, and gθ/D² taken off
ε_i and ε_j and added to ε_a and ε_b. Sijr and Srsi symmetrize pairs of table
entries, so a `Perturber` names both and its multipliers go back onto both,
halved on the diagonal. Srsi's pairs span slabs, so it walks twice: the first
walk fills its whole (v, v, c) tables, the finish runs once, and the second
walk rebuilds each slab pair and takes its adjoint against the multiplier
tables' rows.

**Kept or recomputed.** Recomputed, all of it. The reverse pass is a walk of
its own, so `class_energies` and its builders are untouched and the forward
energy is unchanged. Each class rebuilds its blocks (a16, …) and every slab,
and re-evaluates its norms and H. Srsi rebuilds every slab twice.

**The block bars.** Every class also gets M_G = ∂E/∂G on each block G it
contracts. These are `BlockBars` in `reverse.cpp`, row-major like G, and
`block_bars` allocates them. A metric a class reads straight from an RDM
(Sr's dm1–dm3, Srsi's dm1, Sir's dm1 and dm2) is a block of that class there,
so each class's Eq. 40 is assembled from its own bars:
`eq40[k] = Σ_G ⟨M_G, G⟩`, plus ⟨P, term⟩ for Sir's one term that reads no
block, and ½ ⟨ḡ, g⟩ for Sijrs.

### The sweep

`sweep_blocks` takes every M_G back through the `make_*` that built G, onto
the `h1e`, `h2e` and RDM fields of `EnergyBars` (skipped when all seven are
null). It is two halves, in this order: `sweep_dyall`, the Dyall blocks
(a16, a17, a19, a22, a23, a25, a3, k27, a7, a9, a12, a13), which read h1e and
h2e and only a state's own walk has; and `sweep_metrics`, the metrics (Sr's
dm1–dm3, dm3_h, dm2_h, dm1_h, hdm1, Srsi's dm1, rm2, hdm2, Sir's dm1 and dm2),
which read only RDMs and are all a QD pair's transition RDMs take. Each builder is listed term by term with energy.cpp's subscripts,
coefficients and operands, so the two read side by side, and each term goes
through `einsum_adjoint` unchanged: with h1e and h2e among the bars, every
operand of a term but delta takes one. A transpose is its one-operand einsum,
NumPy's axes spelled as labels (`transpose(f3ca, {1, 4, 0, 2, 5, 3})` is
`"abcdef->beacfd"`). A block that is an RDM itself (Sr's dm1–dm3, Srsi's
dm1, Sir's dm1 and dm2) is the identity einsum.

**Reverse topological order.** An intermediate's bar is complete before it
is swept onto what built it: a9, hdm3, hdm2, then hdm2's dm2b (Sij); a7,
rm3, rm2 (Srs); a3, hdm1 (Sijr). Where a class reads the intermediate as its
metric too (hdm1, rm2, hdm2), the walk's M_G is where the builders above it
add (so `sweep_dyall` runs first), and `BlockBars` is spent by the sweep. Si's dm1_h is `make_hdm1`, so its
bar joins Sijr's hdm1's and is swept once. a17's and a19's h1e − Σ_j (mj|jn)
gets one bar from both and is swept onto h1e and h2e last.

**The diagonal slices need no kernel.** `diag_a16` adds fdm2 onto a diagonal
of a16, `a16[p, i, r, a, b, i] += fdm2[p, r, a, b]`, which no einsum can
write, since an output label cannot repeat. Its adjoint is the gather, and an
input label can: `fdm2_bar = einsum("pirabi->prab", a16_bar)`. `diag_a22a`'s
is `−"xikibc->xkbc"` and `diag_a22b`'s `2 "xikaic->xkac"`, each with its
kernel's sign and factor. fdm2's own term then takes the gathered bar.

**Recomputed, not kept.** The intermediates a bar is taken against (hdm1,
hdm2, hdm3, rm2, rm3, and a17/a19's modified h1e) are rebuilt by the
forward's own builders (rm2 through `make_a7`, which is what builds it), so
nothing is held from the walk. dm2b and the three fdm2s are not rebuilt at
all: `einsum_adjoint` reads every operand of a term but the one whose bar it
forms, and each is the only operand of the term that reads it, or the one
with the bar.

**Memory.** On top of `class_energies`' own, the block bars are buffers of
their own. They outlive their class, four of them n⁶ (Sr's a16 and dm3, Si's
a22 and dm3_h), and live through the sweep. Each slab's bar is one more
slab-sized buffer, and the norm, H and multiplier tables per slab are buffers
too, since they are held across `einsum_adjoint`'s and `slab_adjoint`'s
workspace. The sweep adds at most two n⁶ at once (hdm3 and its bar, or rm3
and its bar) and a few n⁴; every intermediate bar is a buffer of its own for
the same reason.

## The SC forward tangent

    class_energies_tangent(res, blocks, in, batch, directions) -> ClassEnergiesTangent
        energies                     class_energies' norms, energies and total
        norm, h, delta [k][t]        each perturber's N, H and Δ
        norm_dot, h_dot, delta_dot   [k][t m + u]: their tangents along direction u
        energy_dot [u][k], total_dot [u]
    directions (EnergyDirection, a null field zero):
        in       a tangent of every EnergyInputs field, in its layout
        blocks   Ḃ over the MO blocks, laid out as mo_blocks lays B out

`reverse.cpp`, forward mode over the SC reverse pass's own walks, for the
nuclear Hessian (milestone 24, #805): per perturber (Ṅ, Ḣ, Δ̇), and Ė per
class, along each of a block of m directions. A class's perturbers are
`ScAmplitudes`' order, Sir's at r ncore + i, whatever the batch.

**The finish, in (N, d).** With d = NΔ + H, a perturber's energy is
E = −N²/d, and

    Ė = (−2N/d + N²Δ/d²) Ṅ + (N²/d²) Ḣ + (N³/d²) Δ̇,

Park's P, Q and R (`docs/nevpt2/nevpt2.tex` Eq. pqr) with no
1/N: the (H, T) form's removable 1/N factors are fatal near the guard. A
one-function class's Ė reads η instead
([One-function classes](#one-function-classes-949)). A
perturber whose norm the energy guards (|N| ≤ 1e-14) adds nothing to Ė;
its N, H, Δ and their tangents are still reported as the walk forms them,
for #811's second-order term there. Sijrs has no guard and no tables: its
Ė is one kernel's (`device::sijrs_tangent`, Σ (ġθ + gθ̇)/D − gθḊ/D²), and
its N, H and Δ tables are empty.

**Ṅ and Ḣ.** An own walk carries a tangent (`ScTangent`) beside its bars,
and each slab's norm and H terms, the same `ClassTerm` list the energy and
the reverse pass evaluate, are evaluated once more per operand with that
operand at its tangent (`tangent_terms`): a slab read twice is taken twice.
Each operand finds its tangent by its address in a region list: the inputs
the terms read directly (h1e_v's rows at their offsets, Sr's, Srsi's and
Sir's RDMs) for the whole call, the class's blocks and slabs for its walk.
So no class term is written twice.

**A block's tangent is its builder's polarization.** Every block (a16, …,
dm3_h, hdm1, rm2, hdm2) is one integral times one RDM, plus terms linear in
one or the other, so of degree at most two in (h1e, h2e, RDMs), and

    Ġ = (G(x + s ẋ) − G(x − s ẋ)) / 2s

is exact: the forward's own `make_*` at two shifted inputs, nothing
differentiated by hand. **A slab's** is bilinear in B, so
(slab(B + s Ḃ) − slab(B) − s² slab(Ḃ)) / s is exact, with no new
`density_fitting` entry point. Each s is a power of two that sizes s ẋ to
x: the RDMs' ‖r‖/‖ṙ‖ (the integrals' when the RDMs do not move), and
‖B‖/‖Ḃ‖. At s = ‖r‖/‖ṙ‖ the difference's round-off,
ε (‖h‖ + s‖ḣ‖)(‖r‖ + s‖ṙ‖)/s, is 2ε (‖h‖‖ṙ‖ + ‖ḣ‖‖r‖), twice the
tangent's own scale, so the difference cancels nothing past round-off.

**Batched.** Each class walks once for all m directions: its blocks once,
their tangents per direction, held through the walk; per slab, the slab once,
then per direction its tangent (one B + s Ḃ buffer of the MO blocks,
refilled per direction) and the tangent tables, read back. Memory grows with
m by the class's block tangents, at most two n⁶ per direction (Si's a22 and
dm3_h), and while one is built, one shifted copy of the inputs (three n⁶)
and the block twice; a caller with many directions chunks them.

## The SC Hessian-vector product

    class_energies_hvp(res, blocks, in, batch, directions, bars, partners = {})
        -> ClassEnergiesHvp
        energies         class_energies' norms, energies and total
        populated [u][k] class k's empty perturbers direction u populates (K6's)
        population [u]   the largest N^{uu} over those (0: none)
        scale [u]        u's population scale, kEmptyPopulationFloor's base
        defect [u]       the empties' weighted ρ_τ on (u, its partner), over S
        rounding, rounding_at, rounding_norm
                         the kept perturbers' worst σ / |N| (σ = ε/2 in Sr, Si, Sir;
                         4ε |N| in the others), its {0, class, perturber} and N
    bars[u] (EnergyBars, added onto): the tangent of class_energies_adjoint's
        bars along directions[u], every field they name, Λ included
    partners: none, or one EnergyDirection per direction, the pair checked

`reverse.cpp`, forward over the reverse pass, for the nuclear Hessian
(milestone 24, #811; `docs/nevpt2/nevpt2.tex` Eq. finish, "SC's class tangent"). The bars are
Σ_t P_t ∂N_t + Q_t ∂H_t + R_t ∂Δ_t, (P, Q, R) the finish's derivatives, so
their tangent along ẋ is two terms, and Sijrs's.

**The multipliers' tangent: the finish's 3 x 3.** In the (N, d) form,
d = NΔ + H and E = −N²/d (no removable 1/N),

    E_NN = −2/d + 4NΔ/d² − 2N²Δ²/d³    E_NH = 2N/d² − 2N²Δ/d³    E_NΔ = 3N²/d² − 2N³Δ/d³
    E_HH = −2N²/d³                     E_HΔ = −2N³/d³            E_ΔΔ = −2N⁴/d³

times `class_energies_tangent`'s (Ṅ, Ḣ, Δ̇) gives (Ṗ, Q̇, Ṙ) per perturber,
and one reverse pass at those multipliers (`fixed_adjoint`: the walks, the
sweep and R's host sums of `class_energies_adjoint`'s own walk, the finish
replaced by a table) takes them onto every bar. Zero on a guarded perturber,
whose N, Ṅ and Ḣ vanish.

**The bars at fixed multipliers, differenced.** Every N and H table is a
polynomial in (B, h1e, h2e, h1e_v, RDMs) of degree at most six (two slabs,
each quadratic in B, around a block of degree at most two), so the bars at
fixed (P, Q) are of degree at most five along a line, and

    Σ_k w_k (A(x + k s ẋ) − A(x − k s ẋ)) / 2ks,   k = 1, 2, 3,   w = (3/2, −3/5, 1/10)

is their derivative exactly: the weights cancel the s² and s⁴ terms. That is
the "existing contractions at dotted arguments" of every einsum, the sweep's
builders and the slabs' pull-back onto Λ, taken together without one
derivative written by hand. s is the power of two nearest an eighth of the
smallest ‖x‖/‖ẋ‖ over the groups the direction moves (B; h1e, h2e and
h1e_v; the RDMs), so the widest rung moves no group past its own size. The
orbital energies are read by R alone, a constant ∂Δ/∂ε, so they need no
difference. Six reverse passes at shifted inputs, each a copy of every
moved field and of B.

**Sijrs** has no tables: `device::sijrs_adjoint_tangent` differentiates
`sijrs_adjoint`'s bars along (ġ, ε̇), ġ the slab's bilinear tangent, and the
pull-back onto Λ takes that through B and the bar itself through Ḃ (Λ is
linear in B at a fixed slab bar).

**Empty perturbers: K6, and the ρ_τ refusal.** A perturber the guard skips
(|N| ≤ 1e-14; never one of a one-function class's, which has no guard),
empty by symmetry, has N, H and their gradients zero; a
direction that breaks the symmetry populates it at second order, where its
energy is homogeneous of degree two (demo `hessian-theory.md` §5.6). Its
second-order tables, N^{uv} and H^{uv}, are the same kind of difference of
`class_energies_tangent`'s (Ṅ, Ḣ) along u at shifted inputs, six tangent
calls per line, exact. A guarded perturber is populated by u when its N^{uu}
is above 1e-10 of u's scale, the larger of its largest |N^{uu}| and the
largest |N| over (8s)². Then:

- **Along u alone** the energy is a quadratic form, and the perturber takes
  K6, the finish's multipliers at their limit along the ray: P̄ = 2T̄ + ΔT̄²,
  Q̄ = T̄², T̄ = −1/(Δ + η), η = H^{uu}/N^{uu}, in the fixed-multiplier
  difference above. Against v, P̄ N^{uv} + Q̄ H^{uv} = −N^{uv}/(Δ + η) when
  H ∝ N on the pair: the demo's K6.
- **With a partner v**, a perturber both populate takes K6 at the mean η,
  and is scored by the scale-free defect, at unit populations
  c = N^{uv}/√(N^{uu}N^{vv}), κ = H^{uv}/√(N^{uu}N^{vv}),
  η_w = H^{ww}/N^{ww} and η their mean,

      ρ_τ = max(|η_u − η_v|, 2 |κ − c η|) / |Δ + η|,

  zero exactly when H is proportional to N on the pair. A perturber's ρ_τ
  is weighed by its share of the pair's Hessian, w = √(N^{uu}N^{vv}) /
  |Δ + η|, against S, the largest such share over **every** perturber
  (a regular one's η its own H/N, read from the same second tables):

      defect = max_t ρ_τ,t w_t / S,

  and past 1e-6 the call is refused, Numerical, naming the class and the
  perturber that scored it: the energy is not a quadratic form on that pair
  to that bound.

**A direction that moves nothing.** The floor is a fraction of the
direction's own scale, so a direction whose tangent is round-off (a rigid
translation's, in a caller's coordinates) populates everything against its
own round-off. `population` and `scale` are returned so that a caller with
several directions of one size (`hessian::df_sc_nevpt2`'s orthonormal
coordinates) reads each population against the largest scale instead
(#952; `kEmptyPopulationFloor` is exported for that).

**What `defect` means** (#914). Until #914 it was the largest raw ρ_τ, and
its refusal said "no Hessian on the pair". It is now the empties'
ambiguity relative to the pair's Hessian scale: below 1e-6 the pair is
accepted with K6 at the mean η, and the element it gives is the Hessian's
to that relative bound, an error bound and not a missing term. A pair that
is not proportional but whose empty perturbers carry a negligible share
(the twist's tiny real perturbers, below) is accepted. 0 with no partner,
or no empty perturber both populate. `sc_hessian_product` hands it on as
it stands.

**The bars' ladder sees K6.** Off the base point an empty perturber's norm,
t² N^{uu}/2, clears the guard, so the bars there are the ray's limit, which
is K6 at η_u: a ladder of the bars along u converges to the product with K6
and misses without it (Λ by 5e-3 along B on CAS(8,8)). The mixed second
difference of the energy along a pair is what checks η across two
directions.

Cost per direction: seven reverse passes over the whole inputs and, when any
perturber is guarded, six `class_energies_tangent` calls (twelve with a
partner).

### The thresholds, measured (#914)

Measured on the RTX 3080 (`--preset default`, card held), over every
`class_energies_hvp` input, along #813's physical t_x (through
`builders_tangent`) and random directions. A proportional partner is
v = u/2 + w with w keeping the symmetry (PSB3's in-plane t_0x; on N₂ B and
h1e_v scaled per orbital pair). The non-proportional partners are a second
breaking direction. The cases: N₂ cc-pVDZ CAS(8,8) and CAS(4,4) (the
goldens); PSB3 DF-CASSCF(6,6)/cc-pVDZ, state-specific, planar FC (Cs) and
twisted 90° (C1); #803's three STO-3G fixtures. h2co_skew and h2co_neardeg
have no guarded perturber. nh3_cs does: this tree's CASSCF converges to
another state than the fixture's (−55.432707 against −55.437459). There
Sr and Si are empty by stationarity, and no physical t_x populates them.

**The guard, |N| ≤ 1e-14.** Base norms: guarded ≤ 1.2e-32 (CAS(8,8)),
0 (CAS(4,4)), 8.4e-22 (PSB3 FC, 76,240 of 125,021) against real ones
≥ 5.3e-8, 6.0e-8 and 4.3e-14. At the twist there is no gap: 5 real
perturbers sit under the guard (|N| up to 9.8e-15), the smallest above it
at 1.25e-14.

**The population floor, N^{uu} > 1e-10 of u's scale.** Round-off along
directions that keep the symmetry is ≤ 7e-33 (N₂), 0 (CAS(4,4)) and
1.8e-18 (PSB3 FC, every in-plane t_x). Real populations are ≥ 8.5e-6 and
2.3e-6 on N₂, but at PSB3 FC they run continuously from 1 down to 1e-15,
about 900 of them below the floor on each t_iz. At the twist the five tiny
perturbers' N^{uu} sits at 4e-12 to 1.3e-10, across the floor, and flips
between runs. The floor is kept: what it drops is at most 1e-10 of S,
and under the weighted defect a perturber on either side of it changes
the defect by as little.

**ρ_τ has no gap; ρ_τ w / S has six decades.**

| pairs | raw ρ_τ | defect (ρ_τ w / S) |
|---|---|---|
| N₂ CAS(8,8), proportional | ≤ 3.5e-15 | ≤ 1.7e-18 |
| N₂ CAS(4,4), proportional | ≤ 8.6e-15 | ≤ 4.5e-16 |
| PSB3 FC, 14 t_iz, proportional | up to 2.0e-5 (12 of 14 refused) | 5.3e-13 to 3.9e-11 |
| PSB3 FC, random, proportional | ≤ 3.6e-8 | ≤ 1.5e-11 |
| PSB3 twist, every t_x | refused up to 1.9e-4, run to run | ≤ 6.5e-15 |
| N₂ CAS(8,8), random all vs B | 5.8e-5 to 6.8e-2 | 4.7e-5, 7.3e-5 (two runs) |
| N₂ CAS(4,4), random all vs B | 4.0e-5 to 0.22 | 0.10 |
| PSB3 FC, t_iz vs t_(i+1)z, all 14 | 5e-9 to 0.29 | 1.9e-4 to 1.8e-2 |
| PSB3 FC, random vs random or t_0z | 5.5e-6 to 0.58 | 5.1e-2, 0.17 |

ρ_τ's round-off grows as one over the population: max ρ_τ · N^{uu}/scale
is ≤ 8e-12 on the PSB3 proportional pairs, and over the perturbers
populated past 1e-4 of the scale ρ_τ is ≤ 3e-8. Real defects fall to
5e-9 on weakly populated perturbers, so no cut on ρ_τ alone separates the
two at PSB3. Weighted, proportional pairs reach 3.9e-11 and
non-proportional ones start at 4.7e-5. The 1e-6 sits 2.5e4 above the
first and 47 below the second, on N₂'s CAS(8,8). At the twist a random
non-proportional pair reads 1.3e-7 and is accepted: its five empty
perturbers are tiny real ones whose share of the pair is negligible.

**A small real perturber against an empty one.** CAS(8,8) moved by ε along
a breaking direction u puts N ≈ ε² N^{uu}/2 on its 144 empty perturbers:
at ε = 1e-7 all are under the guard (|N| ≤ 1.1e-15) and take K6. At 1e-6,
32 still are, and at 1e-5 one is. Through the crossing, ⟨v, HVP v⟩ for a
second breaking direction v moves linearly in ε, slope −1.1636e3 under the
guard and −1.1635e3 past it (no step above 8e-11 of the element). It
agrees with a ladder of the bars at t ≈ 1e-5 to 2e-7 on either side. Past
the guard the product is symmetric, ⟨v, HVP u⟩ = ⟨u, HVP v⟩ to 1.8e-12.
Under it, with no partner given, K6 on this non-proportional pair leaves
3.2e-6 between the two. So a small real perturber under the guard gets the
ray's limit, which any ladder whose steps are far larger than its ε also
sees, and the step to the finish past the guard is below the ladder's bar.

## The SC-QD reverse pass

    sc_effective_hamiltonian_adjoint(res, blocks, pairs, hbar, batch, bars, transition)
        -> ScEffectiveHamiltonianAdjoint
        elements     sc_effective_hamiltonian's n² ClassEnergies
        eq40[8]      Σ_IJ H̄_IJ H^k_IJ per class, from the bars
    hbar        H̄, n x n row-major: L = Σ_IJ H̄_IJ H_IJ is what is differentiated
    bars        n EnergyBars: bars[J]'s dm1..f3ac state J's own RDMs' bars; the
                shared fields' (h1e, h2e, h1e_v_*, e_*, Λ) bars[0]'s, the same
                pointer in every entry (checked), summed over the pairs
    transition  called once per ordered pair (I, J), I != J, with an
                EnergyBars whose dm1, dm2, dm3 alone are set; null: not formed

`reverse.cpp`, after SC's, whose walk it is: **`class_energies_adjoint` is
its n = 1, H̄ = 1**, einsum for einsum (`sc_couplings_adjoint`; `qd.cpp`
checks the inputs). The coupling −T_IJ / D_J, D_J = Δ + H^J / N^J, has
three kinds of bar:

    on T_IJ:          −H̄_IJ / D_J, half onto each of the norm einsums of (I, J) and (J, I)
    on D_J:           A_J = Σ_I H̄_IJ T_IJ / D_J²   (I = J included, T_JJ = N^J)
    through D_J:      −A_J H^J / (N^J)² on N^J,  A_J / N^J on H^J,  A_J on Δ

At I = J these are Park Eqs. 36–38 times H̄_JJ, and the own walk's finish is
written so: `H̄_JJ (2T + ΔT²) + A′ (−H/N²)`, A′ the pairs' share, so at
n = 1 every multiplier is SC's to the bit. A vanishing ket norm gives its
column no energy and no bars, as in the forward. Sijrs reads no RDM, so its
bars are `sijrs_adjoint`'s with weight Σ_J H̄_JJ.

**Pair at a time.** The walk runs in three parts, every class's slabs
rebuilt in each:

1. Each state's own walk, evaluated only: its N^J and H^J per perturber, in
   walk order, kept on the host (`KetTables`). Skipped at n = 1, which has no
   pair.
2. Each unordered pair M < N: per class, the slabs walked once, the norm
   einsums of (M, N) and (N, M) evaluated (the 2 × 2 block's couplings and
   T), and differentiated at once (onto the slabs and so Λ, h1e_v's rows and
   the pair's metric bars): the multiplier on T needs only the kets' D,
   which part 1 has. A_M and A_N accumulate on the host. Then the pair's
   metric bars are swept onto its transition dm1–dm3 (`sweep_metrics`),
   handed to `transition`, (M, N) then (N, M), and freed.
3. Each state's own walk, as `class_energies_adjoint`'s with A_J added, and
   its sweep onto bars[J] and the shared h1e and h2e.

The hand-off is a callback so that the next consumer (the SC-QD adjoint,
#702) can take each pair's bars through the transition `rdms_adjoint` at
once. The bars are device buffers queued on `res.stream()`, freed when the
walk moves on: the callee works on that stream or synchronizes.

**Eq. 40.** Each coupling is homogeneous of degree one in (T_IJ, N^J, H^J),
so Σ_tables ⟨P, table⟩ + ⟨Q, H⟩ over every walk is Σ_IJ H̄_IJ H_IJ per class:
the pairs' T share gives the off-diagonal elements, and A's share through
N and H cancels. `eq40` is assembled from the block bars as SC's is, and
Euler's degree-4 identity in (B, h1e_v²) holds for L.

**Memory.** Peak, besides the caller's n own-RDM bar sets: one pair's
transition set (one ordered pair's dm1–dm3 bars at a time, each freed after
its call), the pair's two metric `BlockBars`
(Sr's dm3 and dm3_h, n_act⁶ each per side) and its norm intermediates while
a class walks; or, in part 3, one state's full `BlockBars` and blocks, as
`class_energies_adjoint`'s. Never n² sets: at n_act = 12 an n_act⁶ array is
24 MB. The host tables are three doubles per perturber per class per state.

**Conditioning.** The bars inherit the coupling's: a perturber whose own
norm N^J is small has a Rayleigh quotient η = H^J / N^J whose derivative,
η̇ = (Ḣ − η Ṅ) / N, is large, and the transition numerator T_IJ does not
shrink with N^J as fast as the 1/N grows. `h8_ring_sto3g_sa3`'s bars repeat
run to run to 3e-8 of their largest element, 2e-8 already at n = 1
(`class_energies_adjoint`'s own arithmetic); `h8_sto3g_sa3`'s to 3e-13
(RTX 3080). Along raw B, h or RDM directions the energy is not smooth at a
finite-difference ladder's scale on five of the six cases, one state alone
included; the orbital energies move Δ alone and are.

**`h2co_skew_ccpvdz_sa3`'s floor (#1055).** Its bars repeat only to ~1e-9
of their largest element, and #954's duality on it missed by up to 3.7e-8
(RTX 3080). Measured on both cards (`--preset default` and `hip`, one saved
SA-CASSCF point re-read in each process, card held), the source is:

- **Not the point moving.** Three in-process repeats at one point spread
  the c_2 column's miss over 3.4e-9 to 3.7e-8 (RTX 3080) and 5.0e-9 to
  1.7e-8 (gfx1200). The tables of a run at `conv_tol_grad` 1e-9 match the
  loose ones to seven figures, and the spread is the same (up to 3.0e-8).
- **Not the near-guard tuples.** Root 2's smallest kept norms (3.6e-13 to
  3.4e-12, Sijr, Srsi and the one-function Sij) repeat to 1e-16 relative.
  With the guard raised to 1e-8 (145 tuples dropped, in the tangent and
  the adjoint alike), the spread is unchanged (3.3e-8). At 1e-6 it falls
  to 2e-13.
- **Root 2's Si tuples, through the atomics.** With W's entries one at a
  time, only the couplings with ket 2 spread ((0, 1) repeats to 1e-13), and
  among the tangent's per-class Ċ only Si's does: tuples t0, t3 and t4, N
  6.6e-7, 2.0e-7 and 6.6e-7, with |Ṅ| / N up to 4.7e5 along root 2's c and
  η̇ to −1.1e6. Si's norm and H are three einsums of mixed RDM order
  (dm3_h, dm2_h, dm1_h) whose O(0.1) terms cancel to N, and einsum's
  atomics reassociate them, so each walk carries an absolute σ ≤ 4.2e-17 on
  N and H (Sr and Sir alike; every other class repeats to ≤ 3 ε relative).
  That is 1.4e-10 of N at t3. The tangent and the adjoint each walk the
  tables afresh, so through η̇ they evaluate the coupling's first
  derivative at two points 1e-10 apart in N: Ċ's Si share moves by 5e-10
  to 2e-9 per coupling, as measured, and the adjoint's root-2 CI bar y_2
  by 2e-9 to 8e-9 of 2.7 (roots 0 and 1: 3e-11).

So it is a floor, not a defect. The derivative is ill-conditioned at this
point, and a deterministic reduction would only make the two walks agree at
the same perturbed tables. The bound is first-order: the duality misses by
at most 2 Σ_IJ |W_IJ| Σ_t Σ_x |∂Ċ^t_IJ / ∂x| σ_x over x = N, H, T, Ṅ, Ḣ of
the finish's (N, d) form, with σ_x one unit round-off (ε/2) absolute in Sr,
Si and Sir and 4 ε |x| in the others. `Nevpt2QdTangent` computes it from the
tangent's own tables and holds every case to it (never under 1e-10). At the saved
point it is 1.5e-7 along c_2 (misses up to 3.7e-8 RTX 3080 and 1.7e-8
gfx1200), 1.4e-8 along all (4.2e-9, 1.9e-9), 5.6e-9 along B (1.3e-9,
1.4e-9) and 1.6e-9 along h (3.7e-10, 3.9e-10). Along c_1 it is
1.5e-10, along c_0 under 1e-10, and the misses there are at most 4.4e-11. Any SC-QD consumer reading
skew's root-2 bars (the QD Hessian, #965) inherits the same ~1e-8 relative
floor.

**A known limitation of SC, not a defect.** Park (JCTC 2019,
doi:10.1021/acs.jctc.9b00919; arXiv:1907.10180) finds SC-NEVPT2's analytic
gradients numerically unstable, especially over degenerate inactive orbitals
(refused here, #564), and recommends PC. The small-norm pole above is a
second route to the same fragility; PC divides by no norm and is unaffected.
So the tests ladder SC only where it is smooth, and gate one SC density at
its conditioning bound (#746, #766).

## The SC-QD forward tangent

    sc_effective_hamiltonian_tangent(res, blocks, pairs, batch, directions)
        -> ScEffectiveHamiltonianTangent
        elements                  sc_effective_hamiltonian's n² ClassEnergies
        kets [J]                  class_energies_tangent of pair (J, J): N^J, H^J, Δ, their tangents
        transition [IJ][k][t]     T^t_IJ (the diagonal: N^J_t)
        transition_dot [IJ][k][t m + u]
        empty [J][k]              ket J's guarded tuples (|N^J_t| <= 1e-14)
        element_dot [u][IJ][k], total_dot [u][IJ]
    pairs       sc_effective_hamiltonian's, every entry given
    directions  m n² EnergyDirections, [u n² + I n + J]: pair (I, J)'s along u;
                every field but the RDMs' (and the blocks') the same in each
                pair of one u (checked)

`reverse.cpp`, the QD analogue of [the SC forward tangent](#the-sc-forward-tangent)
for the QD-NEVPT2 nuclear Hessian (#954; `docs/nevpt2/nevpt2.tex`
Eqs. qd-rdm2 and qd-scfinish); `qd.cpp` checks the inputs. Along each
direction it gives every pair's transition tables' tangents and Ċ for every
ordered pair. **At n = 1 it is `class_energies_tangent`**, which is how each
state's own walk runs.

**The pair walk.** Each unordered pair M < N walks every class's slabs once
for all m directions, as the reverse pass's part 2 does, with a tangent
(`ScTangent`) that carries both sides' directions: the norm einsums of
(M, N) and (N, M) at their operands' tangents, each side's metric blocks
(dm3_h, dm2_h, hdm1, rm2, hdm2) its builder's polarization along that side's
own RDM tangents, Sr's, Srsi's and Sir's RDMs found among the inputs by
address. A block's ⟨I|J⟩ terms are constants, and cancel in the
polarization. T and Ṫ are the two einsums' means, so the same at (M, N)
and (N, M).

**The finish.** The coupling is C = −T N / d, d = N Δ + H over ket N's own
tables, differentiated in that form, with no 1/N:

    Ċ = −(N/d) Ṫ − (T H/d²) Ṅ + (T N/d²) Ḣ + (T N²/d²) Δ̇

(Eq. qd-scfinish's first derivatives); a one-function class's,
C = T c with c = −1/(Δ + η^N), is c Ṫ + T c² (Δ̇ + η̇^N), η̇ from the ket's
blocks' polarization. Sijrs couples no pair.

**The guarded tuples.** A tuple whose ket norm the coupling drops
(|N^N_t| ≤ 1e-14) adds nothing to Ċ, as the reverse pass gives it no bar,
so the tangent is the derivative of the guarded matrix. Its Ṫ^t_MN is still
formed and reported (`empty` names the tuples per ket, `transition_dot`
holds Ṫ): at a symmetric geometry it is ⟨Φ^M_t|∂Φ^N_t⟩, the first-order
overlap that decides whether the coupling's second order there exists
(Eq. qd-scfail), and which the SC-QD Hessian's refusal reads.

**What the caller forms.** The inputs are `EnergyInputs`, so the RDMs'
tangents are the caller's: a transition RDM is bilinear in its bra and ket,
so its tangent is the bra/ket `nevpt2::rdms` at (ċ_M, c_N) plus at
(c_M, ċ_N) (Eq. qd-rdm2), and each root's own, with its f3s, the
builders' (`builders_tangent`): the digests are linear in (tu|vw), so their
share is the one-vector `rdms` at its tangent. The orbital energies' is the
weights' average of each root's, ε being affine in γ.

**Memory.** Per pair, the reverse pass's walk with the tangent's per
direction held: each side's metric tangents (Si's dm3_h, n_act⁶, the
largest) and the slab tangents' buffer. The host tables are one T and m Ṫ
per tuple per class per ordered pair.

## The SC-QD Hessian-vector product

    sc_effective_hamiltonian_hvp(res, blocks, pairs, hbar, batch, directions, bars,
                                 transition, partners = {}) -> ScEffectiveHamiltonianHvp
        elements          sc_effective_hamiltonian's n² ClassEnergies
        populated [u][J]  per class, ket J's empty tuples direction u populates (K6's)
        defect [u]        the kets' ClassEnergiesHvp::defect on (u, its partner)
        coupling_defect [u]  the same ρ_τ, weighted by the couplings' first order
        coupling_k6 [u]   K6's share of ⟨u, HVP u⟩ through the couplings
        rounding, rounding_at, rounding_norm
                          as class_energies_hvp's, over the kets H̄ weighs, {ket, class, tuple}
    directions  sc_effective_hamiltonian_tangent's, m n²
    bars        m n EnergyBars, [u n + J]: direction u's, as the reverse pass's bars[J]
    transition  (u, bra, ket, bars): each ordered pair's transition bars' tangent
    partners    none, or m n², each pair's entry as the directions'

`reverse.cpp`, forward over [the SC-QD reverse pass](#the-sc-qd-reverse-pass),
for the QD-NEVPT2 nuclear Hessian (#962; `docs/nevpt2/nevpt2.tex` Eq.
qd-scfinish, the QD analogue of #811's
[SC Hessian-vector product](#the-sc-hessian-vector-product)); `qd.cpp`
checks the inputs. Along each direction, the tangent of every bar the
reverse pass forms: the shared fields' and Λ, each state's own RDMs', and
each ordered pair's transition dm1–dm3, handed over as the reverse pass
hands its own. **At n = 1 it is `class_energies_hvp`**, term for term.

**One finish for every coupling, the diagonal included.** The reverse
pass's bars are Σ over its tables of a multiplier times the table's
derivative. Per tuple, the coupling with ket N is g = −T N / d,
d = N Δ + H over ket N's tables, and the diagonal energy is the same g at
T = N, so a ket's own tables (N, H, Δ) take every coupling with that ket,
its own among them, and pair {M, N}'s T takes H̄_MN g_T of ket N and H̄_NM
g_T of ket M, half onto each of its two norm einsums:

    P_N = Σ_I H̄_IN g_N (+ H̄_NN g_T, the I = N coupling's T slot)
    Q_N = Σ_I H̄_IN g_H,   R_N = Σ_I H̄_IN g_Δ,   T̄_MN = H̄_MN g_T(N) + H̄_NM g_T(M)

These are the reverse pass's multipliers exactly (its A_J's share is
g_N, g_H and g_Δ). Their tangent is Eq. qd-scfinish's ten second
derivatives (g_TT = 0) times `sc_effective_hamiltonian_tangent`'s (Ṫ, Ṅ,
Ḣ, Δ̇), the I = N coupling's Ṫ its Ṅ. On the diagonal that is
E_NN = 2 g_TN + g_NN and the rest, Eq. finish's 3 x 3 (the host-only
facts, Euler's three zeros, the rank-one (H, Δ) block and the M = N
reduction, are identities of these formulas; not gated). A one-function
class is C = T c, c = −1/(Δ + η^N): c on T, and D̄ = Σ_I H̄_IN T c² on Δ
and through η^N on the ket's blocks, as the reverse pass puts them; its
tangent through ċ = c² (Δ̇ + η̇^N). One reverse pass at the tangent
multipliers takes them onto every bar: `fixed_adjoint` per state, and per
pair `pair_fixed_adjoint`, the pair walk with a table for its finish, its
metric bars swept onto the two transition sets.

**The bars at fixed multipliers, differenced.** As at n = 1, every table is
a polynomial of degree at most six in its inputs, so `kLineWeights`' six
passes at ±s, ±2s, ±3s are the bars' derivative at fixed multipliers
exactly: each state's own walk along its own entry, each pair's walk along
both its entries (its RDMs moved along their own tangents, the shared
fields and B along the direction's), at the smaller of the two sides'
steps. The transition bars are differenced with the shared ones and handed
over pair by pair. Sijrs reads no RDM: its share is `sijrs_hvp`'s at
Σ_J H̄_JJ.

**Empty tuples, on every root.** A tuple ket N's guard drops (|N^N_t| ≤
1e-14) that u populates, by Part I's rule over ket N's tables (second
tables of N and H along u, the population floor against u's scale), takes
K6 in every coupling with ket N, whatever H̄ weighs it: the multipliers'
limits along the ray, T̄ = −1/(Δ + η), η = H^{uu}/N^{uu} (the mean of η_u and
η_v with a partner both populate):

- on the diagonal, Part I's P̄ = 2T̄ + Δ T̄², Q̄ = T̄², at H̄_NN;
- in coupling (M, N), T̄ on T; and **where the bra's tuple is empty too**, T,
  N and H are all of second order along u, the coupling is the quadratic
  form T̄ T^{uu}, and its N and H multipliers have limits, −ρ η T̄² and
  ρ T̄², ρ = T^{uu}/N^{uu} (T's second table from the pair walk differenced,
  `pair_second_tables`). Against u, ḡ_T T^{uu} + ḡ_N N^{uu} + ḡ_H H^{uu} =
  T̄ T^{uu}; `coupling_k6` is that share;
- where the bra's tuple is populated, T is of first order (Eq. qd-scfail:
  Ṫ = ⟨Φ^M|∂Φ^N⟩), T/N grows as 1/s, and N's and H's multipliers have no
  limit: only T̄ on T is formed. What the omitted terms carry reads the
  tables' third order along the ray, which nothing here forms.

**The refusal, on every model state.** With partners, each ket's empty
tuples both directions populate are scored twice. Part I's weighted defect,
ρ_τ w / S over that ket's tables (`defect`, the diagonal case); and per bra
M, ρ_τ weighted by the tuple's share of coupling (M, N)'s first order,
max(|Ṫ_u|, |Ṫ_v|) / |Δ + η| against the largest such over ket N's tuples
(`coupling_defect`). A tuple with a zero transition tangent scores zero
there, so only a defect with a first-order overlap to some bra counts.
Either past `kEmptyDefectTolerance` (1e-6) refuses the call, Numerical, the
coupling's score first: its message names the coupling (bra, ket), the
class, the tuple and the directions (the block), the diagonal's the state,
the class and the perturber. Every ket is scored, its H̄ weight zero or
not: a state W does not weigh still enters through h_PQ.

**Cost.** Per direction, n own walks and n(n − 1)/2 pair walks at fixed
multipliers, each seven times (one at the tangent multipliers, six
shifted); with any guarded tuple, Part I's six `class_energies_tangent`
calls per ket (twelve with a partner), and six pair walks for each pair
with a tuple empty for both. The forward tangent runs once for every
direction and partner together. Memory: one pair's two transition sets at
a time, besides the reverse pass's.

## The PC reverse pass

    pc_class_energies_adjoint(res, blocks, in, batch, bars) -> PcClassEnergiesAdjoint
        energies         pc_class_energies' energies, dropped counts and total
        homogeneity[8]   Σ_G <M_G, G> per class, from the bars
    bars: EnergyBars, as class_energies_adjoint's

Also `reverse.cpp`, after SC's: the solve's adjoint (demo `docs/gradients.md`
§4.3, §5.6 and §7; demo `docs/nuclear_gradients.tex` Eqs. pcdot and
pcadj). A tuple's
energy is −c bᵀ A⁺ b, A = K + Δ S, b = S x + S₁ h1, and with x = A⁺ b, the
solve the energy ran,

    Ė = −c (2 xᵀ ḃ − xᵀ Ȧ x)

so **no eigenvalue or eigenvector derivative appears**. In the eigenbasis,
with g = y / (λ + Δ), each tuple gives the host its multipliers: ȳ = −2c g,
a weight w = c g and dE/dΔ = c g·g onto the orbital energies; for Sijr's and
Srsi's 2 direct − exchange, ȳ = −(4g − g_x) on the direct row and g on the
exchange row, w = 2g − g_x and dE/dΔ = 2 g·g − g·g_x. Each class walks its slabs as `pc_class_energies` does,
lists its slab einsums once as `ClassTerm`s (T's basis axes row-major after
k, so T is packed and `einsum_adjoint` writes its bar), and takes every
term's adjoint against ȳ: onto the slab and so Λ, h1e_v's rows, T̄ and t̄1.
M = Σ g wᵀ and M_Δ = Σ Δ g wᵀ accumulate on the device. After the walk:

    dE/dK = U M Uᵀ,   dE/dS = U T̄ᵀ + U M_Δ Uᵀ,   dE/dS₁ = U t̄1ᵀ

S and K symmetrized, as the solve symmetrized them (two gemms each, U, T̄
column-major d × m). Those bars go back through the einsums that built S, K
and S₁ onto SC's block bars, and **SC's sweep takes them on unchanged**: PC's
K is SC's Dyall blocks, its S and S₁ permutations of SC's metrics.

| class | S onto | K onto | S₁ onto |
|---|---|---|---|
| Sr | sr_dm3 (`"rpqbac->pqrabc"`) | a16 | sr_dm2 (`"rpqa->pqra"`) |
| Si | dm3_h (the same) | a22 | dm2_h (the same) |
| Sijr | hdm1 | a3 | |
| Srsi | srsi_dm1 | k27 | |
| Srs | rm2 (`"pqba->pqab"`) | a7 | |
| Sij | hdm2 | a9 | |
| Sir | sir_dm1, sir_dm2 (its four block terms) | a12, a13 | sir_dm1 |

a17, a19, a23, a25 and dm1_h stay zero. Sijr's exchange rows y[r, i, j] are
a second einsum with i and j swapped, and Srsi's the SrsiT slab's, so their
adjoints are terms like any other. Sir's two blocks of T are picked by a
selector over T's leading block axis, which keeps T one packed operand.
Sijrs is SC's `reverse_sijrs`.

**Homogeneity, the PC Eq. 40.** At fixed slabs and orbital energies a class
energy is homogeneous of degree one in (S, S₁, K) (the cut is relative to
S's largest eigenvalue), so `homogeneity[k] = Σ_G <M_G, G>` over the class's
blocks equals its energy: an exact identity from the bars, as Eq. 40 is
SC's. Sijrs carries SC's ½ <ḡ, g>.

**Where it is exact, and refusals.** The two-term derivative needs S's kept
rank locally constant, and at a fixed kept span it needs K to annihilate S's
dropped modes, which holds at an eigenstate for the null modes. Real modes
dropped at the cut (#853) break the second, K's part along them being of
order √σ_n; the kept span's rotation ([PC-QD](#pc-qd), #862) is the term
that restores it, at n = 1 as at n > 1. A
class the energy refuses is refused here with the energy's error (the same
`solve_class`, and one `pc_intruder_check` for both), never given a
zero; on an error the bars are unspecified.

**Conditioning.** The bars through U carry the solve's conditioning:
CAS(8,8)'s Sr and Si keep modes at 1.0e-10 and 3.1e-10 of their largest
eigenvalue (Srs 5.6e-7, Sir 1.6e-7), where a kept mode's λ moves as 1/σ under
a K that stops annihilating it. Two whole walks there put the bars on h1e,
h2e and the RDMs 1.3e-8 of their largest element apart on the gfx1200
(einsum's atomics reassociate S and K); the walk's own bars agree to 1e-13.
CAS(4,4) keeps every mode above 1.6e-5 (Sir), and its two walks agree to
1e-12.

**Memory.** Per class, S, K, S₁ and their builders as `pc_class_energies`
holds them (freed once solved), U beside T, and m × m M, M_Δ and the d × m
T̄; after the walk, per pair, two d × m and four d × d matrices, and the
blocks G rebuilt for the homogeneity sum. Each slab's y, x, ȳ, x̄ and
weights are buffers of their own.

### PC-QD

    pc_effective_hamiltonian_adjoint(res, blocks, pairs, hbar, batch, bars, transition)
        -> PcEffectiveHamiltonianAdjoint
        elements         pc_effective_hamiltonian's n² PcClassEnergies
        homogeneity[8]   Σ_G <M_G, G> per class, from the bars: Σ_IJ H̄_IJ E^k_IJ
    hbar, bars, transition: sc_effective_hamiltonian_adjoint's, read the same way

The reverse pass of [PC-QD-NEVPT2](#pc-qd-nevpt2), the bars of
L = Σ_IJ H̄_IJ H_IJ. **`pc_class_energies_adjoint` is its n = 1, H̄ = 1**
(`pc_couplings_adjoint`; `qd.cpp` checks the inputs, as for SC), so
everything above is that case of what follows.

Per class, each ket J's solve of (K_J, S_J, S₁_J) runs backwards as at
n = 1. The coupling of pair (I, J) is −c b_{I←J}ᵀ A_J⁺ b_J, bilinear in
the two rows, so with g_I = y_{I←J} / (λ_J + Δ) and g = y_J / (λ_J + Δ):

    on y_{I←J}:   −c H̄_IJ g             (2 direct − exchange: −H̄_IJ (2g − g_x))
    on y_J:       −c H̄_IJ g_I           (−2 H̄_IJ g_I, and H̄_IJ g_I on J's exchange row)
    on M_J:       c H̄_IJ g_I gᵀ         (H̄_IJ g_I (2g − g_x)ᵀ), and Δ times it on M_Δ
    dL/dΔ:        c H̄_IJ g_I·g          (H̄_IJ (2 g_I·g − g_I·g_x))

At I = J these are the n = 1 multipliers times H̄_JJ, and the own pair is
taken first in the n = 1 arithmetic (its two ȳ terms as one), so at n = 1
every multiplier is the one-state pass's to the bit. No eigenvalue
derivative, and the eigenvectors' only in the kept span's rotation below:
a mode that annihilates |J⟩ annihilates ⟨J| S^{(J,I)}, so b_{I←J} has no
part along a null mode, but a real dropped mode, ‖O_n|J⟩‖ = √σ_n, takes a
part of order √σ_n of it. ȳ_{I←J} goes through the
slab einsums onto the slabs (and Λ), h1e_v's rows and T̄_{I←J}, t̄1_{I←J};
then, U_J held,

    dL/dS^{(J,I)} = T̄_{I←J} U_Jᵀ   (column-major: the row-major buffer's own bar)
    dL/dS₁^{(J,I)} = U_J t̄1_{I←J}ᵀ

**unsymmetrized**, since T_{I←J} reads the raw S^{(J,I)}, onto pair (J, I)'s
metric blocks through the class's builder (the table above, S and S₁
columns), and so onto its transition RDMs. **No (I, J) / (J, I) mean**,
unlike SC: the bar on S^{(J,I)} is pair (I, J)'s alone. Each ket's own M,
M_Δ and T̄ collect every bra's share and go onto its own blocks as at
n = 1. A ket the energy refuses (S_J with no positive eigenvalue, or a
pole, [Poles, not signs](#poles-not-signs)) is refused with `pc_effective_hamiltonian`'s error, naming the
class and the state (`state_error`, the forward's), never given a zero.
Sijrs is SC's, weight Σ_J H̄_JJ.

**The kept span's rotation (#862).** Everything above holds U_J's span
still. Moving S_J rotates it into the dropped modes, δV_k = V_n Θ with
Θ_nl = (V_nᵀ δS V_k)_nl / (σ_l − σ_n), and the couplings see that through
every row's part along V_n: the own row's, σ_n V_nᵀ x, is negligible, but
K's, V_nᵀ K U, and each transition row's, V_nᵀ S^{(J,I)} x, are of order
√σ_n. So with Ū = dL/dU at fixed matrices,

    Ū = Σ_I (S^{(J,I)} T̄_{I←J} + S₁^{(J,I)} t̄1_{I←J}) + K U (M + Mᵀ)

(S U (M_Δ + M_Δᵀ) has no part along V_n), and U = V_k σ^−½ W gives one more
bar on S_J, symmetrized as the others:

    dL/dS_J += V_n [(V_nᵀ Ū Uᵀ V_k) ⊘ (σ_l − σ_n)] V_kᵀ

`pc_kets` keeps each solve's V, σ, V_nᵀ K U and V_nᵀ S, V_nᵀ S₁ for the
reverse pass alone (`DroppedModes`), and each transition's V_nᵀ S^{(J,I)}
and V_nᵀ S₁^{(J,I)} beside its T; `pc_span_bar` forms the bar from the walk
bars before `pc_matrix_bars` spends the own T̄. A pair with σ_l − σ_n at or
below ε √d of S's largest eigenvalue, σ's round-off, is skipped. Nothing
is added where nothing was dropped, so a class with no cut is unchanged.
`../gradient/README.md`, "The kept span's rotation", has why it matters,
and the measurements.

**Homogeneity.** E_IJ is homogeneous of degree one in ket J's (S_J, K_J,
S₁_J) and pair (J, I)'s (S, S₁) together, so `homogeneity[k]`, Σ_G <M_G, G>
over every state's own blocks and every transition's metric blocks, is
Σ_IJ H̄_IJ E^k_IJ; and Euler's degree 4 in (B, h1e_v²) holds for L.

**All pairs at once.** Unlike SC's pair-at-a-time walk, each class walks
its slabs once with every pair, as the forward does (n² einsums per slab,
and their adjoints), since every coupling with ket J reads J's one solve.
Every transition's metric `BlockBars` are held until all eight classes have
walked; then each state's own bars are swept onto bars[J] (and the shared
h1e, h2e), and each transition's onto its dm1–dm3 (`sweep_metrics`),
handed to `transition` in row-major pair order and freed.

**Memory.** Besides the caller's n own-RDM bar sets: per ket with modes
dropped, V (d × d) and V_nᵀ S (nd × d) beside its U, and nd × d per
transition beside its T; n² − n metric
`BlockBars` (Sr's dm3 and Si's dm3_h, n_act⁶ each, the largest), the same
order as the forward's n² − n transition T (d × m_J, n_act⁶ for Sr and
Si) held across a class; per class the walk bars T̄ beside them; and one
transition's dm1–dm3 bars at a time.

**Conditioning.** As at n = 1, the bars through U_J carry each ket's
solve's conditioning. On `h8_ring_sto3g_sa3` they repeat run to run to
6.9e-13 of each bar's largest element for state 0 alone, 7.3e-14 for state
2, but 2.2e-11 for state 1, whose Sr keeps small modes; on `h8_sto3g_sa3`,
whose root 0's PC passes only over round-off, to 1.4e-7 (RTX 3080). Along
raw h, or the own RDMs rotated with f3ca/f3ac raw, a ket's PC is refused
within 1e-3 to 1e-5 of the base point on `h2co_skew_ccpvdz_sa3`,
`h8_sto3g_sa3` and `h8_ring_sto3g_sa3` (state 1's Sr denominator λ + Δ is
−33 at t = 1e-3 along the RDMs there).

## The PC forward tangent

    pc_class_energies_tangent(res, blocks, in, batch, directions, span = true)
        -> PcClassEnergiesTangent
        energies                     pc_class_energies' energies, dropped counts,
                                     cut, kept and total
        energy_dot [u][k], total_dot [u]
    directions: EnergyDirection, as class_energies_tangent's
    span: false leaves the kept span's rotation out, for the tests alone

`reverse.cpp`, for the nuclear Hessian (milestone 24, #810): forward mode
over the PC reverse pass's own one-state walks, as SC's is over SC's. Each
class's solve is the energy's, reused, with no eigenvalue or eigenvector
derivative. Per tuple, with g = y / (λ + Δ) and ȳ, x̄, R the reverse pass's
multipliers,

    Ė_t = ȳ·ẏ + x̄·ẋ + R Δ̇ + gᵀ Ḋ w,   Ḋ = K̇' + Δ Ṡ'

(w the weight, c g or 2g − g_x), summed over the tuples as
<M, K̇'> + <M_Δ, Ṡ'> once the walk has accumulated M and M_Δ. So Ė is the
reverse pass's own expression read the other way, which is what makes the
two dual.

**ẏ and ẋ** are each slab einsum at its operands' tangents
(`tangent_terms`, SC's): the slab's (rebuilt along Ḃ), h1e_v's rows, and T's
and t1's, registered once the class is solved. **The matrices' tangents**
Ṡ, K̇ and Ṡ₁ are their builder's (`pc_*_matrices`) central polarization, exact
as SC's blocks' are (S, S₁ linear and K one integral times one RDM), S and K
symmetrized as the solve symmetrized them. Then, at fixed U,

    Ṫ = Ṡ U,   ṫ1 = Ṡ₁ᵀ U,   K̇' = Uᵀ K̇ U,   Ṡ' = Uᵀ Ṡ U

**The kept span's rotation (#862).** Where the cut drops modes, U also
moves into them at first order, the forward side of the reverse pass's
`pc_span_bar` ([PC-QD](#pc-qd)): S = V diag(σ) Vᵀ moves V_k by V_n Θ,
Θ_nl = (V_nᵀ Ṡ V_k)_nl / (σ_l − σ_n), the pairs within σ's round-off
skipped as there, and U = V_k σ^−½ W by U̇ = V_n Θ V_kᵀ U (U̇'s part within
the kept span changes no energy). Through `DroppedModes`' V_nᵀ S, V_nᵀ S₁
and V_nᵀ K U:

    Ṫ += (V_nᵀ S)ᵀ Θ',   ṫ1 += (V_nᵀ S₁)ᵀ Θ',   K̇' += X + Xᵀ, X = (V_nᵀ K U)ᵀ Θ'

with Θ' = Θ V_kᵀ U; Uᵀ S U̇ = σ_n Uᵀ V_n Θ' = 0, so Ṡ' has no such term,
as Ū has no S U M_Δ one. For exact null modes the term is round-off (V_nᵀ S
and V_nᵀ K U vanish at an eigenstate); for real ones (#853's cut) it is of
order √σ_n, and without it Ė misses the corrected bars.

**Refusals** are the energy's: the same solve, and the walk's denominators
through the same `pc_intruder_check` before Ė is finished, so no positive
metric eigenvalue, a pole (|t| > 1) and a denominator at round-off each come
back with `pc_class_energies`' kind and message, never as a class summed as
zero. **Sijrs** is SC's kernel (`sijrs_tangent_walk`). **`energies.cut`**
reports how far above round-off each class's cut reaches: a few ε √d for
exact null modes, more for real ones; **`energies.kept`** what the solve's
conditioning follows ([The conditioning floor](#the-conditioning-floor)).

**Where duality is checkable.** Ė and the reverse pass's Σ ⟨bar, ẋ⟩ are one
bilinear form evaluated two ways, so they agree to round-off times the
solve's conditioning. With the kept spectrum gapped that is ~1e-11 or
better ([Tests](#tests)). At PSB3 twisted to 90° (STO-3G, state 0) it is
not: real modes run through the cut there, Sr and Si keep modes at about
3e-13 of σ_max, and the solve amplifies the einsums' atomic round-off in S
and K by about 1e12, so the bars along any direction that moves K repeat
run to run only to a few 1e-6 relative (2e-5 before #916), and the tangent
with them. There the span term is checked for visibility only; the physical
directions do not see that noise ([The conditioning
floor](#the-conditioning-floor), #917).

**Batched.** Each class walks once for all m directions: its matrices'
tangents per direction once solved (S, K and S₁ built twice per direction,
then Ṫ, ṫ1, K̇' and Ṡ' held through the walk, d × m and two m × m each), and
per slab, per direction, the slab's tangent and ẏ (and ẋ), read back.

### PC-QD

    pc_effective_hamiltonian_tangent(res, blocks, pairs, batch, directions, span = true)
        -> PcEffectiveHamiltonianTangent
        elements                  pc_effective_hamiltonian's n² PcClassEnergies
        element_dot [u][IJ][k], total_dot [u][IJ]
    pairs, directions  sc_effective_hamiltonian_tangent's: every entry given,
                       m n² directions, [u n² + I n + J] pair (I, J)'s along u
    span               false leaves the kept span's rotation out, for the tests alone

The QD analogue of the one-state tangent above, for the PC-QD-NEVPT2
nuclear Hessian (#960; `docs/nevpt2/nevpt2.tex` Eqs. qd-rdm2 and
qd-pcfinish, its first order); `qd.cpp` checks the inputs as for SC. **The
one-state tangent is its n = 1** (`pc_couplings_tangent`), as
`pc_class_energies_adjoint` is the PC-QD reverse pass's.

Per class and tuple the coupling is −c β, β = b′ᵀ A⁺ b with b′ = b_{I←J},
b = b_J and A = K_J + Δ S_J. With x = A⁺ b and u = A⁺ b′,

    β̇ = xᵀ ḃ′ + uᵀ ḃ − uᵀ Ȧ x   (+ the kept span's rotation, below)

the first-order residuals r = ḃ − Ȧ x and r′ = ḃ′ − Ȧ u read once each. In
ket J's eigenbasis, with g_I = y_{I←J} / (λ + Δ) and g = y_J / (λ + Δ):

    single:   Ċ_IJ = −c (ẏ_{I←J}·g + g_I·ẏ_J) + c g_I·g Δ̇ + g_Iᵀ Ḋ_J (c g)
    pair:     Ċ_IJ = −(2 ẏ_{I←J}·g + 2 g_I·ẏ_J − g_I·ẋ_J − ẏ_{I←J}·g_x)
                     + (2 g_I·g − g_I·g_x) Δ̇ + g_Iᵀ Ḋ_J (2g − g_x)

with Ḋ_J = K̇′_J + Δ Ṡ′_J ket J's projected pencil, summed over the tuples
as <M_IJ, K̇′_J> + <M_Δ,IJ, Ṡ′_J>, M_IJ = Σ_t g_I wᵀ held per pair. At I = J
this is the one-state Ė term for term.

- **Each ket's solve** moves as at n = 1: Ṡ_J, K̇_J and Ṡ₁_J their builder's
  polarization over pair (J, J) along its own directions, Ṫ_J = Ṡ_J U_J.
- **Each transition row** T_{I←J} = S^{(J,I)ᵀ} U_J moves along both of its
  sides: S^{(J,I)} and S₁^{(J,I)} by their builder's polarization over pair
  (J, I) along that pair's directions, which carry Eq. qd-rdm2's bra and
  ket tangents of the transition RDMs, unsymmetrized as T reads them.
- **The kept span's rotation (#862).** Where S_J's cut drops modes, U_J
  rotates into them, U̇_J = V_n Θ′, as at n = 1, and each transition row
  takes it through its leak (`PcKets::leak`): Ṫ_{I←J} += (V_nᵀ S^{(J,I)})ᵀ
  Θ′, ṫ1 the same over V_nᵀ S₁^{(J,I)}. This is the forward side of the
  reverse pass's `pc_span_bar`, whose Ū sums every bra's T̄_{I←J}; a
  transition row's part along a real dropped mode is of order √σ_n, so it
  is the term the transitions need, where the own row's is negligible.

**Refusals** are the energy's: the same `pc_kets` solves and one
`pc_intruder_check` per ket before Ċ is finished, so a ket's metric with no
positive eigenvalue or a pole comes back with `pc_effective_hamiltonian`'s
error, naming the class and the state. **Sijrs** is SC's kernel, on the
diagonal; it couples no pair.

**Conditioning.** The tangent and the reverse pass each walk the solves
afresh, and each solve amplifies its einsums' atomic round-off on S and K by
its metric's condition number over the kept range (#917). On
`h2co_skew_ccpvdz_sa3` root 1 keeps Sijr, Srs and Sir modes at 6e-8 to 2e-7
of σ_max. Repeated in one process at one point (diagnosis), the PC-QD
couplings with ket 1 spread in Sijr and Sir by up to 2.8e-10, and the
duality's c_1 and all columns by 6e-11 and 3e-11 of the largest summand,
in the tangent and the adjoint alike; every other ket's couplings repeat to
round-off. The spread is at most 0.07 of ε |Ċ^k_IJ| / kept_{J,k}, so
`Nevpt2QdPcTangent` holds every case to twice that, summed under |W|, from
the tangent's own elements, where it passes 1e-10 of the largest summand,
as `Nevpt2QdTangent` does with SC's floor. At `pc_floor` 1e-3 the kept
modes clear 1e-3 and skew repeats to 1e-15.

**Memory.** Besides the energy's (n solves and n² − n transition rows per
class): per direction, each ket's Ṫ, ṫ1, K̇′ and Ṡ′ and each transition's
Ṫ and ṫ1 (d × m_J each), and per pair M and M_Δ (m_J × m_J), held across
the class's walk; per slab, every pair's ẏ, read back per direction.

## The PC Hessian-vector product

    pc_class_energies_hvp(res, blocks, in, batch, directions, bars, span = true)
        -> PcClassEnergiesHvp
        energies         pc_class_energies' energies, dropped counts, cut and kept
    bars[u] (EnergyBars, added onto): the tangent of pc_class_energies_adjoint's
        bars along directions[u], every field they name, Λ included
    span: false leaves the kept span's rotation out at both orders, for the tests alone

`reverse.cpp`, forward over the PC reverse pass, for the nuclear Hessian
(milestone 24, #815; `docs/nevpt2/nevpt2.tex`, "PC's class tangent"). SC's is one tangent of
the finish's multipliers and the bars at fixed multipliers differenced
([The SC Hessian-vector product](#the-sc-hessian-vector-product)); PC's is
the same split over the solve.

**The bars at fixed multipliers.** Per class, the reverse pass reads the
solve through U, the slab rows' bars ȳ (and the exchange rows' x̄),
M = Σ g wᵀ and M_Δ = Σ Δ g wᵀ, and for the kept span's rotation S's
eigenvectors V and eigenvalues σ; given those, every bar is a polynomial in
the inputs: T = S U and t1 = S₁ᵀ U rebuilt over the inputs' own matrices,
the slab einsums' adjoints against ȳ, U T̄ᵀ + U M_Δ Uᵀ and U M Uᵀ onto S and
K, and the span bar over V_nᵀ S, V_nᵀ S₁ and V_nᵀ K U formed anew. A *fixed
walk* is that reverse pass with no solve: the class walks of
`pc_class_energies_adjoint` with the multipliers handed in. The tangent is
then three terms, every one a fixed walk:

- **The multipliers' tangent**, one fixed walk at them. With g = y / (λ + Δ)
  (g_x the exchange row's), the solve differentiated through its own
  factorization:

      ġ = (ẏ − Ḋ g) / (λ + Δ),   Ḋ = K̇' + Δ Ṡ' + Δ̇ 1,

  ẏ and Ḋ `pc_class_energies_tangent`'s, the span's first-order rotation in
  them (U̇ = V_n Θ' in Ṫ, ṫ1 and K̇'). The reverse pass's multipliers are
  linear in g at a single coupling (ȳ = −2s g, w = s g) and in (g, g_x) at a
  pair (ȳ = −(4g − g_x), x̄ = g, w = 2g − g_x), so their tangents are the
  same in (ġ, ġ_x); M moves by Σ (ġ wᵀ + g ẇᵀ), M_Δ by
  Σ Δ̇ g wᵀ + Δ (ġ wᵀ + g ẇᵀ), and R = gᵀ (Uᵀ S U) w onto the orbital
  energies by ġ·w + g·ẇ + gᵀ Ṡ' w (UᵀSU = 1 holds at the point only).
  This is exact for any U that moves with g so that U g is the amplitude
  vector A⁺ b; U̇ = V_n Θ' is one such.
- **U's tangent**, U̇ = V_n Θ', Θ' = Θ V_kᵀ U: the bars are quadratic in U
  at fixed inputs, so one symmetric difference, U ± s U̇ with s the power of
  two nearest ‖U‖ / ‖U̇‖, is exact (two fixed walks). It carries S U̇ into
  the slab bars, U̇ into the matrix bars, and K U̇ into the span bar.
- **The inputs' tangent at fixed multipliers**: S, S₁ and the slabs linear,
  K one integral times one RDM, so the bars are of degree at most five
  along a line, and SC's ±s, ±2s, ±3s difference is exact (six fixed walks;
  the orbital energies, read by the multipliers alone, are not moved).

Sijrs's share is SC's (`class_energies_hvp`'s kernel).

**The span bar's derivative in V and σ.** The span bar is
Y = V_n Ψ V_kᵀ, Ψ = (V_nᵀ R V_k) ⊘ Γ, Γ_nl = σ_l − σ_n, R = Ū Uᵀ: a
two-sided divided difference of S's spectral decomposition, so its
derivative at fixed R is Daleckii–Krein's. With E = Vᵀ Ṡ V and
Θ = E_nk ⊘ Γ, and R's blocks off V_kᵀ R V_k zero (U ⊥ V_n),

    Ẏ = V_n [(E_nn Ψ − Ψ E_kk − Θ R_kk) ⊘ Γ] V_kᵀ + V_n Ψ Θᵀ V_nᵀ − V_k Θᵀ Ψ V_kᵀ.

Every divided difference of 1/(σ_l − σ_n) is a product of two first-order
ones, so no pair within one side of the cut is divided by: degenerate
dropped or kept modes need no care. A pair within σ's round-off rotates
nothing, at second order as at first (Ψ and Θ zero there). Added to the
multipliers' fixed walk, symmetrized as the span bar is. These, U's
tangent, and the span's parts of ẏ and Ḋ are **the null-space rotation
terms**: the rotation of S⁺ = U Uᵀ's span into the dropped modes, and of
P = 1 − T Uᵀ = V_n V_nᵀ with it. For exact null modes at an eigenstate
they vanish with V_nᵀ Ṡ V_k and V_nᵀ K U; for real ones (#853's cut, or a
raised floor) they are of the order of the first-order term.

**No on-shell assumption.** The solve's restricted inverse
G = V_k (V_kᵀ A V_k)⁻¹ V_kᵀ is differentiated as it stands, the dropped
modes' K (V_nᵀ K U, through the span bar) included, so nothing assumes
P A = 0: off the eigenstate, where P A is of the CI residual's order and
S⁺ amplifies it by 1/σ_kept, the terms carry it. The ladders below move
the RDMs off any eigenstate.

**Refusals.** The energy's (`pc_intruder_check`, the tangent's walk), and a
class whose smallest kept metric eigenvalue is below 1e-10 of its largest,
Numerical, naming the class and `pc_floor`: there the second derivative is
not attainable ([The conditioning floor](#the-conditioning-floor), #917).
Raising `EnergyInputs::pc_floor` to 1e-10 or more passes it. The Schur lift
is the gradient's (`pc_hessian_product`; `../gradient/README.md`, "The
Schur lift"): along the class inputs S is linear, so nothing lifts here.

**Measured** (RTX 3080, `--preset default`). `n2_ccpvdz_cas44_df` at
`pc_floor` 1e-3, which drops real modes the default keeps (Srs, Sij and
Sir) and leaves every kept spectrum gapped: the product against a ladder of
the bars, every field, along every input group and all at once, worst
1.9e-9 of the field's largest element (the issue's bar is 1e-7); without
the rotation (`span` false) it misses by 1.9e-3 (the orbital energies) to
2.8e-1 (h), 1.9e4 to 2.8e6 bars. Each group runs in under half a second.

**One set of matrices at the base inputs (#1318).** The base walk keeps
every class's S, K and S₁ as it solved over them (`pc_kets`' `matrices`),
one set per pair, and every walk at the base inputs reads them: the
multipliers' fixed walk, U's two, and the span bar's derivative. Only the
six walks of the inputs' line, at shifted inputs, build their own. A
rebuild is not the same matrix: the builders' einsums accumulate with
atomics, so each build carries its own round-off, ε |S| and ε |K|. U, V and
σ are the base build's, and the fixed walks lean on identities that hold
only for the matrices they came from (Uᵀ S U = 1, V_nᵀ S = σ_n V_nᵀ); the
solve amplifies a rebuild's difference by 1 / kept. The product then was
not the derivative of any one function, and its asymmetry was about
0.05 to 0.1 ε / kept of its largest summand, the same as the run-to-run
spread of the bars themselves. Reading the kept set, it is about
1.3e-3 ε / kept: PSB3's twist (`Nevpt2PcEnergyHvp.SymmetricAtTheCut`),
measured in one process against the rebuild, RTX 3080:

| `pc_floor` | kept | asymmetry, rebuilt | kept set | × kept / ε, kept set |
|---|---:|---:|---:|---:|
| 1e-10 | 1.1e-10 | 8.7e-8, 2.2e-7 | 2.6e-9, 2.8e-9 | 1.3e-3 |
| 3e-10 | 3.6e-10 | 6.6e-9, 1.8e-8 | 1.6e-9, 1.7e-9 | 2.6e-3 |
| 1e-9 | 1.2e-9 | 5.2e-9, 9.3e-9 | 2.9e-10, 3.9e-10 | 1.6e-3 to 2.1e-3 |
| 1e-8 | 1.1e-8 | 5.7e-10, 1.2e-9 | 2.3e-11, 4.2e-11 | 1.1e-3 to 2.0e-3 |
| 1e-7 | 1.1e-7 | 4.3e-11, 2.7e-11 | 1.5e-11, 1.5e-11 | (rounding) |

(Each the largest of three products over the largest summand; two A/B
pairs per row.) The run-to-run spread is another matter, and is not
smaller: each call's base walk is one realization of the atomics, and its
solve amplifies their round-off as the energy's adjoint does (#917). The
differences' steps set neither: larger steps (U's 16-fold and 256-fold, the
line's 4-fold, the polarizations' 8-fold) lowered neither.

**One set across calls: `PcMatrixStore`.** A product split over several
calls, as the PC Hessian splits its coordinates into chunks, pairs
⟨t_i, H t_j⟩ from one call with ⟨t_j, H t_i⟩ from another, so the spread
between calls becomes its asymmetry. `EnergyInputs::pc_matrices` names a
store that the first PC walk at the point fills (`pc_kets` keeps a copy of
every class's S, K and S₁) and every later one reads in place of its own
build: the adjoint's, each product's base walk, and so their fixed walks.
The line's walks still build at their shifted inputs, over the call's own
inputs; each product adds the store's matrices less a build at its own base
inputs (`PcHvpClass::offsets`), so the line meets the store at its base
point. Its holder guarantees the point: a store read at other inputs would
solve another point's matrices. `pc_hessian_product` takes one, and the PC
Hessian hands one store to all of its chunks (and none to the degenerate
pairs' products, at turned orbitals). PSB3's twist, two directions in two
calls (`SymmetricAtTheCut`, split; RTX 3080, one process):

| `pc_floor` | without a store | over one store |
|---|---:|---:|
| 1e-10 | 7.9e-8, 8.3e-8 | 1.5e-10, 6.1e-10, 7.4e-10 |
| 3e-10 | 2.8e-10, 1.8e-8 | 7.9e-10, 8.2e-10, 8.5e-10 |
| 1e-9 | 1.9e-8, 2.8e-8 | 6.3e-11, 3.6e-10 |
| 1e-8 | 1.0e-10, 1.2e-9 | 4.2e-12, 6.2e-12 |

Pinning the point's inputs as well (the first call's RDMs and integrals,
read by every later one) was tried at the Cope study and moved nothing past
its scatter, so the store holds the matrices alone.

**Cost.** One tangent walk for every direction at once, then per
direction nine fixed walks (each one reverse pass) and Sijrs's kernel.
The base walk keeps, per class, U, V, M and M_Δ, its matrices (every
pair's S, K and S₁, and over a store as much again in offsets), and per
slab ȳ and x̄ (the slab rows' size, on the host), each with its tangent per
direction, and per direction the span bar's derivative (d × d) and U̇
(d × m).

### PC-QD

    pc_effective_hamiltonian_hvp(res, blocks, pairs, hbar, batch, directions, bars,
                                 transition, span = true) -> vector<PcClassEnergies>
        the n² elements, as pc_effective_hamiltonian computes them
    pairs, hbar        pc_effective_hamiltonian_adjoint's
    directions         pc_effective_hamiltonian_tangent's: m n², [u n² + I n + J]
    bars[u n + J]      added onto: the tangent of that reverse pass's bars[J] along u
    transition         (u, bra, ket, bars): each ordered pair's dm1–dm3 bars' tangent
    span               false leaves the kept span's rotation out, for the tests alone

The QD analogue of the one-state product above, for the PC-QD-NEVPT2
nuclear Hessian (#964; `docs/nevpt2/nevpt2.tex` Eq. qd-pcfinish), forward
over [the PC-QD reverse pass](#pc-qd). **`pc_class_energies_hvp` is its
n = 1, H̄ = 1** (`pc_couplings_hvp` in `reverse.cpp`; `qd.cpp` checks the
inputs as for SC's), so the one-state product is this one's arithmetic.
Per class and tuple the coupling is −c β, β = b′ᵀ A⁺ b, and its second
derivative is

    ∂₁₂β = xᵀ ∂₁₂b′ + uᵀ ∂₁₂b − uᵀ ∂₁₂A x + r′₁ᵀ A⁺ r₂ + r′₂ᵀ A⁺ r₁
           + (the rank-projector terms of A⁺),

x = A⁺ b, u = A⁺ b′, r = ∂b − ∂A x and r′ = ∂b′ − ∂A u. The three terms of
the one-state product carry it over every ket's solve, each a fixed walk
of the pair reverse pass (`pc_fixed_adjoint`, every pair at once):

- **The multipliers' tangent** (the r′ᵀ A⁺ r cross terms). Per ket J and
  bra I, through J's own factorization,
  ġ_I = (ẏ_{I←J} − Ḋ_J g_I) / (λ + Δ), ẏ and Ḋ_J the forward tangent's
  (`pc_effective_hamiltonian_tangent`, each transition row's leak through
  the kept span's rotation in ẏ); the reverse pass's multipliers are linear
  in the g's, so ȳ, x̄, M_J = Σ gs wᵀ (gs = Σ_I H̄_IJ g_I) and M_Δ,J move by
  the same rules in the ġ's, and R onto the orbital energies by
  ġs·w + gs·ẇ + gsᵀ Ṡ′ w (`pc_hvp_slab`).
- **Every ket's U along U̇_J = V_n Θ′_J**, one symmetric difference (the
  bars are quadratic in U): T_{I←J} = S^{(J,I)ᵀ} U_J moves with it, so every
  transition row does.
- **The inputs' tangent at fixed multipliers** (the ∂₁₂ terms), every pair
  along its own direction at once, the six-point line difference; the
  transitions' matrices are rebuilt over each pair's shifted RDMs.

**The rank-projector terms** are the kept span's: each ket's span bar
collects every bra's T̄_{I←J} through Ū = Σ_I (S^{(J,I)} T̄_{I←J} +
S₁^{(J,I)} t̄1_{I←J}) + K U (M + Mᵀ), and so does its Daleckii–Krein
derivative (`pc_hvp_keep`'s R_kk, over each transition's matrix rebuilt),
U's tangent, and the transitions' leaks in the fixed walks. They read
b′'s part along S_J's dropped modes: of order √σ_n for a real mode, zero
at an exact null mode, whose first derivative is not zero where it lifts
(`../gradient/README.md`, "The Schur lift on every model state"). As in
Part I, S_J does not move along an orbital direction, so they enter the
CI–CI and CI–orbital blocks alone. Sijrs's share is SC's at Σ_J H̄_JJ.

**Refusals** are the one-state product's on every ket: a ket whose class
keeps a metric mode below 1e-10 of its largest is Numerical, naming the
state and the class (#917), and the energy's own refusals come back as
`pc_effective_hamiltonian`'s.

**Cost.** One tangent walk for every direction and pair at once, then per
direction nine fixed walks over all n² pairs and Sijrs's kernel; the base
walk keeps, per class and ket, U, V, M and M_Δ and their tangents, and per
slab every pair's ȳ (and the kets' x̄) with theirs.

## Not ported

The nuclear Hessians (`../../hessian/`). The demo's
`energiesDevice` over whole blocks is `batch <= 0` here.

## Tests

`test/gtest/nevpt2_qd_energy_gpu.cpp`, SC-QD, against
`test/gtest/data/qd_nevpt2/` (all six cases). `Nevpt2QdEnergy`: from the
golden transition RDMs, every element's per-class coupling and transition
norm against `couplings` and `t_norms`, and its total against `heff2`, at
1e-10; each diagonal element against `class_energies` on that state's
inputs, and the walk in batches of two against the whole one, at 1e-13.
`Nevpt2QdEnergyDeviceRdms`: the same against the goldens with the
transition RDMs from `nevpt2::rdms` over the golden CI vectors (whose signs
fix the off-diagonals' phases), on the three three-root cases.
`Nevpt2QdEnergyRefusals`: pair inputs that are not n², or that do not share
their integrals, are InvalidConfig (SC and PC), and so are omissions out of
rule (one diagonal without the other, an off-diagonal without its
transpose, the diagonal omitted with no kets or another call's) and an
omitted entry in the SC reverse pass. `Nevpt2QdEnergyPairs`, on every
case: the step's path, one diagonal call then one per pair with the diagonal
omitted, against the 2 × 2 call per pair it replaced, SC and PC, each class
of each element within 1e-10 of the pair's largest diagonal class energy
(the off-diagonal's perturber terms are bounded by its two states' own),
with equal dropped counts.

PC-QD, in the same file. `Nevpt2PcQdEnergy`, on all six cases: from the
golden transition RDMs, every element's per-class coupling
against `pc_couplings` and its total against `pc_heff2` at 1e-10, every
element's dropped counts against its ket's `pc_dropped` row; each diagonal
element against `pc_class_energies` and the walk in batches of two against
the whole one, at 1e-13 with equal dropped counts. `Nevpt2QdEnergyDeviceRdms`
checks PC too, from `nevpt2::rdms`' transition RDMs, on the three
three-root cases against the goldens. `Nevpt2PcQdEnergyRefusals`: on
`h8_sto3g_sa3` with state 1's own RDMs zeroed, which `pc_class_energies`
refuses (Sr has no positive eigenvalue), the effective Hamiltonian is a
Numerical error naming Sr and state 1; states 0 and 2 pass alone.

`test/gtest/nevpt2_energy_gpu.cpp`. SC: from PySCF's dm1/dm2/dm3/f3 and the
slabs over the golden B blocks (`test/gtest/data/nevpt2/`), every class norm and
energy and the total against PySCF's DF-NEVPT2 at 1e-7 Ha, on the three cases
that carry dm3 and f3, walked whole and in batches of three. And the chain
#416 → #417 → #418: the same, with `nevpt2::rdms`' dm3, f3ca and f3ac from the
golden CI vector in place of PySCF's, on all five cases (the two CAS(10,10)
cases carry no dm3 or f3, so this is where they are checked).

PC, in the same two tests: every class energy and the total against the
golden's block2 `pc_class_energies` and `e_pc_total` at 1e-7, and each
class's dropped count against the demo's measured table, from PySCF's RDMs
and from `nevpt2::rdms`', walked whole and in threes. And
`Nevpt2PcEnergy.ZeroMetricIsRefused`: zero RDMs are a Numerical error
naming Sr. `Nevpt2PcEnergy.NegativeAndUncoupledNearPoleAccepted`
(`n2_ccpvdz_cas44_df`, whose nearest Sr mode has y = 0): what #874 stopped
refusing ([Poles, not signs](#poles-not-signs)). Every virtual at −1000 Eh,
every Sr denominator negative, is accepted with Sr's energy positive; and
the virtuals shifted to put that mode at λ + Δ = −1e-6 are accepted, its
|t| zero.

`Nevpt2PcInvariance`: PC's invariance to the active orbitals, which the
pipeline step relies on (it does not rotate to each root's natural
orbitals). Every input with an active index (B_aa, B_ca, B_va, h1e, h2e,
`h1e_v_Sr`, `h1e_v_Si`, dm1, dm2, dm3, f3ca, f3ac) is rotated by one random
orthogonal U, and every SC norm and energy, every PC energy and the PC total
must match the unrotated run's to 1e-10, with the same dropped counts.
CAS(4,4) and CAS(8,8), both of which cut modes in Sr, Si and Sir. SC is the
control: its invariance is PySCF's, so an SC miss would be the rotation's.
Measured on an RTX 3080: PC within 4e-15.

The reverse pass, `Nevpt2EnergyAdjoint` and `Nevpt2EnergyAdjointLadder`, on
CAS(8,8) from PySCF's RDMs. At CAS(4,4), Si vanishes by N₂'s symmetry
((σπ|ππ) and h1e_v_Si are zero), and so would its bars.

- `Eq40AndEulerFromTheBars`: each class's Eq. 40 against its energy at 1e-12.
  Euler's identity checks the slab and h1e_v bars step-free: at fixed orbital
  energies E is homogeneous of degree 4 in (B, h1e_v²), so
  ⟨Λ, B⟩ + 2 Σ ⟨h1e_v̄, h1e_v⟩ = 4E. The energies are checked against the
  golden ones as well.
- `BarsMatchRichardsonLadder`: Σ ⟨bar, ẋ⟩ over every input (B via Λ,
  h1e_v_*, e_core and e_virt, h1e and h2e, dm1–dm3, f3ca and f3ac), along a
  seeded random direction (all together, then each of five groups alone:
  B, h1e_v, e, h, rdm), against central differences of `class_energies`
  itself. Successive changes must fall by a factor in (3.2, 4.8) over at
  least three halvings, and the Richardson extrapolant at that run's finest
  pair must agree to 1e-7 relative. The RDMs move as raw tensors (a duality
  check of linear maps, not a CI displacement), dm1–dm3's direction averaged
  with its index-reversed self, the one permutation symmetry PySCF's carry;
  f3ca and f3ac carry none, and h1e and h2e move as raw arrays.
  - **The RDM direction is a tenth the length of the others.** At full
    length, Sr's energy along it is singular about |t| ≈ 1e-3 from the base
    point. Measured at seed 560, the five RDMs together: Sr's ratios ran
    5.69, 0.03, −0.75, −30.6, 5.50, 4.29, 4.07, 4.02, while every other
    class's were at 4.00 from the third rung on. That fits a t⁴ coefficient
    7e5 times the t² one. The extrapolant then missed by 3.9e-7. Each RDM
    alone, at full length, agreed to 2e-8, and so did all five once the
    ladder started at 1e-3 (7e-10). So the bars were right and the direction
    was too long. It is shortened (`kRdmScale`); the tolerance is unchanged.
  - Measured on the gfx1200: all 4.9e-8, B 2.6e-9, h1e_v 4.0e-10,
    e 6.5e-9, h 6.1e-10, rdm 1.1e-10 relative.
  - **"all" is a slow-tier case** (`slow_test.h`): ~115 s on the gfx1200,
    against 5–16 s for each group. The cost follows the data: one
    `class_energies` is ~0.2 s with only the orbital energies moved, ~0.7 s
    with B and ~5.7 s with everything. That is presumably einsum's atomicAdd
    per term, cheap while most terms are exact zeros, as over N₂'s
    symmetry-sparse arrays, until a dense random direction fills them in;
    the mechanism is not measured. The groups stand for it in the fast
    tier.
- `BatchedBarsMatchWhole`: walked in threes, every bar (h1e, h2e and the
  RDMs' included) matches the whole walk's to 5e-12 of its largest element.
  That is einsum's atomicAdd floor, not the batch's: two whole walks differ
  as much (dm2's bar up to 6.6e-13 on the gfx1200 and 8.9e-13 on the 3080,
  batched vs whole up to 1.06e-12), so the earlier 1e-12 failed one run in
  20-50 on each card (#640); raised by the user's decision.

The SC forward tangent, `Nevpt2EnergyTangent.DualToTheBars`, on the three
cases that carry dm3 and f3, walked whole and in threes (CAS(8,8) in threes
a slow-tier case, ~7 s on the gfx1200): one block call along six seeded
random directions (B, h1e_v, e, h, rdm alone, then all; `random_direction`'s,
an all-zero field passed as null), and each direction's Ė against
Σ ⟨bar, dx⟩ of `class_energies_adjoint`'s bars at 1e-10 of that sum's largest
summand. The bars are ladder-validated (`Nevpt2EnergyAdjointLadder`), so a
ladder of the energy adds nothing. Every case has norm-guarded, empty
perturbers, zero by N₂'s symmetry: at CAS(4,4) all of Si's and over half of
Sr's, Sijr's, Srsi's and Srs's, at CAS(8,8) 24 of Sijr's 102 and 120 of
Srsi's 459. Their bars are zero, so a nonzero tangent there would fail.
Measured on the gfx1200: worst 1.3e-12 relative.

The SC Hessian-vector product, in the same file, on CAS(8,8) (CAS(4,4)'s
real perturbers with norms down to 6e-8 put a pole of the energy within
~1e-5 of the base point along a random direction of every input, so no
ladder reaches 1e-7 there). `Nevpt2EnergyHvp.BarsMatchRichardsonLadder`:
along a seeded random direction of every input (B alone and the RDMs alone
in the slow tier), central differences of `class_energies_adjoint`'s bars at
t = 1e-3 / 2^k, k < 8, Richardson-extrapolated, the coarsest extrapolant
that moves under half a bar to the next taken (else the one that moves
least; `test/support/settled_rung.h`, #1157); every bar field of the product
to 1e-7 of its largest element. `Nevpt2EnergyHvpEmpty`: u random over every input, its
partner v = u/2 + w with w keeping N₂'s symmetry (B and h1e_v scaled entry
by entry, B by one factor per orbital pair; h1e, h2e, the RDMs and ε
random), so H ∝ N on every empty perturber both populate (Sijr's 24 and
Srsi's 120); ⟨v, HVP u⟩ against the energy's mixed second difference along
(u, v) at t = 2.5e-3 / 2^k, k < 8, twice Richardson-extrapolated (once
leaves a t⁴ term of 1.6e-6 where σ_E/t² takes over), the coarsest settled
extrapolant, to 1e-7. Then two
independent random B directions: refused, Numerical, naming the class and
the perturber. Measured on the RTX 3080 (`--preset default`): bars' ladder
worst 3.7e-9 (every input), 1.8e-8 (B), 8.2e-10 (RDMs); K6 1.2e-8 to
2.1e-8 over four runs, 9.4e-4 without K6; refused on Sijr's perturber 42 at
ρ_τ 2.3e-3 before #914, and since on Srsi's at a weighted defect of 4.7e-5
to 7.3e-5. Not run on the gfx1200.

#914's, in the same file. `Nevpt2EnergyHvpEmpty.SmallRealPerturberIsContinuousThroughTheGuard`
(slow tier, ~9 s): CAS(8,8) moved by ε = 1e-7 (every empty perturber under
the guard) and 1e-5 (one), ⟨v, HVP v⟩'s change linear in ε to 1e-7 of
the element (8e-11 measured), and the product symmetric past the guard to
1e-10. `Psb3ProportionalPairAcceptedNonProportionalRefused` (slow tier,
~50 s): PSB3 FC's t_0z, t_5z and t_11z, each with u/2 + t_0x, accepted
with over 70,000 empty perturbers populated and a defect at most 1e-6
(1.7e-14 to 3.9e-11 measured); t_0z with t_1z refused (2.3e-3).

The SC-QD reverse pass, in the same file, over `test/gtest/data/qd_nevpt2/`
with a seeded random, unsymmetric H̄:

- `Nevpt2QdEnergyAdjointCases.Eq40AndEulerFromTheBars`, all six cases: every
  element against `sc_effective_hamiltonian` at 1e-12, each class's Eq. 40
  against Σ H̄_IJ H^k_IJ and Euler's 4L at 1e-12. Measured: ≤ 2.3e-15 and
  ≤ 2.6e-15.
- `Nevpt2QdEnergyAdjoint.OneStateIsClassEnergiesAdjoint` (`h8_sto3g_sa3`,
  state 0, H̄ = 1): energies, Eq. 40 and every bar against
  `class_energies_adjoint`'s, the bars at `kBatchedBarsTolerance`.
- `Nevpt2QdEnergyAdjoint.BatchedBarsMatchWhole` (`h8_sto3g_sa3`, batch 1):
  every bar, the transitions' as handed over included, at
  `kBatchedBarsTolerance`; each ordered off-diagonal pair handed over once.
- `Nevpt2QdEnergyAdjointLadder.BarsMatchRichardsonLadder`: SC's ratio test
  and 1e-7, along every group (B, h1e_v, e, h, rdm over every pair's RDMs,
  all) on `h8_sto3g_sa3`, and along e on the other five (see Conditioning
  above). Measured on the RTX 3080: 3e-12 to 1.5e-10 relative.

The one-function classes (#949), `Nevpt2EnergyOneFunction`, in the same
file, each under half a second; both fail with the guard back in place:

- `CouplingIsLinearThroughAZeroAmplitude` (`h2co_ccpvdz_sa2`, C₂ᵥ, whose
  1A2 root's Sij amplitudes are all zero): along a random B direction, Sij's
  H_01 and H_10 at ε = 1e-8, over ε, against their central difference over
  ±1e-6, to 1e-4. Measured on the A100: 6.5e-6 and 6.7e-6. With the guard,
  H_01 is 0 at 1e-8.
- `TangentAndHessianProductMatchTheBars` (`h6_631g_sa2`'s root 0, Srs, and
  `h2co_ccpvdz_sa2`'s 1A2 root, Sij), along seeded raw directions of every
  input and of the RDMs: Ė against Σ ⟨bar, dx⟩ to 1e-10 of the largest
  summand (≤ 5.3e-14 measured), which take η's derivative by two separate
  rules, and ⟨v, HVP u⟩ = ⟨u, HVP v⟩ to 1e-8 (≤ 2.7e-14; 9.2e-3 on the
  1A2 root with the guard, whose empty Sij took K6).

The SC-QD forward tangent, `test/gtest/nevpt2_qd_tangent_gpu.cpp`'s
`Nevpt2QdTangent.DualToQdScAdjoint`, at this tree's loose DF-SA-CASSCF of
three `data/qd_nevpt2/` cases: the chain from (h, B, c) built as
`qd_sc_energy` builds it (`builders_tangent` per root, the bra/ket
`nevpt2::rdms` per transition), along seeded random columns (h, B, each
root's c alone, then all), each column's Σ W_MN Ċ_MN against
`qd_sc_adjoint`'s ⟨h̄, ḣ⟩ + ⟨Λ, Ḃ⟩ + Σ_r ⟨y_r, ċ_r⟩ at a general symmetric W
(off-diagonal, trace ≠ 1, a negative entry), relative to the largest
summand, 1e-10, or the column's conditioning floor where that is larger,
computed from the tangent's tables (Conditioning, above, #1055) and printed:
`h8_sto3g_sa3` (fast-tier representative), `h2co_ccpvdz_sa2` (C₂ᵥ, 4364
guarded tuples, walked in fours), `h2nb_sto3g_cas23_sa2` and
`h2co_skew_ccpvdz_sa3` (slow tier), whose floor passes 1e-10 in every gauge.
The floor bounds the round-off whatever the column's summands, and 1e-10 of
them only while they are as large as the tables: the orbitals' and roots'
signs are the eigensolvers' and change run to run, so the seeded MO-basis
columns and W are a new gauge each run, and in one `h2co_ccpvdz_sa2`'s h
summands summed to 1.9e-4, its miss 7.3e-14 against a floor of 5.3e-12
(#1153). Each part (the transitions' ket side, their bra
side, the diagonal's digests at the dotted (tu|vw)) is a direction of its
own and must exceed ten bars in some column. Measured on the gfx1200: ≤ 7.6e-15,
≤ 4.1e-12 and ≤ 1.1e-10 (skew, along c_2); every part ≥ 5e8 bars. Skew's
misses against its floor, on both cards, are in Conditioning above. The guarded tuples' Ṫ: on `h2co_ccpvdz_sa2` the same tuples are
empty for both roots, so it vanishes at first order; on
`h2nb_sto3g_cas23_sa2` (`data/qd_nevpt2/`, geometry only: C₂ᵥ H₂NB STO-3G
CAS(2,3), two roots of the CASCI over the RHF's orbitals, since this
tree's SA-CASSCF there falls off the symmetric saddle run to run onto a
point 1.0e-4 Eh lower with no empty tuple) 20 tuples are guarded, Srsi's
empty for one root and populated for the other, and the largest such Ṫ
must exceed ten bars (#962). Measured on the RTX 3080: 2.3e6 to 3.5e6 bars,
every column's duality ≤ 3.3e-13.

`Nevpt2QdPcTangent.DualToQdPcAdjoint`, the PC-QD forward tangent, the same
chain and columns against `qd_pc_adjoint`'s bars at the same W, 1e-10 of the
largest summand or the column's conditioning floor where that is larger
(PC-QD's Conditioning, above), SC-QD's rule, on `h2co_ccpvdz_sa2` (walked in
fours, the fast tier's representative) and `h2co_skew_ccpvdz_sa3`, whose
floor passes 1e-10 at the default cut. The floor is relative to the
tangent's elements, which the walks reassociate, and 1e-10 of the summands
is not, in any gauge: in one run `h2co_ccpvdz_sa2_floored` drew a B column
whose summands summed to 7.3e-7, its miss 1.0e-15 against 1e-10 of them
(7.3e-17) but 0.23 of its floor, 4.4e-15 (#1162). Its parts are the transitions'
ket side, their bra side, and each root's own dm1–dm3 alone (what moves
S_N), each above ten bars in some column. At the default cut every dropped
mode is null and the kept span's rotation is round-off (at most 0.36 bars),
so each case also runs at `pc_floor` 1e-3 (`_floored`), which cuts real
modes on every pair and in the adjoint alike: there the tangent without the
rotation must miss by more than ten bars. Measured on the RTX 3080: misses
≤ 6.5e-13 (sa2); skew's up to 1.6e-10 over separate runs (each run's loose
CASSCF a slightly different point), against a floor of 2.4e-9 to 7.4e-9;
floored ≤ 1.4e-14 on both; every part ≥ 3e9 bars at 1e-10; the rotation
floored 3.9e6 to 8.5e6 (sa2) and 2.0e7 to 4.1e7 (skew) bars.

The SC-QD Hessian-vector product, `test/gtest/nevpt2_qd_hvp_gpu.cpp`:

- `Nevpt2QdHvp.BarsMatchRichardsonLadder`: along one straight line in
  (h, B, c) per case and path, the product at the line's tangent of the
  builders' outputs (`qd_builders_tangent`) against central differences of
  `qd_sc_adjoint`'s builder bars (`QdBuilderBars`), t = 1e-3 / 2^k, k < 10,
  two-level Richardson, per field the coarsest extrapolant that moves under
  half a bar to the next (else the one that moves least;
  `test/support/settled_rung.h`, #1157): every
  field (shared, each state's own RDMs, each transition, Λ) to 1e-7 of its
  largest element, at a general symmetric W. Every path on `h8_sto3g_sa3`
  (B the fast-tier representative); B and c_0 on `h6_631g_sa2` (Srs a
  one-function class) and c_0, c_1 on `h2co_ccpvdz_sa2` (Sij), kQdSmooth's
  random-seed paths. Visibility, per path: the product along the line's
  orbital energies alone (Δ̇ and no fixed-multiplier difference) at W's
  off-diagonal part, Eq. qd-scfinish's off-diagonal terms, and at its
  diagonal, the 3 x 3, each above ten bars in some field. Measured on the
  RTX 3080: worst 5.2e-10 (h8), 1.1e-9 (h6), 7.9e-9 (h2co); parts ≥ 1.8e5
  bars. `h2co_skew_ccpvdz_sa3` (slow tier) along c_0 alone (#1055, #1069;
  Conditioning, above): its bars repeat only to ~1e-9, which floors a
  central difference at t near 1e-9 / t, and c_0 moves the (1, 2)
  transitions only through ε, so those fields (both orders) settle at the
  coarsest rung pair, at 1.9e-6 to 2.5e-6 in #1055's runs and 4.7e-7 to
  5.7e-7 in #1069's, and are printed beside that floor (1.4e-9 / t at the
  extrapolant's finest rung, 5.6e-6) rather than gated. Every other field
  is held to 1e-7: worst 1.4e-8 (RTX 3080), 2.5 s. Along B,
  root 2's Si tuples have N / |Ṅ| of 5e-4 to 8e-4, inside the
  ladder's steps: the coarse rungs reach the norm's near-zero, and root 2's
  fields end at 4.7e-4 to 3e-3. Nor
  `h2co_ccpvdz_sa2`'s B: it populates the empty tuples, whose bars jump as
  their norms cross the guard (Λ's extrapolants settle at 3e-5, then grow).
- `Nevpt2QdHvpEmpty.CouplingK6MatchesTheEnergysSecondDifference` (slow
  tier): that B line in the inputs, where 4364 tuples empty for both roots
  take K6 in every coupling: ⟨u, HVP u⟩ against E[W]'s second difference
  along it (s = 2.5e-3 / 2^k, k < 10, two-level Richardson, the coarsest
  settled extrapolant; the energy moves by about the guard through it) to
  1e-7, and `coupling_k6` above ten
  of those bars. Measured: 2.0e-8; the couplings' K6 −91.4 of −1963.7,
  4.7e5 bars. The roots' relative sign is pinned, dH₀₁/ds > 0 along the
  line (+1.51e-2), with root 1 negated where the SA-CASSCF lands on the
  other sign. Nothing else about its point varies run to run: C repeats to
  1e-8, each root's energy to 1e-12. A root's sign enters E[W] only through
  W's off-diagonal, so flipping it is the same as negating W₀₁ (#1071).
  Measured: flipping a root, negating W₀₁, or both, give ⟨u, HVP u⟩ values
  that agree to 1e-11 relative. At the other sign, ⟨u, HVP u⟩ is −1803.91 with K6 +91.4. There the
  ladder settles only to 1.5e-7 of it: rung-to-rung change 6.8e-7, against
  3.1e-7 at this sign. On the gfx1200 the SA-CASSCF landed on the other
  sign in 4 of 5 runs.
- `Nevpt2QdHvpEmpty.OneSidedTupleIsRefusedOnANonProportionalPair`:
  constructed inputs, `h8_sto3g_sa3`'s MO integrals over a CAS(2,3)
  partition, ket 1's CI never occupying active orbital 0 and virtual 0's B
  to the other two active orbitals zeroed, so Srsi's (r₀, r₀, i) tuples are
  empty for ket 1 and populated for bra 0 (|Ṫ| 2.9e-5). Along a random B
  direction alone, or with itself as partner, accepted; against a second
  one, refused, Numerical, naming the coupling (0, 1), Srsi and the
  directions.
- `Nevpt2QdHvpEmpty.A1B2FixtureRefusesTheExpectedBlocks` (slow tier):
  `data/qd_nevpt2_hessian/h2nb_sto3g_sa2.json` (#1058), this tree's
  SA-CASSCF, `df_sacasscf_cp` and `df_qd_nevpt2_tangents` combined into the
  fixture's C₂ᵥ coordinates; along every pair of its six vibrations, the
  other as partner, the refused blocks are exactly its SC expected ones
  (none). Measured: none refused; b1 populates 114 empty tuples (defects
  ≤ 1.3e-13), the b2 ones are Sij's, a one-function class.

The PC reverse pass, `Nevpt2PcEnergyAdjoint` and `Nevpt2PcEnergyAdjointLadder`:

- `HomogeneityAndEulerFromTheBars` (CAS(8,8)): each class's homogeneity sum
  against its energy at 1e-12, Euler's 4E as SC's, and the energies against
  block2's. Measured: 2.5e-15 and 2.2e-16.
- `BarsMatchRichardsonLadder`: SC's seeded groups, ratio test and 1e-7, kept
  inside the energy's domain. B, h1e_v and e at CAS(8,8); h, rdm and all at
  CAS(4,4), since along h at CAS(8,8) Sr's smallest denominator is −2.3e6 at
  t = 1e-2 and still −1.6e4 at 7.8e-5: a pole within about 1e-8 of the base
  point. PC's rdm moves dm1–dm3 along an active rotation of all their axes
  (which keeps every metric's spectrum; a raw direction lifts S's dropped
  modes at first order, and the energy has no derivative along it) and
  f3ca/f3ac along SC's raw direction. The ladder starts at 1e-3: at
  CAS(4,4), 1e-2 and 5e-3 carry a denominator through zero, across a pole. h1e_v has no ratios
  (the energy is quadratic in it), so each difference is held to the bars.
  Measured on the gfx1200: B 1.1e-9, h1e_v 1.4e-11, e 2.9e-10 (CAS(8,8)); h
  5.8e-10, rdm 4.7e-9, all 3.2e-10 (CAS(4,4)), relative.
- `InactiveRotationsAreRoundOff` (CAS(8,8)): with the core orbital energies
  all equal, and the virtual ones, the PC energy is invariant to rotations
  within the core and within the virtuals, so Σ <bar, dx> along each (B_ca,
  B_cv, B_va and the h1e_v's inactive axes) is held under 1e-12 of its
  terms' magnitude. Measured: 4.6e-18 of 5.4e-4 (core), 1.5e-17 of 6.4e-3
  (virtual). SC is the control and is not invariant (Park): 5.5e-7 and
  5.1e-6.
- `BatchedBarsMatchWhole` (CAS(8,8)): the walk's bars and the homogeneity
  sums at 1e-12; the sweep's bars only through those sums, for the
  conditioning above.
- `RefusesWhatTheEnergyRefuses`: zero RDMs (Sr's metric has no positive
  eigenvalue) and an intruder (every virtual orbital energy shifted to put
  Sr's strongest mode, `PcClassEnergies::strongest`, at a quarter of its
  coupling, |t| = 4): the adjoint fails with `pc_class_energies`' kind and
  message.

The PC forward tangent, in the same file, every check one block call
along six seeded random directions (B, h1e_v, e, h, rdm alone, then all;
an all-zero field null), each direction's Ė against Σ ⟨bar, ẋ⟩ from
`pc_class_energies_adjoint`:

- `Nevpt2PcEnergyTangentNullModes.DualToTheBars` (`n2_ccpvdz_cas44_df`,
  whose Sr, Si and Sir cuts drop exact null modes alone, at most 8e-17 of
  σ_max), walked whole and in threes: within 1e-10 of each sum's largest
  summand. Measured: 9.9e-14 and 1.5e-14.
- `Nevpt2PcEnergyTangent.DualToTheBarsWithRealModesDropped`, the gate: the
  same at 1e-10 on a cut that drops real modes with a gapped kept spectrum.
  The point is `h2co_skew_ccpvdz_sa3`'s (`data/qd_nevpt2/`) superposition
  |0⟩ + 3e-7 |1⟩, its RDMs from the golden own and transition RDMs (f3's
  cross term, not given, left out) times 1 − 1.25 δ²: the trace deficit
  lifts Sij's 8 fully occupied null modes to real eigenvalues up to 7.0e-14
  of σ_max (105 ε √d), below the cut, and ψ, not an eigenstate, makes K's
  part along the dropped modes of order δ. Every class keeps its smallest
  mode at 1.6e-4 or above. The test's comment has why neither a pure state
  nor an ensemble serves. Without the kept span's rotation (`span = false`)
  Ė misses by more than ten bars along some direction. Measured: 8.8e-12
  with it, 5.0e-5 without.
- `Nevpt2PcEnergyTangent.ConditioningFloorReadByEveryPass`
  (`n2_ccpvdz_cas44_df`, floor 1e-3): the floor cuts real modes the
  default keeps (26 more), the energy, reverse pass and tangent agree on
  every class's dropped count, energy and `kept`, every `kept` clears the
  floor and every `cut` sits below it, and Ė is dual to the bars at 1e-10
  (measured 7.8e-16).
- `Nevpt2PcEnergyTangent.SpanTermVisibleAtPsb3Twist` (slow tier, #917): at
  PSB3 twisted to 90°, `PcQdActiveActiveGateAtTheTwist`'s CASSCF, state 0's
  inputs from `builders_tangent`. Over ten reverse passes and ten tangents
  along h, rdm and all, σ_a and σ_t are each sample's deviation (2e-6 to
  7e-6 relative). The two means must agree within 5 √(σ_a² + σ_t²); five
  runs measured at most 1.02. Without the rotation, Ė misses the bars' mean
  by more than 100 times that bound along rdm and all (≥ 11279 spreads;
  h moves no S, so U̇ is zero there). The earlier rule, 3σ over five
  passes with the tangent a single sample, failed 2 runs in 3 after #916.
- `Nevpt2PcEnergyTangent.RefusesWhatTheEnergyRefuses` (CAS(8,8)): zero
  RDMs (no positive eigenvalue), an intruder (|t| = 4, as the adjoint's
  test places it) and a denominator at round-off (every virtual shifted to
  put Sr's nearest mode at λ + Δ = 0): the tangent fails with
  `pc_class_energies`' kind and message.

The PC Hessian-vector product, in the same file, on `n2_ccpvdz_cas44_df`
at `pc_floor` 1e-3 (real modes dropped, every kept spectrum gapped):

- `Nevpt2PcEnergyHvp.BarsMatchRichardsonLadder`, per input group (B, h1e_v,
  e, h, rdm, and all at once, the fast tier's): the product against central
  differences of `pc_class_energies_adjoint`'s bars at t = 1e-3 / 2^k,
  k < 8, the coarsest settled Richardson extrapolant (else the one that
  moves least; `test/support/settled_rung.h`), every field to 1e-7 of its
  largest element; and the product without the kept span's rotation
  missing the same ladder by more than ten bars (but along e, which moves
  no S). Measured: worst 1.9e-9; the rotation 1.9e4 to 2.8e6 bars.
- `Nevpt2PcEnergyHvp.BatchedDirectionsMatchSingle`: two directions walked
  at once against each alone, every field to 1e-10.
- `Nevpt2PcEnergyHvp.RefusesAnIllConditionedSolve` (slow tier): PSB3's
  twist (`SpanTermVisibleAtPsb3Twist`'s state) keeps Sr at 3.9e-13 of its
  largest at the default cut, refused, Numerical, naming the class and
  `pc_floor`; at `pc_floor` 1e-10 the product runs.

The PC-QD Hessian-vector product, `pc_effective_hamiltonian_hvp`, is
checked through the product it feeds (`test/gtest/nevpt2_qd_hessian_product_gpu.cpp`,
`Nevpt2QdPcHessianProduct`; `../gradient/README.md`, Tests): ladders of
`qd_pc_adjoint` on `h2co_skew_ccpvdz_sa3` at `pc_floor` 1e-3 along every
input group, and its three kinds of term each visible, the cross terms
through a call along the orbital energies alone. Its n = 1 is
`pc_class_energies_hvp`, so the one-state cases above check that
arithmetic too.

The PC-QD reverse pass, in the same file, over `test/gtest/data/qd_nevpt2/`
with a seeded random, unsymmetric H̄:

- `Nevpt2PcQdEnergyAdjointCases.HomogeneityAndEulerFromTheBars`, all six
  cases: every element and dropped count against `pc_effective_hamiltonian`
  at 1e-12, each class's homogeneity sum against Σ H̄_IJ E^k_IJ and Euler's
  4L at 1e-12. Measured: ≤ 2.3e-14 and ≤ 5.6e-15.
- `Nevpt2PcQdEnergyAdjoint.OneStateIsPcClassEnergiesAdjoint`
  (`h8_ring_sto3g_sa3`, states 0 and 2, H̄ = 1): energies, dropped counts,
  homogeneity and every bar against `pc_class_energies_adjoint`'s, the bars
  at `kBatchedBarsTolerance` (measured run to run: 6.9e-13).
- `Nevpt2PcQdEnergyAdjoint.BatchedBarsMatchWhole` (`h8_ring_sto3g_sa3`,
  batch 1): the walk's bars at `kBatchedBarsTolerance` (measured 2.5e-13),
  every element and the homogeneity sums at 1e-12; each ordered off-diagonal
  pair handed over once. The sweep's bars carry state 1's spread, as at
  n = 1.
- `Nevpt2PcQdEnergyAdjointLadder.BarsMatchRichardsonLadder`: PC's ratio
  test, t0 = 1e-3 and 1e-7, along SC-QD's random directions with every
  state's own dm1–dm3 on PC's active rotation (transitions' raw: they enter
  no solve). Every group (B, h1e_v, e, h, rdm, all) on `h2co_ccpvdz_sa2`,
  `h2co_ccpvdz_sa2_weighted` and `h6_631g_sa2`; B on the other three, and e
  on `h2co_skew_ccpvdz_sa3` and `h8_ring_sto3g_sa3`. The rest fall outside
  PC's domain within the ladder's reach (h, rdm, all on those three) or
  meet L's run-to-run round-off at the finest rungs (h1e_v there, e on
  `h8_sto3g_sa3`); the test's comment has each measurement. Measured on the
  RTX 3080, six runs: 1e-13 to 1.6e-8 relative.
- `Nevpt2PcQdEnergyAdjoint.RefusesWhatTheEnergyRefuses` (`h8_sto3g_sa3`):
  state 1's own RDMs zeroed, and state 0's Sr intruder placed the same way:
  the adjoint fails with `pc_effective_hamiltonian`'s kind and message,
  naming Sr and the state. `MalformedInputsAreInvalidConfig`: SC's checks.
