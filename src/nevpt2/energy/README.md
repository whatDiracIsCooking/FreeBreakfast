# `free_breakfast.nevpt2.energy`

The eight strongly contracted NEVPT2 class energies of one state on the GPU,
and the eight partially contracted ones ([PC-NEVPT2](#pc-nevpt2)), each class
walking its external index in batches over the density-fitted slabs. Link
`free_breakfast::nevpt2::energy`.

    class_energies(res, blocks, in, batch) -> ClassEnergies
        norms[8], energies[8]   Sr, Si, Sijrs, Sijr, Srsi, Srs, Sij, Sir
        total                   Σ energies, the correlation energy

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
  perturbers whose norm does not vanish. Sijrs has no guard, as in PySCF.

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

## Memory

The intermediates live across `nevpt2::slab`, which takes
`res.device_workspace()` for its gemm product, so they and the slab buffers
are `Buffers::device` of their own, freed when their class returns. The
largest are the n_act⁶ ones (a16, a22, dm3_h, hdm3, rm3): at most two at once.
`fdm2` and each slab's norm and H vectors are workspace regions, taken after
the slab is built and dead before the next one is.

## PC-NEVPT2

    pc_class_energies(res, blocks, in, batch) -> PcClassEnergies
        energies[8], dropped[8]   the same eight classes, and each metric's cut
        total                     Σ energies

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
or below 1e-13 of its largest eigenvalue are dropped: two decades above the
structural null modes' noise, two below the first real mode on cc-pVTZ. A
class is a **Numerical error, naming it**, when S has no positive
eigenvalue, when its smallest kept eigenvalue is under 1e3 times its largest
dropped one (the rank, and so the energy, would depend on where the cut
fell: the demo's N₂ CAS(12,12) is such a state), or when a denominator
λ_k + Δ_t is not positive. The demo reported a refused class and carried on;
here the call fails, since a total without that class is not PC-NEVPT2
(the pipeline step then drops that root's PC and keeps its SC).

**Memory.** The solve's d × d matrices (S, K, their symmetrized copies, V,
and the d × m X, U, T and gemmwich's work) are buffers of their own, freed
when the class returns; at n = 12, d = n³ is 24 MB per matrix, below the n⁶
intermediates. The slab's y is a workspace region, taken after the slab.

## The SC reverse pass

    class_energies_adjoint(res, blocks, in, batch, bars) -> ClassEnergiesAdjoint
        energies     class_energies' norms, energies and total
        eq40[8]      Park Eq. 40, Σ_t (P_t N_t + Q_t H_t), from the bars
    bars (EnergyBars, each added onto, a null one not formed):
        h1e, h2e, dm1, dm2, dm3, f3ca, f3ac             in their inputs' layouts
        h1e_v_sr, h1e_v_si, h1e_v_sir, e_core, e_virt   in their inputs' layouts
        lambda                                          Λ, nevpt2::slab_adjoint's

`reverse.cpp`, `nuclear_gradient.tex` §5.2 (demo `docs/gradients.md`
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
`class_bars` returns them. A metric a class reads straight from an RDM
(Sr's dm1–dm3, Srsi's dm1, Sir's dm1 and dm2) is a block of that class there,
so each class's Eq. 40 is assembled from its own bars:
`eq40[k] = Σ_G ⟨M_G, G⟩`, plus ⟨P, term⟩ for Sir's one term that reads no
block, and ½ ⟨ḡ, g⟩ for Sijrs.

### The sweep

`sweep_blocks` takes every M_G back through the `make_*` that built G, onto
the `h1e`, `h2e` and RDM fields of `EnergyBars` (skipped when all seven are
null). Each builder is listed term by term with energy.cpp's subscripts,
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
add, so `BlockBars` is spent by the sweep. Si's dm1_h is `make_hdm1`, so its
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

## The PC reverse pass

    pc_class_energies_adjoint(res, blocks, in, batch, bars) -> PcClassEnergiesAdjoint
        energies         pc_class_energies' energies, dropped counts and total
        homogeneity[8]   Σ_G <M_G, G> per class, from the bars
    bars: EnergyBars, as class_energies_adjoint's

Also `reverse.cpp`, after SC's: the solve's adjoint (demo `docs/gradients.md`
§4.3, §5.6 and §7; `nuclear_gradients.tex` Eqs. pcdot and pcadj). A tuple's
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
rank locally constant, which the energy's gap check enforces, and, for the
bars on S, K annihilating S's dropped modes, which holds at an eigenstate. A
class the energy refuses is refused here with the energy's error (the same
`solve_class`, and one `pc_denominators_positive` for both), never given a
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
holds them, the blocks G kept for the homogeneity sum, U beside T, and m × m
M, M_Δ and the d × m T̄; after the walk two d × m and four d × d matrices.
Each slab's y, x, ȳ, x̄ and weights are buffers of their own.

## Not ported

The forward tangents and the Hessians, SC's and PC's. The demo's
`energiesDevice` over whole blocks is `batch <= 0` here.

## Tests

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
naming Sr.

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
  RDMs' included) matches the whole walk's to 1e-12 of its largest element.

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
  CAS(4,4), 1e-2 and 5e-3 make a denominator negative. h1e_v has no ratios
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
  eigenvalue) and the virtual orbital energies at −1000 Eh (Sr's
  denominator): the adjoint fails with `pc_class_energies`' kind and message.
