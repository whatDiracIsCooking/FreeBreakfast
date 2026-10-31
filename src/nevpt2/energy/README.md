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
    bars (EnergyBars, each added onto):
        h1e_v_sr, h1e_v_si, h1e_v_sir, e_core, e_virt   in their inputs' layouts
        lambda                                          Λ, nevpt2::slab_adjoint's

`reverse.cpp`, part A of `nuclear_gradient.tex` §5.2 (demo `docs/gradients.md`
§5.1–5.2): from the energy back to the class-level inputs, at fixed h1e, h2e
and RDMs. Each class walks its slabs as `class_energies` does, and lists its
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
block, and ½ ⟨ḡ, g⟩ for Sijrs. Nothing reads them past that yet;
`class_energies_adjoint` drops them.

**What #573 picks up.** The reverse-topological sweep of `BlockBars` back
through the `make_*` builders onto dm1, dm2, dm3, f3ca, f3ac, h1e and h2e.
Each builder's einsums and transposes go through `einsum_adjoint` as they
are (a transpose is the one-operand einsum, its axes spelled as labels), with
the bar on each intermediate (hdm2 under hdm3, rm2 under rm3, fdm2, a17's and
a19's h1e − Σ(mjjn)) taken before its own producer's. The diagonal slices
(`diag_a16`, `diag_a22a`, `diag_a22b`) need their gathers there. Then come
the `h1e`, `h2e` and RDM fields of `EnergyBars`, where `class_energies_adjoint`
now drops the bars. The workspace rule there is `einsum_adjoint`'s: hold no
region of it across the call.

**Memory.** On top of `class_energies`' own, the block bars are buffers of
their own. They outlive their class, four of them n⁶ (Sr's a16 and dm3, Si's
a22 and dm3_h). Each slab's bar is one more slab-sized buffer, and the norm,
H and multiplier tables per slab are buffers too, since they are held across
`einsum_adjoint`'s and `slab_adjoint`'s workspace.

## Not ported

The forward tangents and the Hessians, SC's and PC's. The SC reverse pass's
sweep onto the RDMs, h1e and h2e (#573), and PC's reverse pass. The demo's
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
- `BarsMatchRichardsonLadder`: Σ ⟨bar, ẋ⟩ over (B via Λ, h1e_v_*, e_core,
  e_virt), along a seeded random direction (all four together, then each
  group alone), against central differences of `class_energies` itself. Successive changes must fall by a factor in
  (3.2, 4.8) over at least three halvings, and the Richardson extrapolant at
  that run's finest pair must agree to 1e-7 relative. On an RTX 3080 it agrees
  to between 2e-10 and 5e-9.
- `BatchedBarsMatchWhole`: walked in threes, every bar matches the whole
  walk's to 1e-12 of its largest element.
