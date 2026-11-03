# `free_breakfast.nevpt2.nac`

The QD-NEVPT2 nonadiabatic (derivative) coupling d_PQ = ⟨Ψ_P|∇Ψ_Q⟩ (Park
2020, Eqs. 45–54), SC and PC, and the one piece of it the CASSCF's own
coupling does not have: the PT2 transition density. Link
`free_breakfast::nevpt2::nac`; `free_breakfast.nevpt2` re-exports it.

    qd_sc_nac(res, molecule, basis, classes, aux, h, b, l, solution, ncore, ncas, p, q, batch,
              options = {}, fold = false) -> QdNac
    qd_pc_nac(...the same...) -> QdNac
        numerator   h_PQ = R_Pᵀ (∇H_eff) R_Q, n_atom × 3 (hartree / bohr)
        cas         d^CAS = Σ_MN T^PQ_MN d_MN, the roots' total couplings
        pt2         d^PT2 = d[D^PT2], the basis motion and the orbitals' response
        csf         the CSF parts of both: d^CAS's Σ T d^CSF plus d^PT2's basis motion
        total       as mcscf::NacOptions asks; by default h_PQ / (E_Q − E_P) + cas + pt2
        heff_gap    |E_Q − E_P| (H_eff's eigenvalues; E_Q − E_P is +heff_gap for P < Q)
        root_gap    the smallest CASSCF |E_N − E_M| over the pairs with T^PQ_MN ≠ 0
        max_zeta    h_PQ's max |ζ|
        iterations, residual   the three Z-vectors': h_PQ's, d^CAS's, d^PT2's

    Under fold = true (one Z-vector, "The fold"), only total, csf, heff_gap,
    root_gap and max_zeta are valid: numerator, cas and pt2 are empty, and
    iterations and residual are the one solve's in slot 0, slots 1 and 2 zero.

    sc_pt2_transition_density(res, blocks, pairs, t, batch) -> Buffers::device<double>
    pc_pt2_transition_density(res, blocks, pairs, t, batch) -> Buffers::device<double>
        pairs   sc_effective_hamiltonian's n x n EnergyInputs, [M n + N] bra M, ket N
                (on a solution: nevpt2::qd_inputs, ../gradient/)
        t       T, n x n row-major, antisymmetric
        result  D, nmo x nmo row-major, core, active, virtual: D_xy at [x nmo + y]

The coupling's arguments are `qd_sc_coupling`'s (`../gradient/`), with
`options` `mcscf::df_nac`'s (`../../mcscf/nac/`); its result is modelled on
`mcscf::Nac`.

## The derivative coupling

With Ψ̃_P = Σ_M (|M⟩ + |Ψ⁽¹⁾_M⟩) R_MP, R H_eff's eigenvectors over the
CASSCF roots |M⟩, Park's Eq. 45 splits ⟨Ψ_P|∇Ψ_Q⟩ into

    d_PQ = h_PQ / (E_Q − E_P) + d^CAS + d^PT2
    d^NEV = h_PQ / (E_Q − E_P)                  Eq. 47, R's own derivative
    d^CAS = Σ_MN T^PQ_MN ⟨M|∇N⟩                 Eq. 48
    d^PT2 = Σ_MN T^PQ_MN ⟨Ψ⁽¹⁾_M|∇N⟩            Eq. 49
    T^PQ  = ½ (R_P R_Qᵀ − R_Q R_Pᵀ)

- **d^NEV**: R's own derivative, R_Pᵀ ∂R_Q = h_PQ / (E_Q − E_P), h_PQ =
  R_Pᵀ (∇H_eff) R_Q from `qd_sc_coupling` / `qd_pc_coupling` (#747, #748),
  over its own Z-vector. ⟨Ψ⁽¹⁾_M|N⟩ = 0, so only the zeroth-order overlap
  carries it.
- **d^CAS**: Σ_MN T_MN d_MN, the roots' couplings d_MN = ⟨M|∇N⟩ with the
  SA orbitals and CI relaxed and their CSF terms in: `mcscf::df_nac`'s
  combination (#742) at T^PQ with no D^X.
- **d^PT2**: Ψ⁽¹⁾_M lies outside the active space, so ∇ reaches it through
  |N⟩'s orbitals alone: d^PT2 = Σ_xy D^PT2_xy ⟨φ_x|∂φ_y⟩ with the orbitals
  relaxed, D^PT2 = Σ_MN T_MN ⟨Ψ⁽¹⁾_M|E_xy|N⟩ the PT2 transition density
  below: `df_nac`'s combination at T = 0 with D^X = D^PT2, its basis motion
  plus its orbitals' response.

In order, one call: the interstate coupling (whose R the rest reads), T^PQ
from that R, D^PT2 at T^PQ over `qd_inputs`, then `df_nac`'s combination
twice. `total` is

| `mult_ediff` | `use_etfs` | `total` |
|---|---|---|
| false | false | h_PQ / (E_Q − E_P) + cas + pt2 (the default) |
| false | true | h_PQ / (E_Q − E_P) + cas + pt2 − csf |
| true | false | h_PQ + (E_Q − E_P) (cas + pt2) |
| true | true | h_PQ + (E_Q − E_P) (cas + pt2 − csf) |

as `mcscf::df_nac`'s: `use_etfs` drops every CSF and basis-motion term,
d^CAS's and d^PT2's, and `numerator`, `cas`, `pt2` and `csf` are the same
whatever the options.

**Three Z-vector solves per pair, not two.** #749 counts two, h_PQ's and
one combination at (T^PQ, D^PT2), and by linearity that combination's
total is d^CAS + d^PT2. But `cas` and `pt2` are reported apart (and d^PT2
is tested alone, against #744), and one Z-vector cannot split its response
between its two right-hand sides. So the combination runs twice, at
(T^PQ, no D^X) and at (T = 0, D^PT2). The test checks their sum against the
single combination to 1e-10. Folding them, and h_PQ's solve with them
(Park's Eqs. 51–52), into one is `fold` below.

### The fold

`fold = true` solves **one** Z-vector per pair instead of three (Park
2020, Eqs. 51–52; #767). Every right-hand side is linear in its source,
and h_PQ's, d^CAS's and d^PT2's Z-vectors are over the same CI-relaxed SA
Hessian, so with ΔE = E_Q − E_P

    h_PQ + ΔE (d^CAS + d^PT2 − csf)

is h_PQ's Lagrangian with ΔE times the combination's at (T^PQ, D^PT2)
added: `qd_sc_coupling` / `qd_pc_coupling` take it through their `fold`
callback (`../gradient/README.md`, "The interstate coupling h_PQ"), which
runs once R is known and hands back

- ΔE (γ, Γ), `mcscf::nac_combination_terms`' combined RDMs at (T^PQ,
  D^PT2): added to ρ's RDMs they give the transition orbital gradient in
  the orbital right-hand side and the gate, and added to ζ's they give the
  Hellmann-Feynman term in the transition pass;
- ΔE (D^PT2 − D^PT2ᵀ), added to A: D^PT2's orbital response.

The CSF term and D^PT2's basis motion need no Z-vector, and come from the
same `nac_combination_terms` call. So the folded coupling's `total` is
h_PQ's pass over ΔE plus `csf`, as `NacOptions` asks (`mult_ediff`
multiplies it back, `use_etfs` drops `csf`).

**What it reports.** One Z-vector cannot part its response between h_PQ,
d^CAS and d^PT2, so under the fold `numerator`, `cas` and `pt2` are
**empty**. `total`, `csf`, `heff_gap`, `root_gap` (from the combination's
T^PQ, as before) and `max_zeta` are valid; `iterations` and `residual`
hold the one solve's in slot 0, and 0 in slots 1 and 2. The CASSCF
roots' gap refusal is the combination's (`mcscf::df_nac (combination)`,
naming the pair), reached before the Lagrangian's.

**Cost.** Measured on the RTX 3080 (CUDA), pair (0, 1), the median of five timed
calls of each path after one warm-up call of each, under the card lock:

| case | scheme | three solves | fold | fold / three |
|---|---|---|---|---|
| `h2co_skew_ccpvdz_sa3` | SC | 0.441 s | 0.232 s | 0.53 |
| `h2co_skew_ccpvdz_sa3` | PC | 0.467 s | 0.259 s | 0.55 |
| `h8_ring_sto3g_sa3` | SC | 0.185 s | 0.103 s | 0.56 |
| `h8_ring_sto3g_sa3` | PC | 0.237 s | 0.160 s | 0.68 |

The fold saves two Z-vectors, two `df_nac` setups (the CI-relaxed Hessian
each) and one derivative pass, and pays for two Fock builds in the gate
(below). **It is still off by default**, though it is faster: it is not a
pure performance change, since it drops `numerator`, `cas` and `pt2`, and
the one consumer, the `[nevpt2_nac]` step (`../../pipeline/nevpt2_nac/`),
writes all three. A caller that needs only `total` (a surface-hopping or
MECI driver) passes `fold = true`.

**The gate.** The interstate coupling's gate (X^L symmetric in every block,
1e-8) holds the folded Lagrangian too, but for one term: the fold's RDMs
are no state's, and their own active-active asymmetry 2 (F − Fᵀ) (the core
off) is the CSF term's, which is added outside the Z-vector and which no
multiplier removes. So under a fold the gate's source takes exactly that
term out of the active-active block (two Fock builds); the packed pairs,
which the Z-vector solved for, keep everything.

### Conventions

- **Bra P, ket Q**, as `mcscf::df_nac`: d_PQ = −d_QP, h_PQ = h_QP.
- **Signs.** R's columns are signed with their largest component positive
  (`pipeline/nevpt2`'s rule, `QdScLagrangian::eigenvectors`), and T^PQ is
  built from that same R, so h_PQ's sign and d^CAS + d^PT2's agree. d_PQ
  changes sign with the solution's CI phases as R carries them: a root
  phase s_M flips R_P by s_m(P), m(P) the index of R_P's largest
  component, so d_PQ by s_m(P) s_m(Q). The call fixes no phase.
- **Refusals.** QD states within `mcscf::kRootGapFloor` (1e-8 hartree), p = q
  or a state not one of the roots' are the interstate coupling's (#747); a
  refused PC class is `qd_pc_energy`'s error; and two CASSCF roots closer
  than `kRootGapFloor` that T^PQ couples are `Numerical`, naming the pair:
  the interstate coupling's Lagrangian refuses them first
  (`mcscf::root_pair_multipliers`, which divides by the same gap), and
  `df_nac`'s own refusal (#742) stands behind it. T^PQ is exactly
  antisymmetric in floating point, its diagonal exactly zero.

### Reproducibility, memory and the stream

Every launch is on `res.stream()`, and each piece synchronizes as its own
call does. The result is not bitwise reproducible: the GPU's atomic sums
reassociate, as in `df_nac` and the QD gradient: two calls differ by up
to ~1e-11 of the largest part summed (the swapped pair, measured below). The memory is the largest of the pieces' own, never two at once:
the interstate coupling's (its Lagrangian and derivative passes), the
density's (the effective Hamiltonian and the T-weighted RDMs, inside
`qd_inputs`' shared arrays and the n² pairs' RDMs, dm3 of n⁶ each), and
`df_nac`'s; beside them only T (n²) and D^PT2 on the host (nmo²).

## The PT2 transition density

    D^PT2_xy = Σ_MN T_MN ⟨Ψ_M⁽¹⁾|E_xy|N⟩

E_xy is spin summed, Ψ_M⁽¹⁾ state M's first-order wave function (SC or PC,
as `sc_effective_hamiltonian` and `pc_effective_hamiltonian` build it) and
|N⟩ state N's reference. It is the one quantity d^PT2 (Park Eqs. 49, 51, 53)
adds to the CASSCF coupling: ∇ acts on the reference N alone, so Ψ_M⁽¹⁾'s own
derivative never enters. The consumer takes it at T^PQ = ½(R_P R_Qᵀ −
R_Q R_Pᵀ), as the off-active density D^X of `mcscf::df_nac`'s combination
(#742, nmo² row-major, the same layout).

**Only antisymmetric T.** T is checked antisymmetric to 1e-12 of its largest
element and taken as ½(T − Tᵀ), so T_MM = 0 and only transitions M ≠ N
enter: an antisymmetric-T call sees ½(D^(MN) − D^(NM)) and never the
ordered D^(MN) = ⟨Ψ_M⁽¹⁾|E_xy|N⟩ alone. That keeps every term that reads no
RDM out (they carry ⟨M|N⟩ = 0), as in the effective Hamiltonians'
transitions.

**Three blocks.** E_xy|N⟩ is in the first-order interacting space only for
(x, y) = (t, i), (a, t) and (a, i) (i core, t active, a virtual); every other
element of D is exactly zero. Each block is one class, the class whose
perturber space holds E_xy|N⟩: Si (S_i^(1)), Sr (S_r^(−1)) and Sir
(S_ir^(0)). In it E_xy|N⟩ is the one-body function χ_y^N of V's h1e_v term,
and Ψ_M⁽¹⁾ a combination of the class's internal functions φ^M_μ on |M⟩ (V's
slab terms, the basis PC's metric S is over) and, for SC, of χ^M itself. So
with the **T-weighted transition RDMs** ρ^M = Σ_N T_MN γ^(MN) (dm1 and dm2,
bra M), each block needs only

    ⟨φ^M_μ|E_xy|ρ^M⟩ = S₁(ρ^M)[μ, y],   ⟨χ^M_a|E_xy|ρ^M⟩ = S₁₁(ρ^M)[a, y]

S₁ being the class's PC S₁ (the `PC-NEVPT2` table of `../energy/README.md`)
over ρ^M, and S₁₁ the h1e_v·h1e_v metric of SC's norm over it. No dm3, no
f3, no 4-RDM:

| class | block | S₁(ρ) | S₁₁(ρ) |
|---|---|---|---|
| Sr, over r | D[r, u] | ρ2[r′,p,q,u] at μ = (p, q, r′) | ρ1[a, u] |
| Si, over i | D[u, i] | dm2_h(ρ)[r,p,q,u], dm2_h[a,b,c,d] = 2 ρ1[a,b] δ_cd − ρ2[a,b,d,c] | −ρ1[u, a] |
| Sir, over (r, i) | D[r, i] | [2 ρ1[q,p]; −ρ1[q,p]] at μ = (x, p, q) | 0 (⟨M\|N⟩ times 2) |

Si's S₁₁ drops 2δ⟨M|N⟩ and Sir's is 2⟨M|N⟩: zero for every transition, as
in the energy's.

**SC.** Ψ_M⁽¹⁾ holds c_t P_t V|M⟩ per perturber t, P_t V|M⟩ =
Σ_μ x_tμ φ_μ + Σ_a h1_ta χ_a with x_t the slab row and h1_t h1e_v's, and
c_t = −1 / (Δ_t + H^M_t / N^M_t) (0 at a vanishing norm): `ScAmplitudes`,
handed over by `sc_effective_hamiltonian`. So

    Sr:   D[r, u] = Σ_M c^M_r (Σ_pqr′ x[r,p,q,r′] ρ2^M[r′,p,q,u] + Σ_a h1e_v_Sr[r,a] ρ1^M[a,u])
    Si:   D[u, i] = Σ_M c^M_i (Σ_pqr x[q,p,i,r] dm2_h(ρ^M)[r,p,q,u] − Σ_a h1e_v_Si[a,i] ρ1^M[u,a])
    Sir:  D[r, i] = Σ_M c^M_ri Σ_pq (2 x1[r,p,i,q] − x2[r,p,q,i]) ρ1^M[q,p]

one walk of each class's slabs (Sr; Si; Sir1 and Sir2), in batches of at most
`batch`, with the energy's own slab subscripts.

**PC.** Ψ_M⁽¹⁾ holds −Σ_k g_tk Σ_μ U_M[μ,k] φ_μ per tuple t, U_M state M's
eigenbasis of its solve (its cut modes cut) and g_tk = y_tk / (λ_k + Δ_t)
its amplitudes: `PcAmplitudes`, handed over by `pc_effective_hamiltonian`.
There is no χ part, so with W^M = S₁(ρ^M)ᵀ U_M

    D[·, t] = −Σ_M Σ_k W^M[·, k] g^M_tk

per class: two einsums per state and class, no slab.

**Refusals.** PC refuses whatever `pc_effective_hamiltonian` refuses, any
class of any state (no gap at a metric's cut, a denominator not positive),
with its error: the density of a refused PC is not defined. A T that is not
n x n or not antisymmetric is InvalidConfig.

**Cost and memory.** Each call runs its effective Hamiltonian whole (all
eight classes) for the amplitudes; SC then walks three classes' slabs once
more. Beside the effective Hamiltonian's own: n dm1, dm2 and dm2_h of ρ (n⁴
each), one class's slab buffers (SC), the handed-over amplitudes (PC: U_M,
d x m per state and class, and g, tuples x m on the host) and the three
blocks. Every launch is on res.stream(); the call synchronizes.

## Tests

### The derivative coupling

`test/gtest/nevpt2_nac_gpu.cpp`, on this tree's own DF-SA-CASSCF of each
`qd_nevpt2/` case converged to #696's bars (||g|| < 1e-10), SC and PC,
every pair, each check run once against its fixed bar. A pair's sign
against the data's phases is s_m(P) s_m(Q), read off H_eff's element
(m(P), m(Q)), ours against the data's, never off d (#747's rule).

- `Nevpt2QdNacZeroth.MatchesNumerical`, on #743's three cases
  (`data/qd_nevpt2_nac/coupling/`): d^NEV + d^CAS = h_PQ / (E_Q − E_P) +
  `cas` against `{sc,pc}_zeroth` within 2× each component's error bar,
  the three directions and, on skew H₂CO, all 12 Cartesian components;
  h_QP = h_PQ and d_QP = −d_PQ to 1e-10 of the largest element among the
  parts summed (#748's ruling: h's two derivative passes' terms; for d
  also those over the gap and the combination's three parts), both scales
  printed; `total` against its formula; **the CSF split**, `csf` against
  one `df_nac` combination at (T^PQ, D^PT2) built in the test from the
  coupling's own R and `qd_inputs`, its `csf` + `basis_motion`, to 1e-10;
  and `cas` + `pt2` against that combination's `total`, to 1e-10.
- `Nevpt2QdNacBruteForce.MatchesNumerical`, on #744's three cases
  (`data/qd_nevpt2_nac/brute_force/`, the polished state; ours converges
  to the same one): the whole d_PQ (`total`) and d^PT2 alone (`pt2`)
  within 2× each component's error bar, the three directions and, on the
  ring, all 24 Cartesian components. #744's `zeroth` is printed.
- `Nevpt2QdNacFold.MatchesThreeSolves` (#767), on every pair of #743's and
  #744's five distinct cases, SC and PC: the fold's `total` against the
  three-solve path's to **1e-9** of the largest part that path sums (h's
  summands and |h| over |ΔE|, `cas`, `pt2`, `csf`; the d_QP scale), its
  `csf` against the path's to 1e-10, and the fields it leaves empty. The
  `total` bar is 1e-9, not 1e-10, because it compares two independently
  converged Z-vector solves (each to an absolute residual of 1e-10): solver
  noise, not round-off ([#767's ruling](https://github.com/whatDiracIsCooking/FreeBreakfast/issues/767#issuecomment-6608119792)).
  Measured on the RTX 3080: `total` at most 1.05e-10 relative (SC
  `h2co_skew_ccpvdz_sa3_weighted` (0, 1); every other pair ≤ 9.9e-11),
  `csf` at most 7.7e-12.
- `Nevpt2QdNac.TotalFollowsTheOptions`: `total` against its formula at all
  four `NacOptions` settings, SC and PC, (0, 1) and (1, 0) of
  `h6_631g_sa2`, to 1e-10 of the largest part summed.
- `Nevpt2QdNac.Refuses` (`h2co_ccpvdz_sa2`): QD states made degenerate are
  the interstate coupling's `Numerical`, naming both; two CASSCF roots
  made equal (T^PQ couples them) are `Numerical`, naming the pair.

Measured on the RTX 3080 (CUDA), worst |analytic − reference| / bar:

| case | family | SC | PC |
|---|---|---|---|
| `h8_ring_sto3g_sa3` | #743 d^NEV + d^CAS | 0.703, (1, 2) u1 | 0.140, (1, 2) u0 |
| `h2co_skew_ccpvdz_sa3` | #743 d^NEV + d^CAS (directions + 12 Cartesian) | 0.279, (0, 2) H3 z | 0.150, (1, 2) C0 y |
| `h2co_skew_ccpvdz_sa3_weighted` | #743 d^NEV + d^CAS | 0.352, (1, 2) u0 | 0.357, (1, 2) u0 |
| `h6_631g_sa2` | #744 whole d / d^PT2 | 0.108 / 0.218 | 0.319 / 0.179 |
| `h8_sto3g_sa3` | #744 whole d / d^PT2 | 1.504 / 0.094 | 1.016 / 0.108 |
| `h8_ring_sto3g_sa3` | #744 whole d / d^PT2 (directions + 24 Cartesian) | 0.906 / 0.028 | 0.492 / 0.055 |

The bar is 2. Path agreements, worst over every pair of #743's cases,
relative to their scales: h_QP = h_PQ 5.6e-12 (SC) and 8.1e-12 (PC) of
the largest summand (against max |h| PC reaches 3.3e-10, which is why
#748 ruled the summand scale); d_QP = −d_PQ 1.1e-11 and 8.1e-12 of the
largest part; the CSF split 2.6e-12 and 8.0e-12; cas + pt2 against the
one combination 6.4e-11 and 4.7e-11 of max |d^CAS + d^PT2|; `total`
against its formula 0 at every option setting. Each case takes 1–9 s, so
none is behind `FBF_SKIP_IF_FAST_TESTS`.

**Visibility.** #743's and #744's READMEs tabulate every separately coded
term of every pair clearing 10× its bar on every case (by at least 1.5e2
and 5e5), so no new case was needed.

### The PT2 transition density

`test/gtest/nevpt2_pt2_density_gpu.cpp`, against #744's brute force
(`test/gtest/data/qd_nevpt2_nac/brute_force/`), which is written on
`../qd_nevpt2/`'s own state: the density is evaluated from that fixture's
integrals and golden transition RDMs, so both sides start from the same
input.

- `Nevpt2Pt2Density.MatchesBruteForce`, on `h6_631g_sa2`, `h8_sto3g_sa3` and
  `h8_ring_sto3g_sa3`, SC and PC: every pair M < N through T = ½(E_MN −
  E_NM) against ½(D^(MN) − D^(NM)), a seeded random antisymmetric T against
  Σ T_MN D^(MN), and that T again in batches of two, each to 1e-10 of the
  compared density's largest element; every element outside the three
  blocks exactly zero. Measured on the gfx1200: at most 1.3e-13 (SC) and
  4.4e-12 (PC) of max |D|, but for one case and scheme:
  - **`h8_ring_sto3g_sa3`'s SC is held to 1e-8, its conditioning bound**
    ([#746's ruling](https://github.com/whatDiracIsCooking/FreeBreakfast/issues/746#issuecomment-6600539219)).
    Its Si class's own norms are N = 6e-9 to 1.5e-8, where every other
    class and case has 1e-3 to 1e-1, so c = −1/(Δ + H/N) amplifies the
    Rayleigh quotient's round-off by ~1/N. c is off by ~2e-8 relative,
    partly systematically, and the (t,i) element it scales misses by
    1.4–2.3e-10 of max |D| (1.0–1.6e-12 absolute). The energy cannot see it
    (c N ~ 1e-8), and the SC walk is unchanged. PC, with no such division,
    passes there at 1.8e-11.
- `Nevpt2Pt2Density.AmplitudeOutputLeavesHeffUnchanged`: the effective
  Hamiltonians with the hand-over against the same calls without it, to
  1e-10, PC's dropped counts equal, and the amplitudes' shapes.
- `Nevpt2Pt2DensityRefusals`: h8_sto3g_sa3 with state 1's own RDMs zero
  refuses PC's density with `pc_effective_hamiltonian`'s error (SC runs);
  malformed T is InvalidConfig.
