# `free_breakfast.nevpt2.nac`

The pieces of the QD-NEVPT2 nonadiabatic coupling d_PQ = ⟨Ψ_P|∇Ψ_Q⟩ (Park
2020, Eqs. 45–54) that the CASSCF's own coupling does not have. For now one:
the PT2 transition density, SC and PC. Link `free_breakfast::nevpt2::nac`;
`free_breakfast.nevpt2` re-exports it.

    sc_pt2_transition_density(res, blocks, pairs, t, batch) -> Buffers::device<double>
    pc_pt2_transition_density(res, blocks, pairs, t, batch) -> Buffers::device<double>
        pairs   sc_effective_hamiltonian's n x n EnergyInputs, [M n + N] bra M, ket N
        t       T, n x n row-major, antisymmetric
        result  D, nmo x nmo row-major, core, active, virtual: D_xy at [x nmo + y]

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
