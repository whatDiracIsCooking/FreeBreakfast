# `experimental/oo_nevpt2/`

Orbital-optimized NEVPT2: make E_CAS + E_PT2 stationary in the orbital
rotations, the CI a CASCI at every set of orbitals. An idea and one probe.
[`design.html`](design.html) has the functional, why its gradient is already
in `src/nevpt2/gradient/` (the NEVPT2 nuclear gradient's Z-vector is its
quasi-Newton step), the risks and the measurements.

**Verdict: the bare functional has no usable minimum.** On water (4,4) and
LiF (6,4) at 1.6 and 5.0 Å the total falls by 1 to 7 mEh in the first step
and then keeps falling: E_PT2 runs away while E_CAS rises by hartrees, the
gradient grows, and only PC-NEVPT2's intruder guard stops the run. It would
need regularization (as κ-OOMP2) before it is a method.

**And there is little to gain.** The probe's `no` mode takes the natural
orbitals of the PT2-corrected density instead, minimizing nothing: the first
pass lowers the total by 0.06 to 0.5 mEh on the same cases, E_CAS giving
back most of what E_PT2 gains.

**The state average, the case the idea was for, says the same.** The
probe's `sa` mode takes those passes to LiF's ionic and covalent ¹Σ⁺ pair
(SA-CASSCF(2,2), PC-QD-NEVPT2). The orbitals do move, and one pass shifts
the crossing by 0.46 Å; but the passes settle, and at their fixed point the
crossing is at 6.28 Å against 6.34 Å in the CASSCF orbitals. Both states
rise by 4 to 5 mEh, together: 0.6 mEh between them. Nothing judges which
orbitals are right; the answer barely depends on them.

**A bounded replacement does not rescue it.** The probe's `mo` mode
minimizes W = ⟨Ψ⁽¹⁾|Ψ⁽¹⁾⟩, the first-order wave function's weight
(Brueckner's maximum overlap at first order), which has a minimum and no
parameter. On water in 6-31G its orbitals raise the total: by 0.7 mEh with
the O–H bonds stretched, and by 131 mEh at equilibrium, where a correlating
pair is rotated out of the active space.

**Against FCI the orbitals are not the error.** On the same water cases
PC-NEVPT2 is 6.4 and 8.8 mEh above FCI at the CASSCF orbitals, and 6.2 and
8.3 mEh above it in the FCI state's own natural orbitals. The bare
functional passes through the FCI energy after three to five steps and
keeps falling.

**Capping W makes it well posed, and the cap decides the answer.** The
probe's `cap` mode minimizes E under W ≤ f · W(CASSCF orbitals), which is
bounded below. On the same water cases it converges, stays within 0.04 rad
of the CASSCF orbitals and at f = 1 lowers the total by 0.19 and 0.16 mEh,
towards FCI. The constraint is active at every solution: f = 1.03 buys
1 mEh and f = 1.10 buys 2 to 3 mEh, and nothing in the functional picks f.

## Run

```sh
cmake --preset experimental
cmake --build build-experimental --target free_breakfast_oo_nevpt2
build-experimental/experimental/oo_nevpt2/free_breakfast_oo_nevpt2 \
  experimental/oo_nevpt2/water.toml 10        # quasi-Newton steps
build-experimental/experimental/oo_nevpt2/free_breakfast_oo_nevpt2 \
  experimental/oo_nevpt2/water.toml 6 no      # natural-orbital passes
build-experimental/experimental/oo_nevpt2/free_breakfast_oo_nevpt2 \
  experimental/oo_nevpt2/water_fci.toml 150 mo  # minimize W, with FCI
build-experimental/experimental/oo_nevpt2/free_breakfast_oo_nevpt2 \
  experimental/oo_nevpt2/water_fci.toml 100 cap 1.03  # E under W <= 1.03 W0
build-experimental/experimental/oo_nevpt2/free_breakfast_oo_nevpt2 \
  experimental/oo_nevpt2/lif_sa.toml 12 sa      # passes over a state average
```

The config needs a `[casscf]` of one singlet root over density-fitted
`[eri]` (of two or more singlet roots with `sa`); the second argument is
the iteration count (8 by default), the third `oo` (the default), `no`,
`mo`, `cap` or `sa`, and after `cap` the factor f (1 by default). `sa`
prints one row per pass: the CASCI roots' energies, H_eff's upper triangle,
the QD states' energies and each one's weight on root 0. Every other mode
first prints W at the
CASSCF orbitals and its check; with an `[fci]` table in the config, also the
FCI energy and the functional in the FCI natural orbitals. `mo` and `cap`
difference over every rotation, so they are for small cases: 54 rotations
and a minute or two on 6-31G water. In `oo` mode, one row
per iteration: both energies, ‖g‖, ‖pack(A)‖, ‖g + pack(A)‖, the energy
change at t = ½ and t = 1 along the step, and the active natural
occupations. Not a ctest entry.

## Files

| File | Role |
|------|------|
| `design.html` | The design note and the results |
| `main.cpp` | The probe: CASCI, PC-NEVPT2 energy and adjoint, Z-vector step and line search; the natural-orbital pass, and a state average's; W, its minimization, and E's under a cap on it |
| `CMakeLists.txt` | `free_breakfast_oo_nevpt2` |
| `water.toml`, `water.xyz` | CASSCF(4,4) water, cc-pVDZ |
| `water_fci.toml` | The same water in 6-31G with `[fci]`: the exact answer |
| `water_stretched_fci.toml`, `water_stretched.xyz` | That with both O–H bonds at 1.8 times their length |
| `lif_1.6.toml`, `lif_5.0.toml` and their `.xyz` | LiF CASSCF(6,4) by AVAS over F 2p and Li 2s, aug-cc-pVDZ |
| `lif_sa.toml`, `lif_sa.xyz` | LiF SA-CASSCF(2,2) of the ionic and covalent ¹Σ⁺ states over F 2pσ and Li 2s, at 6.25 Å; the scan is this with R replaced |
