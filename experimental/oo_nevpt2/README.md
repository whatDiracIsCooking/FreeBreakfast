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
back most of what E_PT2 gains. Not tested: a state-averaged reference whose
CASSCF orbitals are qualitatively wrong, which is the case the idea was for.

## Run

```sh
cmake --preset experimental
cmake --build build-experimental --target free_breakfast_oo_nevpt2
build-experimental/experimental/oo_nevpt2/free_breakfast_oo_nevpt2 \
  experimental/oo_nevpt2/water.toml 10        # quasi-Newton steps
build-experimental/experimental/oo_nevpt2/free_breakfast_oo_nevpt2 \
  experimental/oo_nevpt2/water.toml 6 no      # natural-orbital passes
```

The config needs a `[casscf]` of one singlet root over density-fitted
`[eri]`; the second argument is the iteration count (8 by default), the
third `oo` (the default) or `no`. In `oo` mode, one row
per iteration: both energies, ‖g‖, ‖pack(A)‖, ‖g + pack(A)‖, the energy
change at t = ½ and t = 1 along the step, and the active natural
occupations. Not a ctest entry.

## Files

| File | Role |
|------|------|
| `design.html` | The design note and the results |
| `main.cpp` | The probe: CASCI, PC-NEVPT2 energy and adjoint, Z-vector step and line search; the natural-orbital pass |
| `CMakeLists.txt` | `free_breakfast_oo_nevpt2` |
| `water.toml`, `water.xyz` | CASSCF(4,4) water, cc-pVDZ |
| `lif_1.6.toml`, `lif_5.0.toml` and their `.xyz` | LiF CASSCF(6,4) by AVAS over F 2p and Li 2s, aug-cc-pVDZ |
