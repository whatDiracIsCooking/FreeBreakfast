# `examples/`

Small configs for the `free-breakfast` binary, one per step family, each
over a molecule of a few atoms in a small basis, so every one runs in
seconds on one card. Run one from the root of the tree, after a build:

```sh
build-hip/src/free-breakfast -i examples -o runs/rhf examples/rhf.toml
```

`-i examples` is where the configs' `xyz` paths resolve; every file the run
writes lands in `-o`, named after its step. The full grammar is
[`src/pipeline/config/README.md`](../src/pipeline/config/README.md), each
step's own options its README under [`src/pipeline/`](../src/pipeline/README.md).

| File | Shows | Writes |
|------|-------|--------|
| [`rhf.toml`](rhf.toml) | The smallest config: RHF water in STO-3G over integral-direct ERIs, every other table filled in | `scf.molden` |
| [`memory.toml`](memory.toml) | `rhf.toml` under `[memory]` caps: the plan held to `max_device_mem_per_gpu` and `max_pinned_mem_per_gpu` before the first step, and run inside them (`src/pipeline/README.md`, "Memory caps") | `scf.molden` |
| [`casscf.toml`](casscf.toml) | CASSCF(4,4) water in cc-pVDZ over density-fitted ERIs (cc-pVDZ-JKFIT), the frontier orbitals active | `scf.molden`, `casscf.molden` |
| [`nevpt2.toml`](nevpt2.toml) | `[nevpt2]`: SC- and PC-NEVPT2 over that CASSCF | `scf.molden`, `casscf.molden` (the energies are printed) |
| [`qd_nevpt2.toml`](qd_nevpt2.toml) | `[nevpt2] quasi_degenerate = true`: SC- and PC-QD-NEVPT2 over a CASSCF of two roots | `scf.molden`, `casscf.molden` (the states are printed) |
| [`nac.toml`](nac.toml) | `[nac]`: the coupling between the two roots of a DF-SA-CASSCF(4,3) of formaldehyde, `src/pipeline/nac/README.md`'s | `scf.molden`, `casscf.molden`, `nac.json` |
| [`qd_nevpt2_nac.toml`](qd_nevpt2_nac.toml) | `[nevpt2_nac]`: the SC-QD-NEVPT2 derivative coupling between the two QD states of `nac.toml`'s formaldehyde | `scf.molden`, `casscf.molden`, `nevpt2_nac.json` |
| [`meci.toml`](meci.toml) | `[optimize] meci = [0, 1]`: the S0/S1 minimum-energy conical intersection of ethylene's DF-SA2-CASSCF(2,2), `src/pipeline/optimize/README.md`'s | `optimize.xyz`, `optimize.json`, `final.xyz`, `scf.molden`, `casscf.molden` |
| [`qd_meci.toml`](qd_meci.toml) | `[optimize] meci = [0, 1]` beside `[nevpt2_gradient] quasi_degenerate = true`: the MECI of the two lowest SC-QD-NEVPT2 states over ethylene's DF-SA3-CASSCF(2,2), `src/pipeline/optimize/README.md`'s | `optimize.xyz`, `optimize.json`, `final.xyz`, `scf.molden`, `casscf.molden` |
| [`hessian.toml`](hessian.toml) | `[hessian]`: the analytic DF-RHF Hessian of water, its harmonic frequencies and IR intensities (logged) | `hessian.json` alone: its SCF is a nested run's |
| [`nevpt2_hessian.toml`](nevpt2_hessian.toml) | `[hessian]` beside `[nevpt2_gradient]`: the analytic SC-NEVPT2 Hessian of a skew (C1) formaldehyde, #803's `h2co_skew_sto3g` fixture's, over its single-state DF-CASSCF(4,4) in STO-3G, its harmonic frequencies (logged) | `scf.molden`, `casscf.molden`, `nevpt2_gradient.json`, `hessian.json` |
| [`qd_nevpt2_hessian.toml`](qd_nevpt2_hessian.toml) | `[hessian]` beside `[nevpt2_gradient] quasi_degenerate = true`: the analytic SC-QD-NEVPT2 Hessian of QD state 1 of that skew formaldehyde over its DF-SA2-CASSCF(4,4), its harmonic frequencies (logged) and `hessian.json`'s `qd` report | `scf.molden`, `casscf.molden`, `nevpt2_gradient.json`, `hessian.json` |
| [`water.xyz`](water.xyz) | The water every water config reads (Angstrom) | |
| [`ethylene.xyz`](ethylene.xyz) | `meci.toml`'s and `qd_meci.toml`'s ethylene, twisted 80° and pyramidalized (Angstrom) | |
| [`h2co_skew.xyz`](h2co_skew.xyz) | `nevpt2_hessian.toml`'s and `qd_nevpt2_hessian.toml`'s skew formaldehyde (Angstrom) | |
| [`h2co.xyz`](h2co.xyz) | `nac.toml`'s and `qd_nevpt2_nac.toml`'s formaldehyde (Angstrom) | |

## Tested

`test/examples/` runs `rhf.toml`, `memory.toml` and `hessian.toml` through
the binary as the ctest entries `example_rhf`, `example_memory` and
`example_hessian` (label `gpu`), in the fast tier: each checks the exit
status, the closing `completed: 0 errors` line, the files in the table above,
`memory.toml`'s pre-flight lines, and `hessian.json`'s keys. A config
that stops parsing fails its entry on any machine, since `main` plans the
config before it looks for a device; with no device the entry skips. The
other nine are run by hand; `nevpt2_hessian.toml` and
`qd_nevpt2_hessian.toml` are also parsed by
`test/pipeline/hessian_host.cpp`. A config added here goes in the table
above, and in `test/examples/CMakeLists.txt` only if it is as cheap as
those three.
