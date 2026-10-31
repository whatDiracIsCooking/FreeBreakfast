# `bench/`

Wall-clock benchmarks that drive the real modules.

```sh
cmake --preset default -B build-bench -DFREE_BREAKFAST_BUILD_BENCH=ON
cmake --build build-bench
```

Opt-in, and with no preset of its own: the option rides on whichever preset
already matches the machine, into a build directory of its own so it never
reconfigures `build/`.

**A benchmark is not a test.** These are plain executables, not ctest entries —
a timing threshold that fails on a busy machine is noise, and a benchmark that
cries wolf gets ignored. Read the numbers; do not gate on them.

**Bound the container or the number is meaningless.** Two containers sharing a
box contend, and nothing in the output says so. Pin disjoint cores with
`CPUSET` (see `devtools/config.sh` and `.devcontainer/README.md`) before
comparing one run to another.

## Benchmarks

| Executable | Times |
|---|---|
| `free_breakfast_ci_sigma_precision CASES.json [n_roots] [max_iterations] [device] [shift] [pspace_size] [olsen]` | `CiSigma::apply` against `apply_fp32` per column, in the full and the symmetric layout with the device workspace each takes, and `ci_solve` FP64-only against its two-stage solve at several `fp32_tolerance`s, in the symmetric layout and held to the full one, each with `spin_averaged_diagonal` off and on (`shift` 1: the M_s = 1 triplets, full layout only; `pspace_size` and `olsen` set those options, `ci_solve`'s defaults when left out), over the active spaces `tools/emit/ci_bench_integrals.py` writes. `src/ci/solver/README.md` has its numbers. |
| `free_breakfast_ci_sigma_blocked CASES.json columns LIMIT_MIB [device]` | `CiSigma::apply` and `apply_fp32` per column, unbounded against blocked under `set_workspace_limit(LIMIT_MIB)`, in both layouts, with the workspace each takes. `../src/ci/sigma/README.md` has its numbers. |
| `free_breakfast_ci_sigma_blocked CASES.json solve LIMIT_MIB [n_roots] [device]` | `ci_solve`'s singlets under that limit (0 leaves it unbounded), for a space like CAS(14,14) whose G is past the card. |
| `free_breakfast_cis_sigma_df [reps]` | `cis::DfSigma::apply` against the integral-direct `cis::Sigma::apply` per block of k = 1, 4 and 8 columns, over H₂O / cc-pVDZ's RHF (device 0) and a cc-pVDZ-JKFIT fit. `../src/cis/sigma/README.md` has its numbers. |
| `free_breakfast_cphf_batched [basis] [reps] [u_out] [u_reference]` | `response::cphf` on 3 n_atom random right-hand sides over decane (32 atoms) in a DF-RHF over `basis` (cc-pVDZ) and cc-pVDZ-JKFIT, the best of `reps` solves; writes U to `u_out` and compares it with `u_reference`, so two builds of the solver can be timed and checked against each other. `../src/response/cphf/README.md` has its numbers. |
