# ---------------------------------------------------------------------------
# bench/ -- wall-clock benchmarks over the real modules
# ---------------------------------------------------------------------------
# Configured only under -DFREE_BREAKFAST_BUILD_BENCH=ON, so a benchmark's build
# cost never lands on a normal configure. There is no `bench` preset: add the
# option on top of one that already matches the machine, e.g.
#
#   cmake --preset default -B build-bench -DFREE_BREAKFAST_BUILD_BENCH=ON
#
# A benchmark is an ordinary executable linking free_breakfast::* -- it is not a
# ctest entry, because a timing that fails a threshold on a busy machine is
# noise rather than a regression. Bound the container first (CPUSET/CPUS in
# devtools/config.sh) or the number means nothing; see .devcontainer/README.md.

# CiSigma's apply against apply_fp32, and ci_solve with and without its
# single-precision stage, over tools/emit/ci_bench_integrals.py's active
# spaces.
add_executable(free_breakfast_ci_sigma_precision ci_sigma_precision.cpp)
set_target_properties(
  free_breakfast_ci_sigma_precision PROPERTIES CXX_MODULE_STD 1
                                               CXX_SCAN_FOR_MODULES ON
)
target_link_libraries(
  free_breakfast_ci_sigma_precision
  PRIVATE free_breakfast::core free_breakfast::core::resources
          free_breakfast::io free_breakfast::ci
)

# CiSigma's apply unbounded against blocked under a workspace limit, per
# column, and ci_solve under one, for active spaces whose G is past the card.
add_executable(free_breakfast_ci_sigma_blocked ci_sigma_blocked.cpp)
set_target_properties(
  free_breakfast_ci_sigma_blocked PROPERTIES CXX_MODULE_STD 1
                                             CXX_SCAN_FOR_MODULES ON
)
target_link_libraries(
  free_breakfast_ci_sigma_blocked
  PRIVATE free_breakfast::core free_breakfast::core::resources
          free_breakfast::io free_breakfast::ci
)

# cis::DfSigma's apply against the integral-direct cis::Sigma's, per block of
# k = 1, 4 and 8 columns, over H2O / cc-pVDZ's RHF and a cc-pVDZ-JKFIT fit.
add_executable(free_breakfast_cis_sigma_df cis_sigma_df.cpp)
set_target_properties(
  free_breakfast_cis_sigma_df PROPERTIES CXX_MODULE_STD 1
                                         CXX_SCAN_FOR_MODULES ON
)
target_link_libraries(
  free_breakfast_cis_sigma_df
  PRIVATE free_breakfast::core
          free_breakfast::core::resources
          free_breakfast::cis::precondition
          free_breakfast::cis::sigma
          free_breakfast::pipeline::step
          free_breakfast::pipeline::eri
          free_breakfast::pipeline::scf
          free_breakfast::pipeline::config
)

# response::cphf on 3 n_atom random right-hand sides over decane's DF-RHF, its
# solution written for a second build's run to compare against.
add_executable(free_breakfast_cphf_batched cphf_batched.cpp)
set_target_properties(
  free_breakfast_cphf_batched PROPERTIES CXX_MODULE_STD 1
                                         CXX_SCAN_FOR_MODULES ON
)
target_link_libraries(
  free_breakfast_cphf_batched
  PRIVATE free_breakfast::core
          free_breakfast::core::resources
          free_breakfast::response
          free_breakfast::pipeline::step
          free_breakfast::pipeline::eri
          free_breakfast::pipeline::scf
          free_breakfast::pipeline::config
)

# mcscf::zvectors on 3 n_atom right-hand sides by each CpSolver over the same
# H~, water or octatetraene in a DF-CASSCF(4,4), against PerColumn.
add_executable(free_breakfast_mcscf_zvectors mcscf_zvectors.cpp)
set_target_properties(
  free_breakfast_mcscf_zvectors PROPERTIES CXX_MODULE_STD 1
                                           CXX_SCAN_FOR_MODULES ON
)
target_link_libraries(
  free_breakfast_mcscf_zvectors
  PRIVATE free_breakfast::core
          free_breakfast::core::resources
          free_breakfast::integrals
          free_breakfast::ci
          free_breakfast::response
          free_breakfast::mcscf
          free_breakfast::mcscf::relaxed_hessian
          free_breakfast::mcscf::zvector
          free_breakfast::pipeline::step
          free_breakfast::pipeline::eri
          free_breakfast::pipeline::scf
          free_breakfast::pipeline::casci
          free_breakfast::pipeline::config
)

# integrals.eri.density_fitting.jk's df_exchange_symmetrized against #595's
# block two-factor df_exchange of the symmetrized density, over random B.
add_executable(
  free_breakfast_df_exchange_symmetrized df_exchange_symmetrized.cpp
)
set_target_properties(
  free_breakfast_df_exchange_symmetrized PROPERTIES CXX_MODULE_STD 1
                                                    CXX_SCAN_FOR_MODULES ON
)
target_link_libraries(
  free_breakfast_df_exchange_symmetrized
  PRIVATE free_breakfast::core free_breakfast::core::resources
          free_breakfast::integrals::eri::density_fitting::jk
)
