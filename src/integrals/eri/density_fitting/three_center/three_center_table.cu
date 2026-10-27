// The generated eri3<la, lb, lP> table, checked against (mn|P)'s limits
// (three_center_bridge.h) at compile time. No launcher yet: this TU defines
// nothing. One source for both backends -- nvcc under CUDA, clang -x hip under
// HIP.
//
// kernels/eri3.h declares the template, every specialization and eri3_kernels,
// their [la][lb][lP] table, nullptr where la > lb (only la <= lb is emitted).

// Standard headers before anything that reaches the vendor runtime: under HIP a
// libc++ header must not follow it (see primitives.h). The bridge includes
// integral_types.h first; the kernels' header brings the runtime in through
// felizjueves/runtime.h.
#include <type_traits>

#include "three_center_bridge.h"

#include "kernels/eri3.h"

static_assert(std::extent_v<decltype(eri3_kernels), 0> ==
                      free_breakfast::device::kThreeCenterMaxMomentum + 1 &&
                  std::extent_v<decltype(eri3_kernels), 1> ==
                      free_breakfast::device::kThreeCenterMaxMomentum + 1,
              "kernels/'s [la][lb] extents and kThreeCenterMaxMomentum disagree: "
              "rerun tools/emit_df_kernels.py");

static_assert(std::extent_v<decltype(eri3_kernels), 2> ==
                  free_breakfast::device::kThreeCenterAuxMaxMomentum + 1,
              "kernels/'s [lP] extent and kThreeCenterAuxMaxMomentum disagree: "
              "rerun tools/emit_df_kernels.py");
