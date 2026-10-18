/// @file
/// @brief The `[la][lb]` dispatch table of a one-electron kernel family, built
/// at compile time from the family's kernel template.
///
/// FelizJueves emits each class of a family as one explicit specialization
/// `name<la, lb>` of a kernel template, and the classes it emitted as
/// `name_classes` (tools/emit_one_electron_kernels.py). A launcher holds that
/// list to @ref lists_every_class and builds its table with
/// @ref kernel_table, so the table follows kMaxAngularMomentum and names no
/// kernel by hand. A class the list names but no unit defines is a link error.
///
/// Shared with device code, so everything here is public (docs/style.md,
/// "Headers shared with device code"). Standard headers only: a launcher
/// includes this before anything that reaches the vendor runtime.

#pragma once

#include <array>
#include <cstddef>
#include <utility>

#include "angular_momentum.h"

namespace free_breakfast {

/// @brief The extent of each index of a @ref KernelTable.
inline constexpr int32_t kKernelTableWidth = kMaxAngularMomentum + 1;

/// @brief A kernel per `[la][lb]` class; null where `la > lb`, because the
/// host swaps every shell pair into `la <= lb` (free_breakfast.integrals.mmd).
template <class Kernel>
using KernelTable = std::array<std::array<Kernel, kKernelTableWidth>, kKernelTableWidth>;

/// @brief Whether @p classes is every `la <= lb <= kMaxAngularMomentum`, each
/// once: as many rows as there are classes, and every class among them.
template <std::size_t N>
constexpr bool lists_every_class(const int32_t (&classes)[N][2]) {
  if (N != static_cast<std::size_t>(kKernelTableWidth * (kKernelTableWidth + 1) / 2)) {
    return false;
  }
  for (int32_t la = 0; la < kKernelTableWidth; ++la) {
    for (int32_t lb = la; lb < kKernelTableWidth; ++lb) {
      bool listed = false;
      for (const auto &c : classes) {
        listed = listed || (c[0] == la && c[1] == lb);
      }
      if (!listed) {
        return false;
      }
    }
  }
  return true;
}

/// @brief The table whose `[la][lb]` cell, for `la <= lb`, is
/// `Family::template kernel<la, lb>()`: a struct whose static member template
/// names the family's specialization, `return &overlap<la, lb>;`. A struct, not
/// a captureless template lambda: nvcc warns (#445) on such a lambda's
/// conversion to a function pointer, whose template parameters it cannot deduce.
template <class Kernel, class Family>
constexpr KernelTable<Kernel> kernel_table() {
  KernelTable<Kernel> table{};
  const auto set = [&]<int32_t la, int32_t lb>() {
    if constexpr (la <= lb) {
      table[la][lb] = Family::template kernel<la, lb>();
    }
  };
  [&]<int32_t... I>(std::integer_sequence<int32_t, I...>) {
    (set.template operator()<I / kKernelTableWidth, I % kKernelTableWidth>(), ...);
  }(std::make_integer_sequence<int32_t, kKernelTableWidth * kKernelTableWidth>());
  return table;
}

} // namespace free_breakfast
