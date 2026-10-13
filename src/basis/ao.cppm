/// @file
/// @brief The atomic-orbital view of a shell: the canonical Cartesian ordering
/// (@ref free_breakfast::cartesian_powers) and one self-describing Cartesian
/// atomic orbital (@ref free_breakfast::AO).
///
/// A shell of angular momentum L spans the Cartesian Gaussians x^a y^b z^c with
/// a+b+c = L. @ref cartesian_powers enumerates those (a, b, c) triples in the
/// **one canonical order** every integral tile and normalization diagonal in
/// this project assumes -- a highest, then b highest: s -> {(0,0,0)}; p ->
/// (x, y, z); d -> (xx, xy, xz, yy, yz, zz). It takes a plain integer L and is
/// defined for arbitrary L (the host reference engines use it beyond d); this
/// partition is the single source of truth for that order.
///
/// @ref AO binds one such triple to the shell that owns it. It carries no
/// center or exponents -- its @c shell_index resolves those through the owning
/// @ref BasisSet -- so @c :basis_set imports this partition to lay out its AO
/// list without a dependency cycle.
module;

#include "basis.h"

export module free_breakfast.basis:ao;

import std;
import free_breakfast.core.types;

export namespace free_breakfast {

/// @brief The Cartesian powers (a, b, c) of a shell of momentum @p l, in the
/// project's canonical order (a highest, then b highest).
///
/// s -> {(0,0,0)}; p -> (x, y, z); d -> (xx, xy, xz, yy, yz, zz). The result has
/// @ref cartesian_function_count(l) entries. This exact ordering is load-bearing:
/// the GPU kernels' Cartesian tiles and the per-AO normalization diagonal are
/// laid out against it. Defined for arbitrary @p l >= 0, so the host reference
/// engines can raise angular momentum past d.
[[nodiscard]] std::vector<std::array<int32_t, 3>> cartesian_powers(int32_t l) {
  std::vector<std::array<int32_t, 3>> out;
  out.reserve(static_cast<size_t>(cartesian_function_count(l)));
  for (int32_t a = l; a >= 0; --a) {
    for (int32_t b = l - a; b >= 0; --b) {
      out.push_back({a, b, l - a - b});
    }
  }
  return out;
}

/// @brief One Cartesian atomic orbital: the shell that owns it and its Cartesian
/// powers (a, b, c), with a + b + c = L.
///
/// Plain aggregate, brace-initialised (`AO{shell_index, a, b, c}`). It stores no
/// center or exponents; @c shell_index indexes back into the owning @ref
/// BasisSet, which resolves the shell's center, momentum and primitives. In the
/// AO list a @ref BasisSet lays out (@ref BasisSet::aos), an orbital's position
/// is its global AO index, so none is stored here.
struct AO {
  uint32_t shell_index{0};
  uint8_t a{0};
  uint8_t b{0};
  uint8_t c{0};
};

} // namespace free_breakfast
