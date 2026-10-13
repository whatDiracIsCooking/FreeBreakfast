/// @file
/// @brief The geometry-free basis *definition* -- @ref free_breakfast::BasisStamp,
/// a per-element table of @ref free_breakfast::ShellStamp, and @ref
/// free_breakfast::place, which stamps it onto a molecule's atoms to make a
/// centered @ref free_breakfast::BasisSet.
///
/// A @ref BasisSet embeds a center per shell, so it describes a basis *over one
/// geometry*. The information a basis-set file actually carries is
/// center-independent: for each element, the contracted shells (@ref
/// ShellStamp) that an atom of that element wears wherever it sits. @ref place
/// is the cheap operation that copies the stamps onto real atom positions, in
/// molecule order, to produce the @ref BasisSet the integral engines consume.
/// Splitting the two removes the old wart where the same geometry lived in both
/// @ref Molecule (positions) and @ref BasisSet (shell centers) with only a doc
/// comment keeping them consistent: build one stamp, @ref place it at each
/// geometry.
///
/// Everything stays in atomic units: primitive exponents in inverse bohr^2, and
/// the centers @ref place stamps on are the molecule's atom positions in bohr.
module;

#include "basis.h"

export module free_breakfast.basis:basis_stamp;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.molecule;
import :shell;
import :shell_stamp;
import :basis_set;

export namespace free_breakfast {

/// @brief A geometry-free basis definition: for each element (by atomic number
/// Z), the contracted shells (@ref ShellStamp) an atom of that element wears.
///
/// This is what a basis-set file specifies -- no positions. @ref place stamps it
/// onto a molecule's atoms, in molecule order, to make the centered @ref BasisSet
/// the integral engines consume. One stamp placed at two geometries yields two
/// consistent @ref BasisSet objects, which is the point of separating the two.
class BasisStamp {
public:
  BasisStamp() = default;

  /// @param element_shells The element -> shells table (Z as returned by
  ///                       @c nuclear_charge). Elements a molecule does not use
  ///                       may be present; @ref place ignores them.
  explicit BasisStamp(std::map<int, std::vector<ShellStamp>> element_shells)
      : element_shells_(std::move(element_shells)) {}

  /// @brief The element -> shells table.
  [[nodiscard]] const std::map<int, std::vector<ShellStamp>> &element_shells() const {
    return element_shells_;
  }

  /// @brief The shell stamps for element @p z, or @c nullptr if the stamp
  /// carries none.
  [[nodiscard]] const std::vector<ShellStamp> *shells_for(int z) const {
    const auto it = element_shells_.find(z);
    return it == element_shells_.end() ? nullptr : &it->second;
  }

  /// @brief A copy with every shell's per-primitive normalization folded in (see
  /// @ref ShellStamp::primitive_normalized). Center-independent, so it is done
  /// once on the stamp, not once per @ref place.
  [[nodiscard]] BasisStamp primitive_normalized() const {
    std::map<int, std::vector<ShellStamp>> out;
    for (const auto &[z, shells] : element_shells_) {
      std::vector<ShellStamp> normalized;
      normalized.reserve(shells.size());
      for (const ShellStamp &sh : shells) {
        normalized.push_back(sh.primitive_normalized());
      }
      out.emplace(z, std::move(normalized));
    }
    return BasisStamp{std::move(out)};
  }

private:
  std::map<int, std::vector<ShellStamp>> element_shells_{};
};

/// @brief Stamp a @ref BasisStamp onto a molecule's atoms to make a centered
/// @ref BasisSet.
///
/// For each atom, in molecule order, the element's shell stamps (@ref
/// BasisStamp::shells_for) are placed at that atom's center and appended to the
/// basis. The AO layout therefore follows atom order, then the stamp's shell
/// order. This is the cheap, behavior-preserving successor to the old
/// parse-then-assign path: build one stamp, @c place it at each geometry.
///
/// @param basis_stamp The geometry-free definition.
/// @param molecule    The atoms (positions in bohr) to stamp onto.
/// @returns The centered basis, or ErrorKind::InvalidConfig if any atom's element
///   is absent from the stamp -- a basis that cannot cover the molecule is a
///   configuration error, not a silently smaller basis.
[[nodiscard]] Result<BasisSet> place(const BasisStamp &basis_stamp, const Molecule &molecule) {
  std::vector<Shell> shells;
  for (const Atom<double> &atom : molecule.atoms()) {
    const int z = nuclear_charge(atom);
    const std::vector<ShellStamp> *element = basis_stamp.shells_for(z);
    if (element == nullptr) {
      return err_config("basis has no shells for element Z={}", z);
    }
    const wwr::double3 center = position(atom);
    for (const ShellStamp &proto : *element) {
      shells.push_back(proto.at(center));
    }
  }
  return BasisSet{std::move(shells)};
}

} // namespace free_breakfast
