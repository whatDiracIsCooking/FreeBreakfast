/// @file
/// @brief Implementation of the Molden writer.

module free_breakfast.io.molden;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.primitives;
import free_breakfast.atom;
import free_breakfast.basis;
import free_breakfast.io.base;

#include "diagnostics.h"

namespace free_breakfast {

// The highest l Molden has a spherical flag for: [9G].
constexpr int32_t kMoldenMaxAngularMomentum = 4;

// N(alpha, l), the factor primitive_normalized folds into a coefficient.
// Dividing it back out gives the coefficient over normalized primitives that
// [GTO] carries.
[[nodiscard]] double molden_primitive_norm(double alpha, int32_t l) {
  return std::pow(2.0 * alpha / std::numbers::pi, 0.75) *
         std::pow(4.0 * alpha, static_cast<double>(l) / 2.0) /
         std::sqrt(math::double_factorial_odd((2 * l) - 1));
}

// The norm of each of a shell's spherical AOs: its contraction over
// normalized primitives c_i, whose pair overlap is
// (2 sqrt(a_i a_j) / (a_i + a_j))^(l + 3/2). A Molden reader renormalizes
// every contraction it reads, so a coefficient over this tree's AO is this
// many times one over the reader's.
[[nodiscard]] double molden_contraction_norm(const Shell &shell) {
  const int32_t l = shell.angular_momentum();
  double norm2 = 0.0;
  for (const Primitive &pi : shell.primitives()) {
    const double ci = pi.coefficient / molden_primitive_norm(pi.exponent, l);
    for (const Primitive &pj : shell.primitives()) {
      const double cj = pj.coefficient / molden_primitive_norm(pj.exponent, l);
      const double ratio = 2.0 * std::sqrt(pi.exponent * pj.exponent) / (pi.exponent + pj.exponent);
      norm2 += ci * cj * std::pow(ratio, static_cast<double>(l) + 1.5);
    }
  }
  return std::sqrt(norm2);
}

// The spherical column (m + l) that Molden's k-th component of an l shell
// reads: m = 0, +1, -1, +2, -2, ... For l <= 1 that is the identity: s is
// trivial and p is (x, y, z) on both sides.
[[nodiscard]] uint32_t molden_component(int32_t l, uint32_t k) {
  if (l <= 1) {
    return k;
  }
  const auto q = static_cast<int32_t>((k + 1) / 2);
  const int32_t m = (k % 2 == 1) ? q : -q;
  return static_cast<uint32_t>(m + l);
}

[[nodiscard]] Result<std::string> molden_to_string(std::span<const Atom> atoms,
                                                   const BasisSet &basis, std::span<const double> c,
                                                   std::span<const double> energies,
                                                   std::span<const double> occupations) {
  if (atoms.empty() || basis.natoms() != atoms.size()) {
    return err_config("Molden: the basis groups its shells over {} atoms, but {} were given",
                      basis.natoms(), atoms.size());
  }
  const size_t nao = basis.nao_sphe();
  const size_t nmo = energies.size();
  if (occupations.size() != nmo || c.size() != nao * nmo) {
    return err_config("Molden: {} coefficients, {} energies and {} occupations do not describe "
                      "orbitals over {} AOs",
                      c.size(), energies.size(), occupations.size(), nao);
  }
  for (const Shell &shell : basis.shells()) {
    if (shell.angular_momentum() > kMoldenMaxAngularMomentum) {
      return err_unsupported("Molden: a shell of l = {} has no spherical flag in the format",
                             shell.angular_momentum());
    }
  }

  std::string out = "[Molden Format]\n[Atoms] AU\n";
  for (size_t a = 0; a < atoms.size(); ++a) {
    const Atom &atom = atoms[a];
    std::format_to(std::back_inserter(out), "{:<2} {:>5} {:>3} {:>22.15e} {:>22.15e} {:>22.15e}\n",
                   element_symbol(atom.element), a + 1, static_cast<int32_t>(atom.element),
                   atom.position.x, atom.position.y, atom.position.z);
  }

  out += "[GTO]\n";
  for (uint32_t a = 0; a < basis.natoms(); ++a) {
    std::format_to(std::back_inserter(out), "{} 0\n", a + 1);
    const uint32_t first = basis.atom_shell_offset(a);
    for (uint32_t s = first; s < first + basis.atom_shell_count(a); ++s) {
      const Shell &shell = basis.shells()[s];
      const int32_t l = shell.angular_momentum();
      std::format_to(std::back_inserter(out), " {} {} 1.00\n", "spdfg"[l], shell.nprimitives());
      for (const Primitive &prim : shell.primitives()) {
        std::format_to(std::back_inserter(out), "{:>22.15e} {:>22.15e}\n", prim.exponent,
                       prim.coefficient / molden_primitive_norm(prim.exponent, l));
      }
    }
    out += '\n';
  }
  out += "[5D7F]\n[9G]\n";

  // Per Molden row: the spherical row of c it reads, and the factor taking a
  // coefficient over this tree's AO to one over the reader's.
  std::vector<uint32_t> row(nao);
  std::vector<double> scale(nao);
  for (uint32_t s = 0; s < basis.nshells(); ++s) {
    const Shell &shell = basis.shells()[s];
    const uint32_t offset = basis.ao_offset_sphe(s);
    const double norm = molden_contraction_norm(shell);
    for (uint32_t k = 0; k < shell.function_count_sphe(); ++k) {
      row[offset + k] = offset + molden_component(shell.angular_momentum(), k);
      scale[offset + k] = norm;
    }
  }

  out += "[MO]\n";
  for (size_t j = 0; j < nmo; ++j) {
    std::format_to(std::back_inserter(out),
                   " Sym= A\n Ene= {:.15e}\n Spin= Alpha\n Occup= {:.6f}\n", energies[j],
                   occupations[j]);
    for (size_t i = 0; i < nao; ++i) {
      std::format_to(std::back_inserter(out), "{:>6} {:>22.15e}\n", i + 1,
                     c[row[i] + (j * nao)] * scale[i]);
    }
  }
  return out;
}

[[nodiscard]] Status write_molden(const std::filesystem::path &path, std::span<const Atom> atoms,
                                  const BasisSet &basis, std::span<const double> c,
                                  std::span<const double> energies,
                                  std::span<const double> occupations) {
  auto text = FBF_TRY(molden_to_string(atoms, basis, c, energies, occupations));
  return write_file(path, text);
}

} // namespace free_breakfast
