/// @file
/// @brief data/df_high_aux/'s hand-built bases, read from its manifest: the
/// fixture df_high_aux_gpu.cpp runs the top of the aux ladder over, and
/// df_high_aux_host.cpp checks reaches every class.
///
/// Include it AFTER `import std;`, `import free_breakfast.basis;` and
/// `import free_breakfast.io.json;`, like scratch_dir.h: it includes nothing
/// itself.

#pragma once

namespace free_breakfast::test::high_aux {

/// @brief The manifest's shells, built exactly as it holds them: one
/// primitive each, its coefficient already the primitive's own normaliser,
/// so these go into a Shell as given (as data/README.md's one_electron
/// shells do) and need no primitive_normalized.
inline BasisSet basis_of(const json &shells) {
  std::vector<Shell> built;
  for (const json &shell : shells) {
    const auto centre = shell["centre"].get<std::vector<double>>();
    built.push_back(Shell{shell["l"].get<std::int32_t>(),
                          {{shell["exponent"].get<double>(), shell["coefficient"].get<double>()}},
                          {centre[0], centre[1], centre[2]}});
  }
  return BasisSet{std::move(built)};
}

struct Fixture {
  BasisSet basis;
  BasisSet aux;
  std::size_t nao;
  std::size_t naux;
};

/// @brief Both bases as the manifest holds them, with the counts it records.
/// Read per case rather than once into a static: this code does not throw
/// (docs/style.md), so the read's failure belongs in a case's ASSERT.
inline Fixture fixture(const json &doc) {
  return Fixture{basis_of(doc["orbital_shells"]), basis_of(doc["aux_shells"]),
                 doc["nao"].get<std::size_t>(), doc["naux"].get<std::size_t>()};
}

} // namespace free_breakfast::test::high_aux
