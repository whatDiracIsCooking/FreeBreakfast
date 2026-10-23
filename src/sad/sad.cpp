/// @file
/// @brief Implementation of the SAD initial guess: one @c AtomicDensityStamp
/// per distinct (Z, shells), baked or computed, each placed on its atoms'
/// diagonal blocks -- one after another, or as one task per stamp in a subflow.

module free_breakfast.sad;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.basis;
import free_breakfast.third_party.taskflow;
import :scf;

#include "diagnostics.h"

namespace free_breakfast {

// Two atoms share an SCF when their shells match exactly -- momentum,
// exponents and coefficients -- whatever their centers.
bool same_shells(std::span<const Shell> a, std::span<const Shell> b) {
  return std::ranges::equal(a, b, [](const Shell &x, const Shell &y) {
    return x.angular_momentum() == y.angular_momentum() &&
           std::ranges::equal(x.primitives(), y.primitives(),
                              [](const Primitive &p, const Primitive &q) {
                                return p.exponent == q.exponent && p.coefficient == q.coefficient;
                              });
  });
}

// One atom's converged SCF density over its own spherical AOs, column-major.
// Every integral of the atomic SCF is one-center, so the block is
// geometry-free: every atom with this Z and these shells wears the same one.
struct AtomicDensityStamp {
  int32_t z;
  std::span<const Shell> shells;
  std::vector<double> density;
};

// Write @p stamp's block onto the diagonal of the n x n column-major
// @p density, starting at AO @p offset.
void place(const AtomicDensityStamp &stamp, size_t offset, size_t n, std::span<double> density) {
  size_t block = 0;
  for (const Shell &shell : stamp.shells) {
    block += shell.function_count_sphe();
  }
  for (size_t j = 0; j < block; ++j) {
    for (size_t i = 0; i < block; ++i) {
      density[(offset + i) + (n * (offset + j))] = stamp.density[i + (block * j)];
    }
  }
}

// One distinct (Z, shells) of a molecule and where it goes: its stamp, whose
// density stays empty until its SCF runs, and the first AO of each atom that
// wears it.
struct StampSites {
  AtomicDensityStamp stamp;
  std::vector<size_t> offsets;
};

// The distinct (Z, shells) of @p atoms over @p basis, in order of each one's
// first atom: the first of them to fail is the one the first failing atom
// wears, so both forms of sad_density report the same error.
Result<std::vector<StampSites>> distinct_stamps(const BasisSet &basis,
                                                std::span<const Atom> atoms) {
  if (basis.natoms() != atoms.size()) {
    return err_config("sad_density: the basis groups its shells over {} atoms but {} were given; "
                      "the basis was not placed over these atoms",
                      basis.natoms(), atoms.size());
  }
  std::vector<StampSites> sites;
  for (uint32_t a = 0; a < basis.natoms(); ++a) {
    const uint32_t count = basis.atom_shell_count(a);
    if (count == 0) {
      continue;
    }
    const uint32_t first = basis.atom_shell_offset(a);
    const std::span<const Shell> shells(basis.shells().data() + first, count);
    const int32_t z = nuclear_charge(atoms[a]);
    auto it = std::ranges::find_if(sites, [&](const StampSites &s) {
      return s.stamp.z == z && same_shells(s.stamp.shells, shells);
    });
    if (it == sites.end()) {
      sites.push_back(StampSites{AtomicDensityStamp{z, shells, {}}, {}});
      it = std::prev(sites.end());
    }
    it->offsets.push_back(basis.ao_offset_sphe(first));
  }
  return sites;
}

// @p stamp's density: @p baked's for @p basis, when it is named and @p baked
// has this element, else its own SCF's. A baked block is checked against the
// shells only by size, the one thing a lookup by name can get wrong that a
// guess cannot survive.
Result<std::vector<double>> stamp_density(const AtomicDensityStamp &stamp, const BasisSet &basis,
                                          SadStampLookup baked) {
  if (baked != nullptr && !basis.name().empty()) {
    if (std::optional<std::vector<double>> density = baked(basis.name(), stamp.z)) {
      size_t block = 0;
      for (const Shell &shell : stamp.shells) {
        block += shell.function_count_sphe();
      }
      if (density->size() != block * block) {
        return err_config("sad_density: the baked density for Z={} in {} holds {} values, but "
                          "the atom's shells need {}x{}",
                          stamp.z, basis.name(), density->size(), block, block);
      }
      return std::move(*density);
    }
  }
  AtomicScf scf = FBF_TRY(atomic_scf(stamp.z, stamp.shells));
  return std::move(scf.density);
}

// Fill @p site's stamp, then place it on each of its atoms' blocks of the
// n x n @p density. Every atom's block is its own, so sites placed
// concurrently never write the same element.
Status stamp_and_place(StampSites &site, const BasisSet &basis, SadStampLookup baked, size_t n,
                       std::span<double> density) {
  site.stamp.density = FBF_TRY(stamp_density(site.stamp, basis, baked));
  for (const size_t offset : site.offsets) {
    place(site.stamp, offset, n, density);
  }
  return {};
}

// stamp_and_place as a task of @p flow, its outcome in @p status. @p site,
// @p density's storage and @p status are written when the task runs, not when
// this returns, so they must outlive it.
tf::Task atomic_density_stamp(tf::FlowBuilder &flow, StampSites &site, const BasisSet &basis,
                              SadStampLookup baked, size_t n, std::span<double> density,
                              Status &status) {
  return flow.emplace([&site, &basis, baked, n, density, &status] {
    status = stamp_and_place(site, basis, baked, n, density);
  });
}

Result<std::vector<double>> sad_density(const BasisSet &basis, std::span<const Atom> atoms,
                                        SadStampLookup baked) {
  std::vector<StampSites> sites = FBF_TRY(distinct_stamps(basis, atoms));
  const size_t n = basis.nao_sphe();
  std::vector<double> density(n * n, 0.0);
  for (StampSites &site : sites) {
    FBF_TRY(stamp_and_place(site, basis, baked, n, density));
  }
  return density;
}

tf::Task sad_density(tf::FlowBuilder &flow, const BasisSet &basis, std::span<const Atom> atoms,
                     Result<std::vector<double>> &out, SadStampLookup baked) {
  return flow.emplace([&basis, atoms, &out, baked](tf::Subflow &subflow) {
    auto sites = distinct_stamps(basis, atoms);
    if (!sites) {
      out = std::unexpected(std::move(sites.error()));
      return;
    }
    const size_t n = basis.nao_sphe();
    std::vector<double> density(n * n, 0.0);
    // Sized once, before any task exists: each task holds a reference to its
    // site and its status, which a reallocation would leave dangling.
    std::vector<Status> statuses(sites->size());
    for (size_t k = 0; k < sites->size(); ++k) {
      atomic_density_stamp(subflow, (*sites)[k], basis, baked, n, density, statuses[k]);
    }
    // Join here, explicitly. Taskflow's implicit join runs the subflow only
    // after this lambda returns, by which point sites, density and statuses --
    // locals of this frame -- are gone.
    subflow.join();
    // In stamp order, which is first-atom order: the error the serial form
    // stops at.
    for (Status &status : statuses) {
      if (!status) {
        out = std::unexpected(std::move(status.error()));
        return;
      }
    }
    out = std::move(density);
  });
}

} // namespace free_breakfast
