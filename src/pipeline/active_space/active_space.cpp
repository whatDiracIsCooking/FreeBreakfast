/// @file
/// @brief ActiveSpaceSelection: the orbital order, the AVAS selection, and
/// the rotation C R they give.

module free_breakfast.pipeline.active_space;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.atom;
import free_breakfast.io.basis;
import free_breakfast.recipes.blas; // gemm
import free_breakfast.recipes.diagonalize;
import free_breakfast.integrals.one_electron.overlap_mixed;

#include "diagnostics.h"

namespace free_breakfast {

[[nodiscard]] Result<std::vector<int32_t>> ActiveSpaceSelection::orbital_order(int32_t n_electrons,
                                                                               int32_t nmo) const {
  if (ncas < 1 || nelecas < 0 || nelecas > 2 * ncas || nelecas > n_electrons) {
    return err_config("active space: needs ncas >= 1 and nelecas in [0, min(2 ncas, {})] (ncas {}, "
                      "nelecas {})",
                      n_electrons, ncas, nelecas);
  }
  if ((n_electrons - nelecas) % 2 != 0) {
    return err_config(
        "active space: {} electrons less nelecas {} leaves an odd core electron count", n_electrons,
        nelecas);
  }
  const int32_t ncore = (n_electrons - nelecas) / 2;
  if (ncore + ncas > nmo) {
    return err_config("active space: ncore {} + ncas {} exceeds the {} orbitals", ncore, ncas, nmo);
  }
  std::vector<int32_t> chosen = active;
  if (chosen.empty()) {
    chosen.resize(static_cast<size_t>(ncas));
    std::iota(chosen.begin(), chosen.end(), ncore);
  }
  if (std::cmp_not_equal(chosen.size(), ncas)) {
    return err_config("active space: active names {} orbitals, not ncas {}", chosen.size(), ncas);
  }
  std::vector<bool> is_active(static_cast<size_t>(nmo), false);
  for (const int32_t i : chosen) {
    if (i < 0 || i >= nmo) {
      return err_config("active space: active orbital {} is not in [0, {})", i, nmo);
    }
    if (is_active[static_cast<size_t>(i)]) {
      return err_config("active space: active orbital {} is named twice", i);
    }
    is_active[static_cast<size_t>(i)] = true;
  }
  // The core is the lowest ncore orbitals not active, then the active ones,
  // then the rest: PySCF's sort_mo.
  std::vector<int32_t> order;
  order.reserve(static_cast<size_t>(nmo));
  for (int32_t i = 0; i < nmo && std::cmp_less(order.size(), ncore); ++i) {
    if (!is_active[static_cast<size_t>(i)]) {
      order.push_back(i);
    }
  }
  const auto first_rest = order.empty() ? 0 : order.back() + 1;
  order.insert(order.end(), chosen.begin(), chosen.end());
  for (int32_t i = first_rest; i < nmo; ++i) {
    if (!is_active[static_cast<size_t>(i)]) {
      order.push_back(i);
    }
  }
  return order;
}

// AVAS's R, as ActiveSpace::r, and the core and active counts it settles.
struct AvasRotation {
  std::vector<double> r;
  int32_t ncore;
  int32_t ncas;
};

// The eigenvalues of the symmetric n x n @p a (column-major, on the host),
// ascending, by diagonalize on res's device; @p a becomes the eigenvectors.
// The two buffers are this call's own: diagonalize takes the workspace.
[[nodiscard]] Result<std::vector<double>> active_space_eigh(DeviceResources &res, size_t n,
                                                            std::vector<double> &a) {
  std::vector<double> w(n);
  if (n == 0) {
    return w;
  }
  const wwr::wwrStream_t stream = res.stream();
  Buffers::device<double> a_device(n * n, res.device());
  Buffers::device<double> w_device(n, res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(a_device.data(), a.data(), n * n * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(diagonalize<double>(res, static_cast<int32_t>(n), a_device.data(), w_device.data()));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(a.data(), a_device.data(), n * n * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(w.data(), w_device.data(), n * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  return w;
}

// Which spherical AOs of @p minao the AVAS labels name. A label is
// "<symbol> <n><l>": every component of the shell of that n and l on every
// atom of that element, n counted from l + 1 over the atom's shells of that l
// in file order, as PySCF's ao_labels counts them.
[[nodiscard]] Result<std::vector<bool>> avas_minao_aos(const Molecule &molecule,
                                                       const BasisSet &minao,
                                                       std::span<const std::string> labels) {
  constexpr std::string_view kMomenta = "spdfghi";
  struct Label {
    Element element;
    int32_t n;
    int32_t l;
  };
  std::vector<Label> parsed;
  for (const std::string &label : labels) {
    std::istringstream fields(label);
    std::string symbol;
    std::string shell;
    std::string extra;
    fields >> symbol >> shell;
    const Element element = element_from_symbol(symbol);
    // The shell is the n's digits and one letter, nothing after it.
    int32_t n = 0;
    const auto [end, ec] = std::from_chars(shell.data(), shell.data() + shell.size(), n);
    const size_t letter = static_cast<size_t>(end - shell.data());
    const size_t l_at = letter + 1 == shell.size() ? kMomenta.find(shell[letter]) : kMomenta.npos;
    const auto l = static_cast<int32_t>(l_at);
    if (fields >> extra || element == Element::Sentinel || ec != std::errc{} ||
        l_at == kMomenta.npos || n <= l) {
      return err_config("active space: avas label '{}' is not '<symbol> <n><l>', as 'O 2p'", label);
    }
    parsed.push_back({element, n, l});
  }
  std::vector<bool> chosen(minao.nao_sphe(), false);
  std::vector<bool> matched(parsed.size(), false);
  for (uint32_t atom = 0; atom < minao.natoms(); ++atom) {
    const Element element = molecule.atoms()[atom].element;
    std::array<int32_t, kMomenta.size()> seen{};
    const uint32_t first = minao.atom_shell_offset(atom);
    for (uint32_t s = first; s < first + minao.atom_shell_count(atom); ++s) {
      const int32_t l = minao.shells()[s].angular_momentum();
      if (std::cmp_greater_equal(l, seen.size())) {
        continue; // no label can name it
      }
      const int32_t n = l + 1 + seen[static_cast<size_t>(l)]++;
      for (size_t k = 0; k < parsed.size(); ++k) {
        if (parsed[k].element == element && parsed[k].n == n && parsed[k].l == l) {
          matched[k] = true;
          const uint32_t ao = minao.ao_offset_sphe(s);
          std::fill_n(chosen.begin() + ao, (2 * l) + 1, true);
        }
      }
    }
  }
  for (size_t k = 0; k < parsed.size(); ++k) {
    if (!matched[k]) {
      return err_config("active space: avas label '{}' names no MINAO AO of the molecule",
                        labels[k]);
    }
  }
  return chosen;
}

// AVAS, as PySCF's mcscf.avas at its defaults for a closed shell (ncore 0,
// no IAOs, canonicalized): the occupied and the virtual orbitals each rotated
// to the eigenvectors of P, the projector onto the labelled MINAO AOs, sa =
// S21^T S2^-1 S21 in the MO basis; weight >= threshold active. Then each
// space is made canonical: rotated to diagonalize the ScfOutput's Fock matrix,
// diag(eps) in the MO basis, within it. The overlaps are built on the device;
// the matrices after them are a few dozen orbitals across, so they are
// assembled on the host and each eigenproblem solved by diagonalize.
[[nodiscard]] Result<AvasRotation> avas_space(DeviceResources &res, const Molecule &molecule,
                                              const BasisSet &basis, const ScfOutput &scf,
                                              std::span<const std::string> labels,
                                              double threshold) {
  if (!(threshold > 0.0 && threshold < 1.0)) {
    return err_config("active space: avas_threshold {} is not in (0, 1)", threshold);
  }
  const BasisStamp stamp = FBF_TRY(resolve_basis("minao"));
  const BasisSet minao = FBF_TRY(place(stamp.primitive_normalized(), molecule));
  const std::vector<bool> chosen = FBF_TRY(avas_minao_aos(molecule, minao, labels));

  const size_t nms = minao.nao_sphe();
  const size_t nao = basis.nao_sphe();
  const size_t nmo = scf.mo_energies.size();
  const auto nocc = static_cast<size_t>(scf.n_alpha);
  const wwr::wwrStream_t stream = res.stream();
  // <MINAO AO | MO> = S(MINAO, basis) C, nms x nmo, and S(MINAO, MINAO).
  Buffers::device<double> s_mix_device(nms * nao, res.device());
  Buffers::device<double> s_mm_device(nms * nms, res.device());
  Buffers::device<double> s_mo_device(nms * nmo, res.device());
  FBF_TRY(overlap_mixed_spherical(res, minao, basis, s_mix_device));
  FBF_TRY(overlap_mixed_spherical(res, minao, minao, s_mm_device));
  FBF_TRY(gpu_status(gemm(res.blas(), static_cast<int32_t>(nms), static_cast<int32_t>(nmo),
                          static_cast<int32_t>(nao), &kOne<double>, s_mix_device.data(),
                          scf.c.data(), &kZero<double>, s_mo_device.data())));
  std::vector<double> s_mo(nms * nmo);
  std::vector<double> s_mm(nms * nms);
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(s_mo.data(), s_mo_device.data(), s_mo.size() * sizeof(double),
                                     wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(s_mm.data(), s_mm_device.data(), s_mm.size() * sizeof(double),
                                     wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());

  std::vector<size_t> picked;
  for (size_t p = 0; p < chosen.size(); ++p) {
    if (chosen[p]) {
      picked.push_back(p);
    }
  }
  const size_t m = picked.size();
  // S21 = <labelled AO | MO>, m x nmo, and S2 = <labelled AO | labelled AO>:
  // the labelled rows (and columns) of the two.
  std::vector<double> s21(m * nmo);
  std::vector<double> s2(m * m);
  for (size_t i = 0; i < m; ++i) {
    for (size_t k = 0; k < nmo; ++k) {
      s21[i + (m * k)] = s_mo[picked[i] + (nms * k)];
    }
    for (size_t i2 = 0; i2 < m; ++i2) {
      s2[i + (m * i2)] = s_mm[picked[i] + (nms * picked[i2])];
    }
  }
  // P = X^T X, X = w^-1/2 V^T S21 for S2 = V w V^T: S21^T S2^-1 S21.
  const std::vector<double> w2 = FBF_TRY(active_space_eigh(res, m, s2));
  if (w2.empty() || w2.front() <= 0.0) {
    return err_numerical("active space: the labelled MINAO AOs' overlap is not positive definite");
  }
  std::vector<double> x(m * nmo, 0.0);
  for (size_t k = 0; k < nmo; ++k) {
    for (size_t q = 0; q < m; ++q) {
      for (size_t i = 0; i < m; ++i) {
        x[q + (m * k)] += s2[i + (m * q)] * s21[i + (m * k)];
      }
      x[q + (m * k)] /= std::sqrt(w2[q]);
    }
  }
  // Each space's orbitals as columns over the nmo MOs.
  std::vector<std::vector<double>> spaces(3);
  const auto split = [&](size_t first, size_t count) -> Status {
    std::vector<double> p(count * count, 0.0);
    for (size_t k = 0; k < count; ++k) {
      for (size_t l = 0; l < count; ++l) {
        for (size_t q = 0; q < m; ++q) {
          p[k + (count * l)] += x[q + (m * (first + k))] * x[q + (m * (first + l))];
        }
      }
    }
    const std::vector<double> weights = FBF_TRY(active_space_eigh(res, count, p));
    for (size_t j = 0; j < count; ++j) {
      // Occupied below the threshold is core, virtual below it the rest.
      std::vector<double> &space = spaces[weights[j] >= threshold ? 1 : (first == 0 ? 0 : 2)];
      const size_t at = space.size();
      space.resize(at + nmo, 0.0);
      std::copy_n(p.begin() + static_cast<std::ptrdiff_t>(count * j), count,
                  space.begin() + static_cast<std::ptrdiff_t>(at + first));
    }
    return {};
  };
  FBF_TRY(split(0, nocc));
  FBF_TRY(split(nocc, nmo - nocc));

  AvasRotation out{.r = {},
                   .ncore = static_cast<int32_t>(spaces[0].size() / nmo),
                   .ncas = static_cast<int32_t>(spaces[1].size() / nmo)};
  if (out.ncas == 0) {
    return err_config("active space: avas selects no orbital at threshold {}", threshold);
  }
  out.r.reserve(nmo * nmo);
  for (const std::vector<double> &b : spaces) {
    const size_t count = b.size() / nmo;
    std::vector<double> f(count * count, 0.0);
    for (size_t i = 0; i < count; ++i) {
      for (size_t j = 0; j < count; ++j) {
        for (size_t k = 0; k < nmo; ++k) {
          f[i + (count * j)] += b[k + (nmo * i)] * scf.mo_energies[k] * b[k + (nmo * j)];
        }
      }
    }
    FBF_TRY(active_space_eigh(res, count, f));
    for (size_t j = 0; j < count; ++j) {
      for (size_t k = 0; k < nmo; ++k) {
        double value = 0.0;
        for (size_t i = 0; i < count; ++i) {
          value += b[k + (nmo * i)] * f[i + (count * j)];
        }
        out.r.push_back(value);
      }
    }
  }
  return out;
}

// The lowest root's natural orbitals to @p path as Molden: columns core,
// natural, rest of C R', R' = R with its active block rotated to gamma's

[[nodiscard]] Result<ActiveSpace> ActiveSpaceSelection::select(DeviceResources &res,
                                                               const Molecule &molecule,
                                                               const BasisSet &basis,
                                                               const ScfOutput &scf) const {
  const int32_t n_electrons = scf.n_alpha + scf.n_beta;
  const auto nmo = static_cast<int32_t>(scf.mo_energies.size());
  const ScopedDeviceIndex on{res.dev_idx()};
  const size_t nao = basis.nao_sphe();
  const auto mo_count = static_cast<size_t>(nmo);

  AvasRotation space{.r = std::vector<double>(mo_count * mo_count, 0.0), .ncore = 0, .ncas = ncas};
  if (avas.empty()) {
    // orbital_order's permutation as R: C R's columns are C's, exactly.
    const std::vector<int32_t> order = FBF_TRY(orbital_order(n_electrons, nmo));
    space.ncore = (n_electrons - nelecas) / 2;
    // An open shell's first n_beta orbitals are doubly occupied and the next
    // n_alpha - n_beta singly: the core holds only the first kind, and the
    // active space every one of the second. (A closed shell's core is
    // orbital_order's alone, as PySCF's sort_mo leaves it.)
    if (scf.n_alpha != scf.n_beta) {
      const auto ncore = static_cast<size_t>(space.ncore);
      for (size_t col = 0; col < order.size(); ++col) {
        const int32_t mo = order[col];
        if (col < ncore && mo >= scf.n_beta) {
          return err_config("active space: core orbital {} is not doubly occupied by the "
                            "ScfOutput's {} alpha and {} beta electrons",
                            mo, scf.n_alpha, scf.n_beta);
        }
        if (col >= ncore + static_cast<size_t>(ncas) && mo >= scf.n_beta && mo < scf.n_alpha) {
          return err_config("active space: singly occupied orbital {} of the ScfOutput's {} alpha "
                            "and {} beta electrons is not active",
                            mo, scf.n_alpha, scf.n_beta);
        }
      }
    }
    for (size_t col = 0; col < order.size(); ++col) {
      space.r[static_cast<size_t>(order[col]) + (mo_count * col)] = 1.0;
    }
  } else {
    if (scf.n_alpha != scf.n_beta) {
      return err_unsupported("active space: avas over an open-shell reference is not "
                             "implemented, and the ScfOutput has {} alpha and {} beta electrons",
                             scf.n_alpha, scf.n_beta);
    }
    if (ncas != 0 || nelecas != 0 || !active.empty()) {
      return err_config(
          "active space: ncas, nelecas and active are avas's to pick, so are not set");
    }
    space = FBF_TRY(avas_space(res, molecule, basis, scf, avas, avas_threshold));
  }

  // C R. Each element of a permutation's product is one C element times 1
  // plus exact zeros, so the index path copies C's columns exactly.
  SphericalMatrixAO c(res, basis);
  Buffers::device<double> r_device(space.r.size(), res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(r_device.data(), space.r.data(),
                                         space.r.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, res.stream())));
  FBF_TRY(gpu_status(gemm(res.blas(), static_cast<int32_t>(nao), nmo, nmo, &kOne<double>,
                          scf.c.data(), r_device.data(), &kZero<double>, c.data())));
  return ActiveSpace{
      .ncore = space.ncore,
      .ncas = space.ncas,
      .nelecas = n_electrons - (2 * space.ncore),
      .r = std::move(space.r),
      .c = std::move(c),
  };
}

} // namespace free_breakfast
