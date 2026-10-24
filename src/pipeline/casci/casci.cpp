/// @file
/// @brief CasciStep: the orbital order, the AVAS selection, and run's path
/// from the Scf's orbitals through the active-space Hamiltonian and the CI
/// solve to the 1-RDMs and the natural-orbital Molden file.

module free_breakfast.pipeline.casci;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.atom;
import free_breakfast.io.basis;
import free_breakfast.io.molden;
import free_breakfast.recipes.diagonalize;
import free_breakfast.integrals.one_electron.overlap_mixed;
import free_breakfast.integrals.eri.integral_direct.coulomb;
import free_breakfast.integrals.eri.integral_direct.exchange;
import free_breakfast.integrals.active_space;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;
import free_breakfast.ci.solver;
import free_breakfast.ci.rdm;

#include "diagnostics.h"

namespace free_breakfast {

[[nodiscard]] Result<std::vector<int32_t>> CasciStep::orbital_order(int32_t n_electrons,
                                                                    int32_t nmo) const {
  if (ncas < 1 || nelecas < 0 || nelecas > 2 * ncas || nelecas > n_electrons) {
    return err_config("casci: needs ncas >= 1 and nelecas in [0, min(2 ncas, {})] (ncas {}, "
                      "nelecas {})",
                      n_electrons, ncas, nelecas);
  }
  if ((n_electrons - nelecas) % 2 != 0) {
    return err_config("casci: {} electrons less nelecas {} leaves an odd core electron count",
                      n_electrons, nelecas);
  }
  const int32_t ncore = (n_electrons - nelecas) / 2;
  if (ncore + ncas > nmo) {
    return err_config("casci: ncore {} + ncas {} exceeds the {} orbitals", ncore, ncas, nmo);
  }
  std::vector<int32_t> chosen = active;
  if (chosen.empty()) {
    chosen.resize(static_cast<size_t>(ncas));
    std::iota(chosen.begin(), chosen.end(), ncore);
  }
  if (std::cmp_not_equal(chosen.size(), ncas)) {
    return err_config("casci: active names {} orbitals, not ncas {}", chosen.size(), ncas);
  }
  std::vector<bool> is_active(static_cast<size_t>(nmo), false);
  for (const int32_t i : chosen) {
    if (i < 0 || i >= nmo) {
      return err_config("casci: active orbital {} is not in [0, {})", i, nmo);
    }
    if (is_active[static_cast<size_t>(i)]) {
      return err_config("casci: active orbital {} is named twice", i);
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

// The active space as a rotation of the Scf's orbitals: Casci::c = C R, R
// nmo x nmo, column-major, orthogonal, its columns core, active, the rest.
struct CasciSpace {
  std::vector<double> r;
  int32_t ncore;
  int32_t ncas;
};

// The eigenvalues of the symmetric n x n @p a (column-major, on the host),
// ascending, by diagonalize on res's device; @p a becomes the eigenvectors.
// The two buffers are this call's own: diagonalize takes the workspace.
[[nodiscard]] Result<std::vector<double>> casci_eigh(DeviceResources &res, size_t n,
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
      return err_config("casci: avas label '{}' is not '<symbol> <n><l>', as 'O 2p'", label);
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
      return err_config("casci: avas label '{}' names no MINAO AO of the molecule", labels[k]);
    }
  }
  return chosen;
}

// AVAS, as PySCF's mcscf.avas at its defaults for a closed shell (ncore 0,
// no IAOs, canonicalized): the occupied and the virtual orbitals each rotated
// to the eigenvectors of P, the projector onto the labelled MINAO AOs, sa =
// S21^T S2^-1 S21 in the MO basis; weight >= threshold active. Then each
// space is made canonical: rotated to diagonalize the Scf's Fock matrix,
// diag(eps) in the MO basis, within it. The overlaps are built on the device;
// the matrices after them are a few dozen orbitals across, so they are
// assembled on the host and each eigenproblem solved by diagonalize.
[[nodiscard]] Result<CasciSpace> avas_space(DeviceResources &res, const Molecule &molecule,
                                            const BasisSet &basis, const Scf &scf,
                                            std::span<const std::string> labels, double threshold) {
  if (!(threshold > 0.0 && threshold < 1.0)) {
    return err_config("casci: avas_threshold {} is not in (0, 1)", threshold);
  }
  const BasisStamp stamp = FBF_TRY(resolve_basis("minao"));
  const BasisSet minao = FBF_TRY(place(stamp.primitive_normalized(), molecule));
  const std::vector<bool> chosen = FBF_TRY(avas_minao_aos(molecule, minao, labels));

  const size_t nms = minao.nao_sphe();
  const size_t nao = basis.nao_sphe();
  const size_t nmo = scf.mo_energies.size();
  const auto nocc = static_cast<size_t>(molecule.electron_count() / 2);
  const wwr::wwrStream_t stream = res.stream();
  // <MINAO AO | MO> = S(MINAO, basis) C, nms x nmo, and S(MINAO, MINAO).
  Buffers::device<double> s_mix_device(nms * nao, res.device());
  Buffers::device<double> s_mm_device(nms * nms, res.device());
  Buffers::device<double> s_mo_device(nms * nmo, res.device());
  FBF_TRY(overlap_mixed_spherical(res, minao, basis, s_mix_device));
  FBF_TRY(overlap_mixed_spherical(res, minao, minao, s_mm_device));
  constexpr double kOneScale = 1.0;
  constexpr double kNoAccumulate = 0.0;
  FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, static_cast<int32_t>(nms),
      static_cast<int32_t>(nmo), static_cast<int32_t>(nao), &kOneScale, s_mix_device.data(),
      static_cast<int32_t>(nms), scf.c.data(), static_cast<int32_t>(nao), &kNoAccumulate,
      s_mo_device.data(), static_cast<int32_t>(nms))));
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
  const std::vector<double> w2 = FBF_TRY(casci_eigh(res, m, s2));
  if (w2.empty() || w2.front() <= 0.0) {
    return err_numerical("casci: the labelled MINAO AOs' overlap is not positive definite");
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
    const std::vector<double> weights = FBF_TRY(casci_eigh(res, count, p));
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

  CasciSpace out{.r = {},
                 .ncore = static_cast<int32_t>(spaces[0].size() / nmo),
                 .ncas = static_cast<int32_t>(spaces[1].size() / nmo)};
  if (out.ncas == 0) {
    return err_config("casci: avas selects no orbital at threshold {}", threshold);
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
    FBF_TRY(casci_eigh(res, count, f));
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
// eigenvectors by descending occupation; each energy sum_k R'_kj^2 eps_k.
[[nodiscard]] Status write_casci_molden(DeviceResources &res, const std::filesystem::path &path,
                                        const Molecule &molecule, const BasisSet &basis,
                                        const Scf &scf, const CasciSpace &space,
                                        std::span<const double> gamma) {
  const size_t nao = basis.nao_sphe();
  const size_t nmo = scf.mo_energies.size();
  const auto ncore = static_cast<size_t>(space.ncore);
  const auto n = static_cast<size_t>(space.ncas);
  std::vector<double> v(gamma.begin(), gamma.end()); // symmetric: row- is column-major
  const std::vector<double> occupation = FBF_TRY(casci_eigh(res, n, v));
  std::vector<double> r = space.r;
  for (size_t j = 0; j < n; ++j) {
    for (size_t k = 0; k < nmo; ++k) {
      double value = 0.0;
      for (size_t t = 0; t < n; ++t) {
        value += space.r[k + (nmo * (ncore + t))] * v[t + (n * (n - 1 - j))];
      }
      r[k + (nmo * (ncore + j))] = value;
    }
  }
  std::vector<double> c(nao * nmo);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(c.data(), scf.c.data(), c.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  std::vector<double> orbitals(nao * nmo, 0.0);
  std::vector<double> energies(nmo, 0.0);
  std::vector<double> occupations(nmo, 0.0);
  for (size_t j = 0; j < nmo; ++j) {
    for (size_t k = 0; k < nmo; ++k) {
      const double rkj = r[k + (nmo * j)];
      energies[j] += rkj * rkj * scf.mo_energies[k];
      for (size_t mu = 0; mu < nao; ++mu) {
        orbitals[mu + (nao * j)] += c[mu + (nao * k)] * rkj;
      }
    }
    if (j < ncore) {
      occupations[j] = 2.0;
    } else if (j < ncore + n) {
      occupations[j] = occupation[n - 1 - (j - ncore)];
    }
  }
  return write_molden(path, molecule.atoms(), basis, orbitals, energies, occupations);
}

[[nodiscard]] Result<Casci> CasciStep::run(
    View<Molecule, BasisSet, SphericalTransform, PrimitivePairClasses, CoreHamiltonian, Scf> in,
    DeviceResources &res) const {
  const Molecule &molecule = in.get<Molecule>();
  const BasisSet &basis = in.get<BasisSet>();
  const Scf &scf = in.get<Scf>();
  const auto nmo = static_cast<int32_t>(scf.mo_energies.size());
  const ScopedDeviceIndex on{res.dev_idx()};
  const wwr::wwrStream_t stream = res.stream();
  const size_t nao = basis.nao_sphe();
  const auto mo_count = static_cast<size_t>(nmo);

  CasciSpace space{.r = std::vector<double>(mo_count * mo_count, 0.0), .ncore = 0, .ncas = ncas};
  if (avas.empty()) {
    // orbital_order's permutation as R: C R's columns are C's, exactly.
    const std::vector<int32_t> order = FBF_TRY(orbital_order(molecule.electron_count(), nmo));
    space.ncore = (molecule.electron_count() - nelecas) / 2;
    for (size_t col = 0; col < order.size(); ++col) {
      space.r[static_cast<size_t>(order[col]) + (mo_count * col)] = 1.0;
    }
  } else {
    if (ncas != 0 || nelecas != 0 || !active.empty()) {
      return err_config("casci: ncas, nelecas and active are avas's to pick, so are not set");
    }
    space = FBF_TRY(avas_space(res, molecule, basis, scf, avas, avas_threshold));
  }
  const int32_t ncore = space.ncore;
  const int32_t n_active = space.ncas;
  const int32_t n_electrons = molecule.electron_count() - (2 * ncore);

  // Casci::c = C R. Each element of a permutation's product is one C element
  // times 1 plus exact zeros, so the index path copies C's columns exactly.
  SphericalMatrixAO c(res, basis);
  Buffers::device<double> r_device(space.r.size(), res.device());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(r_device.data(), space.r.data(),
                                         space.r.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  constexpr double kOneScale = 1.0;
  constexpr double kNoAccumulate = 0.0;
  FBF_TRY(gpu_status(wwr::gemm<double, int32_t>(
      res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, static_cast<int32_t>(nao), nmo, nmo,
      &kOneScale, scf.c.data(), static_cast<int32_t>(nao), r_device.data(), nmo, &kNoAccumulate,
      c.data(), static_cast<int32_t>(nao))));

  const std::vector<PrimitivePairClass> &pairs = in.get<PrimitivePairClasses>().classes;
  std::vector<CoulombPairClass> j_classes = FBF_TRY(coulomb_pair_classes(res, pairs));
  std::vector<OrderedPairClass> k_classes = FBF_TRY(ordered_pair_classes(res, basis, pairs));
  const ActiveSpaceHamiltonian ham = FBF_TRY(active_space_hamiltonian(
      res, basis, molecule, in.get<CoreHamiltonian>().h, in.get<SphericalTransform>().u, j_classes,
      k_classes, c, ncore, n_active));

  // The sigma takes host arrays, row-major: h_eff transposed, and (tu|vw)
  // from the Hamiltonian's t-fastest order to w-fastest.
  const auto n = static_cast<size_t>(n_active);
  std::vector<double> h_col(n * n);
  std::vector<double> eri_col(n * n * n * n);
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(h_col.data(), ham.h_eff.data(), h_col.size() * sizeof(double),
                                     wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(eri_col.data(), ham.eri.data(),
                                         eri_col.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  std::vector<double> h_eff(n * n);
  std::vector<double> eri(n * n * n * n);
  for (size_t t = 0; t < n; ++t) {
    for (size_t u = 0; u < n; ++u) {
      h_eff[(t * n) + u] = h_col[t + (n * u)];
      for (size_t v = 0; v < n; ++v) {
        for (size_t w = 0; w < n; ++w) {
          eri[(((((t * n) + u) * n) + v) * n) + w] = eri_col[t + (n * (u + (n * (v + (n * w)))))];
        }
      }
    }
  }

  const int32_t n_beta = n_electrons / 2;
  const StringSet alpha = FBF_TRY(StringSet::create(n_active, n_electrons - n_beta));
  const StringSet beta = FBF_TRY(StringSet::create(n_active, n_beta));
  CiSigma sigma = FBF_TRY(CiSigma::create(res, alpha, beta));
  FBF_TRY(sigma.set_integrals(res, ham.e_core, h_eff, eri));
  const auto dim = static_cast<size_t>(sigma.size());
  Buffers::device<double> diagonal_device(dim, res.device());
  FBF_TRY(sigma.diagonal(res, diagonal_device.data()));
  std::vector<double> diagonal(dim);
  FBF_TRY(
      gpu_status(wwr::wwrMemcpyAsync(diagonal.data(), diagonal_device.data(), dim * sizeof(double),
                                     wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  CiRoots roots = FBF_TRY(ci_solve(res, sigma, alpha, beta, diagonal, nroots));

  const auto count = static_cast<size_t>(nroots);
  std::vector<double> gamma(count * n * n);
  for (size_t r = 0; r < count; ++r) {
    FBF_TRY(rdm1(alpha, beta, std::span<const double>{roots.vectors.data() + (r * dim), dim},
                 std::span<double>{gamma.data() + (r * n * n), n * n}));
  }
  if (!molden.empty()) {
    FBF_TRY(write_casci_molden(res, molden, molecule, basis, scf, space,
                               std::span<const double>{gamma.data(), n * n}));
  }
  return Casci{
      .ncore = ncore,
      .ncas = n_active,
      .nelecas = n_electrons,
      .c = std::move(c),
      .energies = std::move(roots.energies),
      .spin_squares = std::move(roots.spin_squares),
      .vectors = std::move(roots.vectors),
      .rdm1 = std::move(gamma),
      .davidson = roots.davidson,
  };
}

} // namespace free_breakfast
