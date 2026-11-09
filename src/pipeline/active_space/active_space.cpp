/// @file
/// @brief ActiveSpaceSelection: the orbital order, the AVAS selection, and
/// the rotation C R they give.

module free_breakfast.pipeline.active_space;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants; // kOne, kZero, kNegOne
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.core.primitives; // math::binomial
import free_breakfast.atom;
import free_breakfast.io.basis;
import free_breakfast.io.npy;
import free_breakfast.recipes.blas; // gemm, gemmwich
import free_breakfast.recipes.diagonalize;
import free_breakfast.recipes.orthogonalize;
import free_breakfast.integrals.one_electron.overlap_mixed;
import free_breakfast.ci.strings;
import free_breakfast.ci.sigma;

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
[[nodiscard]] Result<std::vector<bool>> avas_minao_aos(std::span<const Atom> atoms,
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
    const Element element = atoms[atom].element;
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

// AVAS, as PySCF's mcscf.avas at its defaults (ncore 0, no IAOs,
// openshell_option 2, canonicalized): the occupied orbitals, the first
// n_alpha (an open shell's singly occupied ones among them, projected with
// the doubly occupied ones), and the virtual orbitals each rotated to the
// eigenvectors of P, the projector onto the labelled MINAO AOs, sa =
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
  const std::vector<bool> chosen = FBF_TRY(avas_minao_aos(molecule.atoms(), minao, labels));

  const size_t nms = minao.nao_sphe();
  const size_t nao = basis.nao_sphe();
  const size_t nmo = scf.mo_energies.size();
  const auto nocc = static_cast<size_t>(scf.n_alpha);
  const wwr::wwrStream_t stream = res.stream();
  // <MINAO AO | MO> = S(MINAO, basis) C, nms x nmo, and S(MINAO, MINAO), to
  // the host; their device buffers are freed before the eigenproblems.
  std::vector<double> s_mo(nms * nmo);
  std::vector<double> s_mm(nms * nms);
  {
    Buffers::device<double> s_mix_device(nms * nao, res.device());
    Buffers::device<double> s_mm_device(nms * nms, res.device());
    Buffers::device<double> s_mo_device(nms * nmo, res.device());
    FBF_TRY(overlap_mixed_spherical(res, minao, basis, s_mix_device));
    FBF_TRY(overlap_mixed_spherical(res, minao, minao, s_mm_device));
    FBF_TRY(gpu_status(gemm(res.blas(), static_cast<int32_t>(nms), static_cast<int32_t>(nmo),
                            static_cast<int32_t>(nao), &kOne<double>, s_mix_device.data(),
                            scf.c.data(), &kZero<double>, s_mo_device.data())));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(s_mo.data(), s_mo_device.data(),
                                           s_mo.size() * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                                           stream)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(s_mm.data(), s_mm_device.data(),
                                           s_mm.size() * sizeof(double), wwr::wwrMemcpyDeviceToHost,
                                           stream)));
    FBF_TRY(res.sync());
  }

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

// The orbitals of the .npy at @p path, columns in @p order, into @p c,
// orthonormalized in this geometry's overlap S block by block: the core and
// active columns [0, nocc) by Löwdin, A' = A (A^T S A)^-1/2; then the rest
// with A' projected out of them, B - A' (A'^T S B), by Löwdin too. Returns
// their rotation of @p scf's orbitals, R = C_scf^T S C' (nmo square,
// column-major), so C_scf R = C' to rounding: C_scf is a complete orthonormal
// set (nmo = nao). The largest |A^T S A - I| element is logged: 0 to rounding
// at the geometry the orbitals were converged at, and how far the carried
// space moved otherwise.
[[nodiscard]] Result<std::vector<double>>
carried_orbitals(DeviceResources &res, const BasisSet &basis, const ScfOutput &scf,
                 const std::filesystem::path &path, std::span<const int32_t> order, size_t nocc,
                 SphericalMatrixAO &c) {
  const NpyArray file = FBF_TRY(read_npy(path));
  const size_t nao = basis.nao_sphe();
  const size_t nmo = scf.mo_energies.size();
  if (file.shape.size() != 2 || file.shape[0] != nao || file.shape[1] != nmo) {
    return err_config("active space: orbitals '{}' have shape {}, not the basis's {} AOs by {} "
                      "orbitals",
                      path.string(), file.shape, nao, nmo);
  }
  // The file's columns in the selection's order, so A is [0, nocc).
  std::vector<double> host(nao * nmo);
  for (size_t col = 0; col < nmo; ++col) {
    std::copy_n(file.data.begin() + static_cast<std::ptrdiff_t>(nao * order[col]), nao,
                host.begin() + static_cast<std::ptrdiff_t>(nao * col));
  }
  const auto ao = static_cast<int32_t>(nao);
  const auto n_a = static_cast<int32_t>(nocc);
  const auto n_b = static_cast<int32_t>(nmo - nocc);
  const wwr::wwrStream_t stream = res.stream();
  // S, and scratch held across orthogonalize, which takes the workspace.
  Buffers::device<double> s(nao * nao, res.device());
  Buffers::device<double> work(nao * nmo, res.device());
  Buffers::device<double> m(nmo * nmo, res.device());
  Buffers::device<double> x(nmo * nmo, res.device());
  double *const a = c.data();
  double *const b = c.data() + (nao * nocc);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(a, host.data(), host.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, stream)));
  FBF_TRY(overlap_mixed_spherical(res, basis, basis, s));

  // A' = A (A^T S A)^-1/2.
  FBF_TRY(gpu_status(gemmwich(res.blas(), n_a, ao, a, s.data(), work.data(), m.data())));
  std::vector<double> m_a(nocc * nocc);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(m_a.data(), m.data(), m_a.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  double deviation = 0.0;
  for (size_t j = 0; j < nocc; ++j) {
    for (size_t i = 0; i < nocc; ++i) {
      deviation = std::max(deviation, std::abs(m_a[i + (nocc * j)] - (i == j ? 1.0 : 0.0)));
    }
  }
  log::info(std::format("active space: orbitals '{}', max |C^T S C - I| {:.3e} over the {} core "
                        "and active ones here",
                        path.string(), deviation, nocc));
  FBF_TRY(orthogonalize(res, n_a, m.data(), x.data()));
  FBF_TRY(gpu_status(
      gemm(res.blas(), ao, n_a, n_a, &kOne<double>, a, x.data(), &kZero<double>, work.data())));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(a, work.data(), nao * nocc * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, stream)));
  if (n_b > 0) {
    // B <- B - A' (A'^T S B), then B (B^T S B)^-1/2.
    FBF_TRY(gpu_status(
        gemm(res.blas(), ao, n_b, ao, &kOne<double>, s.data(), b, &kZero<double>, work.data())));
    FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), n_a, n_b, ao, &kOne<double>, a,
                                               work.data(), &kZero<double>, m.data())));
    FBF_TRY(gpu_status(
        gemm(res.blas(), ao, n_b, n_a, &kNegOne<double>, a, m.data(), &kOne<double>, b)));
    FBF_TRY(gpu_status(gemmwich(res.blas(), n_b, ao, b, s.data(), work.data(), m.data())));
    FBF_TRY(orthogonalize(res, n_b, m.data(), x.data()));
    FBF_TRY(gpu_status(
        gemm(res.blas(), ao, n_b, n_b, &kOne<double>, b, x.data(), &kZero<double>, work.data())));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(b, work.data(), nao * (nmo - nocc) * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, stream)));
  }
  // R = C_scf^T (S C').
  const auto n = static_cast<int32_t>(nmo);
  FBF_TRY(gpu_status(
      gemm(res.blas(), n, &kOne<double>, s.data(), c.data(), &kZero<double>, work.data())));
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(res.blas(), n, &kOne<double>, scf.c.data(),
                                             work.data(), &kZero<double>, m.data())));
  std::vector<double> r(nmo * nmo);
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(r.data(), m.data(), r.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, stream)));
  FBF_TRY(res.sync());
  return r;
}

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
  if (!orbitals.empty() && !avas.empty()) {
    return err_config("active space: avas selects from the ScfOutput's orbitals, so takes no "
                      "orbitals file");
  }
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
    if (!orbitals.empty()) {
      // The file's orbitals in this order, orthonormalized here: they are
      // C R and R the rotation, with no gemm of the ScfOutput's C after.
      SphericalMatrixAO c(res, basis);
      space.r = FBF_TRY(carried_orbitals(res, basis, scf, orbitals, order,
                                         static_cast<size_t>(space.ncore + ncas), c));
      return ActiveSpace{
          .ncore = space.ncore,
          .ncas = space.ncas,
          .nelecas = n_electrons - (2 * space.ncore),
          .r = std::move(space.r),
          .c = std::move(c),
      };
    }
    for (size_t col = 0; col < order.size(); ++col) {
      space.r[static_cast<size_t>(order[col]) + (mo_count * col)] = 1.0;
    }
  } else {
    if (ncas != 0 || nelecas != 0 || !active.empty()) {
      return err_config(
          "active space: ncas, nelecas and active are avas's to pick, so are not set");
    }
    space = FBF_TRY(avas_space(res, molecule, basis, scf, avas, avas_threshold));
    // Refused, not truncated: a cut could split a degenerate set.
    if (avas_max_ncas > 0 && space.ncas > avas_max_ncas) {
      return err_config(
          "active space: AVAS picked {} orbitals at threshold {}, avas_max_ncas is {}", space.ncas,
          avas_threshold, avas_max_ncas);
    }
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

MemoryFootprint ActiveSpaceSelection::select_footprint(const FootprintDimensions &dims) const {
  const BasisSet &basis = *dims.basis;
  const size_t nao = dims.nao;
  const size_t nmo = nao; // the ScfOutput's orbitals span the basis
  const size_t nn = nao * nmo;
  const size_t n_electrons = dims.n_alpha + dims.n_beta;
  const size_t ncore = dims.nelecas <= n_electrons ? (n_electrons - dims.nelecas) / 2 : 0;
  const size_t ncas = std::min(dims.ncas, nmo - std::min(ncore, nmo));
  const auto held = [](size_t bytes) { return MemoryFootprint{.device = {.persistent = bytes}}; };
  // diagonalize at order n of an n x n matrix and its eigenvalues held across
  // it: active_space_eigh's.
  const auto eigh = [&](const size_t n) {
    MemoryFootprint out = diagonalize_footprint<double>(dims.syevd_lwork[n]);
    out.device.scratch = n == 0 ? 0 : sizeof(double) * ((n * n) + n);
    return out;
  };
  const auto widest = [](MemoryFootprint a, const MemoryFootprint &b) {
    a.device.scratch = std::max(a.device.scratch, b.device.scratch);
    a.device.workspace = std::max(a.device.workspace, b.device.workspace);
    a.pinned.workspace = std::max(a.pinned.workspace, b.pinned.workspace);
    return a;
  };
  // ActiveSpace::c throughout.
  const MemoryFootprint c = held(sizeof(double) * nn);

  if (!orbitals.empty()) {
    // carried_orbitals: S, the work, M and X held across S's build and the two
    // Löwdin orthogonalizations, of the ncore + ncas core and active columns
    // and of the rest.
    const size_t n_a = ncore + ncas;
    const MemoryFootprint steps = widest(
        widest(overlap_mixed_spherical_footprint(basis, basis),
               orthogonalize_footprint<double>(static_cast<int32_t>(n_a), dims.syevd_lwork[n_a])),
        orthogonalize_footprint<double>(static_cast<int32_t>(nmo - n_a),
                                        dims.syevd_lwork[nmo - n_a]));
    return then(c, drop(then(held(sizeof(double) * ((2 * nao * nao) + (2 * nmo * nmo))), steps)));
  }

  // R's upload, beside C R.
  const MemoryFootprint rotation =
      then(c, MemoryFootprint{.device = {.scratch = sizeof(double) * nmo * nmo}});
  if (avas.empty()) {
    return rotation;
  }
  // AVAS: the three overlaps held across the two mixed builds, then the
  // eigenproblems one at a time: the labelled AOs' overlap, the occupied and
  // the virtual projectors, and each of core, active and the rest
  // canonicalized.
  const BasisSet &minao = *dims.minao;
  const size_t nms = minao.nao_sphe();
  const std::vector<bool> chosen = check_or_abort(avas_minao_aos(dims.atoms, minao, avas));
  const auto m = static_cast<size_t>(std::ranges::count(chosen, true));
  const size_t nocc = dims.n_alpha;
  const MemoryFootprint overlaps = then(held(sizeof(double) * nms * (nao + nms + nmo)),
                                        widest(overlap_mixed_spherical_footprint(minao, basis),
                                               overlap_mixed_spherical_footprint(minao, minao)));
  MemoryFootprint eighs;
  for (const size_t n : {m, nocc, nmo - nocc, ncore, ncas, nmo - ncore - ncas}) {
    eighs = widest(eighs, eigh(n));
  }
  return then(drop(then(drop(overlaps), eighs)), rotation);
}

MemoryFootprint active_space_ci_footprint(const int32_t ncas, const int32_t n_alpha,
                                          const int32_t n_beta, const int32_t nroots,
                                          const ci::SolverOptions &options,
                                          const std::span<const size_t> syevd_lwork,
                                          const size_t sigma_scratch_limit) {
  const auto strings = [&](const int32_t electrons) {
    return static_cast<int32_t>(std::round(math::binomial(ncas, electrons)));
  };
  const int32_t alpha_strings = strings(n_alpha);
  const int32_t beta_strings = strings(n_beta);
  const size_t dim = static_cast<size_t>(alpha_strings) * static_cast<size_t>(beta_strings);
  const auto n = static_cast<size_t>(ncas);
  const auto held = [](size_t bytes) { return MemoryFootprint{.device = {.persistent = bytes}}; };
  MemoryFootprint sets = ci::device_string_set_footprint(ncas, n_alpha);
  if (n_alpha != n_beta) {
    sets = then(sets, ci::device_string_set_footprint(ncas, n_beta));
  }
  return then(then(then(then(sets, ci::sigma_footprint(ncas, alpha_strings, beta_strings)),
                        held(sizeof(double) * dim)),
                   ci::solve_footprint(ncas, n_alpha, n_beta, nroots, options, false, syevd_lwork,
                                       sigma_scratch_limit)),
              held(sizeof(double) * static_cast<size_t>(nroots) * n * n));
}

} // namespace free_breakfast
