// GPU cases for hessian::df_sacasscf_z: a state-averaged DF-CASSCF Hessian's
// z terms at one root's multipliers (casscf_hessian.tex eq. sa, bullets 3 and
// 4), L_xy's z part and L_xlambda, against test/gtest/data/casscf_hessian/
// (tools/emit/casscf_hessian_reference.py): the stretched water's
// SA-CAS(4,4) / 6-31G over cc-pVDZ-JKFIT at two roots (0.5, 0.5) and
// (0.7, 0.3), three equal and (0.5, 0.3, 0.2), every root. The orbitals,
// roots and z are the fixture's, never re-converged (PySCF's two-root
// solutions are saddle points). Checked: L_xy's z part piece by piece
// against z_one_electron ... exy_z, its symmetry and translational sum rule;
// L_xlambda's covectors against root_rhs + lxl_z (their CI rows' singlet
// part, as the fixture's are over the singlet space) and L_xlambda lambda^y
// against lxl_lambda; and the refusals.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.io.basis;
import free_breakfast.io.json;
import free_breakfast.io.npy;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.integrals;
import free_breakfast.integrals.eri.density_fitting;
import free_breakfast.integrals.eri.density_fitting.fit;
import free_breakfast.ci;
import free_breakfast.hessian;
import free_breakfast.test.ci_oracles;

namespace {

namespace fb = free_breakfast;

// Against the fixture, absolute: the issue's bar is 1e-8; the same formulas
// over the same fit measure near 1e-12 (README.md).
constexpr double kFixtureTol = 1e-10;
// L_xy's z part's asymmetry and every term's sum over the atoms, absolute.
constexpr double kSymmetryTol = 1e-11;
constexpr double kSumRuleTol = 1e-9;

const std::filesystem::path kDataDir = std::filesystem::path{FBF_TEST_DATA_DIR} / "casscf_hessian";

// <i|i> of a shell's (l, 0, 0) Cartesian, which is each of its spherical
// functions' too, over its coefficients as held (df_jk_derivative_gpu.cpp).
double self_overlap(const fb::Shell &shell) {
  const std::int32_t l = shell.angular_momentum();
  double dfact = 1.0;
  for (std::int32_t k = (2 * l) - 1; k > 1; k -= 2) {
    dfact *= k;
  }
  double sum = 0.0;
  for (const fb::Primitive &p : shell.primitives()) {
    for (const fb::Primitive &q : shell.primitives()) {
      const double g = p.exponent + q.exponent;
      sum += p.coefficient * q.coefficient * dfact * std::pow(std::numbers::pi / g, 1.5) /
             std::pow(2.0 * g, l);
    }
  }
  return sum;
}

fb::Buffers::device<double> upload(fb::DeviceResources &res, const std::vector<double> &host) {
  fb::Buffers::device<double> d(host.size(), res.device());
  EXPECT_EQ(wwr::wwrMemcpyAsync(d.data(), host.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyHostToDevice, res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return d;
}

std::vector<double> download(fb::DeviceResources &res, const double *d, const std::size_t count) {
  std::vector<double> host(count);
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), d, sizeof(double) * count, wwr::wwrMemcpyDeviceToHost,
                                res.stream()),
            wwr::wwrSuccess);
  EXPECT_TRUE(res.sync().has_value());
  return host;
}

std::vector<double> read_array(const std::string &file, const std::vector<std::size_t> &shape) {
  const auto a = fb::read_npy(kDataDir / file);
  EXPECT_TRUE(a) << file;
  if (!a) {
    return {};
  }
  EXPECT_EQ(a->shape, shape) << file;
  return a->data;
}

double max_abs(const std::vector<double> &a) {
  return std::ranges::fold_left(
      a, 0.0, [](const double m, const double x) { return std::max(m, std::abs(x)); });
}

double max_diff(const std::vector<double> &a, const std::vector<double> &b) {
  EXPECT_EQ(a.size(), b.size());
  double worst = 0.0;
  for (std::size_t i = 0; i < std::min(a.size(), b.size()); ++i) {
    worst = std::max(worst, std::abs(a[i] - b[i]));
  }
  return worst;
}

// max |A - A^T| of an n x n matrix.
double asymmetry(const std::vector<double> &a, const std::size_t n) {
  double worst = 0.0;
  for (std::size_t y = 0; y < n; ++y) {
    for (std::size_t x = 0; x < n; ++x) {
      worst = std::max(worst, std::abs(a[x + (n * y)] - a[y + (n * x)]));
    }
  }
  return worst;
}

// max over k and every row i of |Sum_A M[i][3 A + k]|, M rows x 3 n_atom
// column-major: translational invariance in the coordinate index.
double sum_rule(const std::vector<double> &m, const std::size_t rows, const std::size_t n) {
  double worst = 0.0;
  for (std::size_t k = 0; k < 3; ++k) {
    for (std::size_t i = 0; i < rows; ++i) {
      double sum = 0.0;
      for (std::size_t x = k; x < n; x += 3) {
        sum += m[i + (rows * x)];
      }
      worst = std::max(worst, std::abs(sum));
    }
  }
  return worst;
}

// The fixture's (orb, ci, theta) layout of @p columns vectors into the packed
// one: the pairs (q core then q active, p ascending), each root's d, then
// Theta_sr for r < s (mcscf_integral_derivative_gpu.cpp).
std::vector<double> pack_vectors(const std::vector<double> &orb, const std::vector<double> &ci,
                                 const std::vector<double> &theta, std::size_t nmo,
                                 std::size_t ncore, std::size_t ncas, std::size_t dim,
                                 std::size_t nroots, bool rotations, std::size_t columns) {
  const std::size_t nocc = ncore + ncas;
  std::vector<double> out;
  for (std::size_t col = 0; col < columns; ++col) {
    const double *k = orb.data() + (col * nmo * nmo);
    for (std::size_t q = 0; q < nocc; ++q) {
      for (std::size_t p = std::max(q < ncore ? ncore : nocc, q + 1); p < nmo; ++p) {
        out.push_back(k[p + (nmo * q)]);
      }
    }
    const double *c = ci.data() + (col * dim * nroots);
    out.insert(out.end(), c, c + (dim * nroots));
    if (rotations) {
      const double *t = theta.data() + (col * nroots * nroots);
      for (std::size_t r = 0; r < nroots; ++r) {
        for (std::size_t s = r + 1; s < nroots; ++s) {
          out.push_back(t[s + (nroots * r)]);
        }
      }
    }
  }
  return out;
}

// The singlet part of every length-@p dim column of @p v, in place.
void project_singlet(const fb::ci::StringSet &alpha, const fb::ci::StringSet &beta,
                     std::span<double> v, const std::size_t dim) {
  const std::int32_t s_max = (alpha.n_electrons() + beta.n_electrons()) / 2;
  std::vector<double> s2c(dim);
  for (std::size_t at = 0; at < v.size(); at += dim) {
    const std::span<double> col(v.data() + at, dim);
    for (std::int32_t s = 1; s <= s_max; ++s) {
      const double eig = s * (s + 1.0);
      ASSERT_TRUE(fb::test::host_spin_square(alpha, beta, col, s2c));
      for (std::size_t i = 0; i < dim; ++i) {
        col[i] = (s2c[i] - (eig * col[i])) / -eig;
      }
    }
  }
}

// One state-averaged case set up as the DF steps would, the fixture's
// orbitals made ours (rows in this tree's AO order and normalisation).
class HessianDfSaCasscfZ : public ::testing::Test {
protected:
  void load(const std::string &name) {
    const auto doc = fb::read_json(kDataDir / "casscf_hessian.json");
    ASSERT_TRUE(doc) << doc.error().message;
    for (const fb::json &entry : (*doc)["cases"]) {
      if (entry["name"].get<std::string>() == name) {
        case_ = entry;
      }
    }
    ASSERT_FALSE(case_.is_null()) << "no case " << name;
    res_.emplace(0);
    std::vector<fb::Atom> atoms;
    for (const auto &atom : case_["atoms"]) {
      const auto xyz = atom["coord"].get<std::vector<double>>();
      atoms.push_back(fb::Atom{fb::Vec<double, 3>{xyz[0], xyz[1], xyz[2]},
                               static_cast<fb::Element>(atom["charge"].get<std::int32_t>())});
    }
    molecule_.emplace(*res_, std::move(atoms));
    const auto stamp = fb::resolve_basis(case_["basis"].get<std::string>());
    ASSERT_TRUE(stamp) << stamp.error().message;
    auto basis = fb::place(stamp->primitive_normalized(), *molecule_);
    ASSERT_TRUE(basis) << basis.error().message;
    basis_.emplace(std::move(*basis));
    const auto aux_stamp = fb::resolve_aux_basis(case_["aux_basis"].get<std::string>());
    ASSERT_TRUE(aux_stamp) << aux_stamp.error().message;
    auto aux = fb::place(aux_stamp->primitive_normalized(), *molecule_);
    ASSERT_TRUE(aux) << aux.error().message;
    aux_.emplace(std::move(*aux));
    nmo_ = case_["nmo"].get<std::size_t>();
    naux_ = case_["naux"].get<std::size_t>();
    ASSERT_EQ(basis_->nao_sphe(), nmo_);
    ASSERT_EQ(aux_->nao_sphe(), naux_);
    n_x_ = 3 * basis_->natoms();
    ncore_ = case_["ncore"].get<std::int32_t>();
    ncas_ = case_["ncas"].get<std::int32_t>();
    const auto nelecas = case_["nelecas"].get<std::vector<std::int32_t>>();
    nalpha_ = nelecas[0];
    nbeta_ = nelecas[1];
    weights_ = case_["weights"].get<std::vector<double>>();
    nroots_ = weights_.size();
    rotations_ = !case_["equal_weights"].get<bool>();
    dim_ = case_["na"].get<std::size_t>() * case_["nb"].get<std::size_t>();

    auto classes = fb::primitive_pair_classes(*res_, *basis_);
    ASSERT_TRUE(classes) << classes.error().message;
    classes_ = std::move(*classes);
    auto su = fb::upload_spherical_transform_matrix(*res_, *basis_);
    ASSERT_TRUE(su) << su.error().message;
    h_.emplace(*res_, *basis_, std::move(*su));
    ASSERT_TRUE(fb::core_hamiltonian(*res_, classes_, *molecule_, h_->cartesian()));
    ASSERT_TRUE(h_->cartesian_to_spherical(*res_));
    auto u = fb::upload_spherical_transform_matrix(*res_, *aux_);
    ASSERT_TRUE(u) << u.error().message;
    metric_.emplace(*res_, *aux_, std::move(*u));
    b_.emplace(naux_ * nmo_ * nmo_, res_->device());
    ASSERT_TRUE(fb::metric(*res_, *aux_, *metric_));
    ASSERT_TRUE(fb::three_center(*res_, *basis_, classes_, *aux_, b_->data()));
    ASSERT_TRUE(fb::density_fit(*res_, static_cast<std::int32_t>(nmo_),
                                static_cast<std::int32_t>(naux_), metric_->spherical().data(),
                                b_->data()));

    // Row k of the fixture's C is this tree's function order[k]: PySCF sorts
    // each atom's shells by l, this tree keeps the file's order.
    std::vector<double> norms;
    for (const fb::Shell &shell : basis_->shells()) {
      norms.insert(norms.end(), shell.function_count_sphe(), std::sqrt(self_overlap(shell)));
    }
    std::vector<std::size_t> order;
    for (std::uint32_t atom = 0; atom < basis_->natoms(); ++atom) {
      std::vector<std::uint32_t> shells(basis_->atom_shell_count(atom));
      std::iota(shells.begin(), shells.end(), basis_->atom_shell_offset(atom));
      std::ranges::stable_sort(shells, {}, [&](const std::uint32_t i) {
        return basis_->shells()[i].angular_momentum();
      });
      for (const std::uint32_t i : shells) {
        const std::size_t first = basis_->ao_offset_sphe(i);
        for (std::size_t f = 0; f < basis_->shells()[i].function_count_sphe(); ++f) {
          order.push_back(first + f);
        }
      }
    }
    ASSERT_EQ(order.size(), nmo_);
    const std::vector<double> c_pyscf =
        read_array(case_["arrays"]["c"].get<std::string>(), {nmo_, nmo_});
    std::vector<double> c_host(nmo_ * nmo_, 0.0);
    for (std::size_t j = 0; j < nmo_; ++j) {
      for (std::size_t k = 0; k < nmo_; ++k) {
        c_host[order[k] + (nmo_ * j)] = c_pyscf[k + (nmo_ * j)] / norms[order[k]];
      }
    }
    c_.emplace(*res_, *basis_);
    ASSERT_EQ(wwr::wwrMemcpyAsync(c_->data(), c_host.data(), sizeof(double) * c_host.size(),
                                  wwr::wwrMemcpyHostToDevice, res_->stream()),
              wwr::wwrSuccess);
    ASSERT_TRUE(res_->sync());
    roots_.emplace(
        upload(*res_, read_array(case_["arrays"]["ci"].get<std::string>(), {dim_, nroots_})));
    ASSERT_FALSE(HasFailure());
  }

  [[nodiscard]] std::vector<double> root_array(const std::size_t root, const std::string &key,
                                               const std::vector<std::size_t> &shape) const {
    return read_array(case_["roots"][root]["arrays"][key].get<std::string>(), shape);
  }

  // Covector-sized fixture arrays (orb, ci, theta) under @p stem, packed.
  [[nodiscard]] std::vector<double> packed(const std::size_t root, const std::string &stem,
                                           const std::size_t columns) const {
    const std::size_t n = columns;
    return pack_vectors(
        root_array(root, stem + "_orb",
                   n == 1 ? std::vector{nmo_, nmo_} : std::vector{nmo_, nmo_, n}),
        root_array(root, stem + "_ci",
                   n == 1 ? std::vector{dim_, nroots_} : std::vector{dim_, nroots_, n}),
        rotations_
            ? root_array(root, stem + "_theta",
                         n == 1 ? std::vector{nroots_, nroots_} : std::vector{nroots_, nroots_, n})
            : std::vector<double>{},
        nmo_, static_cast<std::size_t>(ncore_), static_cast<std::size_t>(ncas_), dim_, nroots_,
        rotations_, n);
  }

  fb::Result<fb::hessian::SaCasscfZTerms> terms(const std::vector<double> &weights,
                                                const std::int32_t root,
                                                const fb::Buffers::device<double> &z) {
    return fb::hessian::df_sacasscf_z(*res_, *molecule_, *basis_, classes_, *aux_, h_->spherical(),
                                      *b_, metric_->spherical().data(), *c_, ncore_, ncas_, nalpha_,
                                      nbeta_, *roots_, weights, root, z);
  }

  void check_root(const std::size_t root) {
    SCOPED_TRACE(std::format("root {}", root));
    const std::size_t n = n_x_;
    const std::vector<double> z_host = packed(root, "z", 1);
    ASSERT_FALSE(HasFailure());
    const fb::Buffers::device<double> z = upload(*res_, z_host);
    const auto start = std::chrono::steady_clock::now();
    auto got = terms(weights_, static_cast<std::int32_t>(root), z);
    ASSERT_TRUE(got) << got.error().message;
    const double seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    const std::size_t size = z_host.size();
    ASSERT_EQ(got->lxlambda.num_elements(), size * n);

    // L_xy's z part, piece by piece.
    std::string report;
    const fb::hessian::CasscfExy &lxy = got->lxy;
    for (const auto &[key, piece] :
         {std::pair{"z_one_electron", &lxy.one_electron},
          std::pair{"z_two_electron", &lxy.two_electron.total},
          std::pair{"z_connection_second", &lxy.connection_second},
          std::pair{"z_connection_fock", &lxy.connection_fock},
          std::pair{"z_connection_twice", &lxy.connection_twice}, std::pair{"exy_z", &lxy.exy}}) {
      const std::vector<double> ref = root_array(root, key, {n, n});
      const double worst = max_diff(*piece, ref);
      const double asym = asymmetry(*piece, n);
      EXPECT_LE(worst, kFixtureTol) << key << ", of max " << max_abs(ref);
      EXPECT_LE(asym, kSymmetryTol) << key;
      report += std::format("; {} {:.1e} of {:.2e}", key, worst, max_abs(ref));
    }
    EXPECT_EQ(max_abs(lxy.nuclear_repulsion), 0.0);
    const double lxy_drift = sum_rule(lxy.exy, n, n);
    EXPECT_LE(lxy_drift, kSumRuleTol);

    // L_xlambda's covectors against root_rhs + lxl_z, the CI rows' singlet
    // parts, and L_xlambda lambda^y (before projecting: lambda is a singlet).
    std::vector<double> lxl_host = download(*res_, got->lxlambda.data(), size * n);
    const std::vector<double> lambda = pack_vectors(
        read_array(case_["arrays"]["lambda_orb"].get<std::string>(), {nmo_, nmo_, n}),
        read_array(case_["arrays"]["lambda_ci"].get<std::string>(), {dim_, nroots_, n}),
        rotations_
            ? read_array(case_["arrays"]["lambda_theta"].get<std::string>(), {nroots_, nroots_, n})
            : std::vector<double>{},
        nmo_, static_cast<std::size_t>(ncore_), static_cast<std::size_t>(ncas_), dim_, nroots_,
        rotations_, n);
    ASSERT_EQ(lambda.size(), size * n);
    std::vector<double> lxl_lambda(n * n, 0.0);
    for (std::size_t y = 0; y < n; ++y) {
      for (std::size_t x = 0; x < n; ++x) {
        double sum = 0.0;
        for (std::size_t i = 0; i < size; ++i) {
          sum += lxl_host[i + (size * x)] * lambda[i + (size * y)];
        }
        lxl_lambda[x + (n * y)] = sum;
      }
    }
    const std::vector<double> lxl_ref = root_array(root, "lxl_lambda", {n, n});
    const double lxl_worst = max_diff(lxl_lambda, lxl_ref);
    EXPECT_LE(lxl_worst, kFixtureTol) << "L_xlambda lambda^y, of max " << max_abs(lxl_ref);
    const double lxl_drift = sum_rule(lxl_host, size, n);
    EXPECT_LE(lxl_drift, kSumRuleTol);

    std::vector<double> want = packed(root, "root_rhs", n);
    const std::vector<double> lxl_z = packed(root, "lxl_z", n);
    ASSERT_EQ(want.size(), lxl_z.size());
    ASSERT_EQ(want.size(), lxl_host.size());
    for (std::size_t i = 0; i < want.size(); ++i) {
      want[i] += lxl_z[i];
    }
    const std::size_t pairs =
        size - (dim_ * nroots_) - (rotations_ ? nroots_ * (nroots_ - 1) / 2 : 0);
    const auto alpha = fb::ci::StringSet::create(ncas_, nalpha_);
    const auto beta = fb::ci::StringSet::create(ncas_, nbeta_);
    ASSERT_TRUE(alpha && beta);
    for (std::size_t x = 0; x < n; ++x) {
      project_singlet(*alpha, *beta,
                      std::span(lxl_host).subspan((size * x) + pairs, dim_ * nroots_), dim_);
    }
    std::array<double, 3> worst{};
    std::array<double, 3> scale{};
    for (std::size_t x = 0; x < n; ++x) {
      for (std::size_t i = 0; i < size; ++i) {
        const std::size_t part = i < pairs ? 0 : (i < pairs + (dim_ * nroots_) ? 1 : 2);
        worst[part] =
            std::max(worst[part], std::abs(lxl_host[i + (size * x)] - want[i + (size * x)]));
        scale[part] = std::max(scale[part], std::abs(want[i + (size * x)]));
      }
    }
    EXPECT_LE(worst[0], kFixtureTol) << "orbital part, of max " << scale[0];
    EXPECT_LE(worst[1], kFixtureTol) << "CI part, of max " << scale[1];
    EXPECT_LE(worst[2], kFixtureTol) << "Theta part, of max " << scale[2];

    std::cout << std::format("[ SA z     ] {} root {}: {:.2f} s{}; L_xlambda covectors orb {:.1e} "
                             "ci {:.1e} theta {:.1e} (of {:.2e} {:.2e} {:.2e}), L_xlambda "
                             "lambda^y {:.1e} of {:.2e}; sum rules {:.1e} {:.1e}\n",
                             case_["name"].get<std::string>(), root, seconds, report, worst[0],
                             worst[1], worst[2], scale[0], scale[1], scale[2], lxl_worst,
                             max_abs(lxl_ref), lxy_drift, lxl_drift);
  }

  void check(const std::string &name) {
    FBF_REQUIRE_GPU();
    load(name);
    ASSERT_FALSE(HasFatalFailure());
    for (std::size_t root = 0; root < nroots_; ++root) {
      check_root(root);
      ASSERT_FALSE(HasFatalFailure());
    }
  }

  fb::json case_;
  std::optional<fb::DeviceResources> res_;
  std::optional<fb::Molecule> molecule_;
  std::optional<fb::BasisSet> basis_;
  std::optional<fb::BasisSet> aux_;
  std::vector<fb::PrimitivePairClass> classes_;
  std::optional<fb::MatrixAOPair> h_;
  std::optional<fb::MatrixAOPair> metric_;
  std::optional<fb::Buffers::device<double>> b_;
  std::optional<fb::SphericalMatrixAO> c_;
  std::optional<fb::Buffers::device<double>> roots_;
  std::vector<double> weights_;
  std::size_t nmo_ = 0;
  std::size_t naux_ = 0;
  std::size_t n_x_ = 0;
  std::size_t dim_ = 0;
  std::size_t nroots_ = 0;
  bool rotations_ = false;
  std::int32_t ncore_ = 0;
  std::int32_t ncas_ = 0;
  std::int32_t nalpha_ = 0;
  std::int32_t nbeta_ = 0;
};

TEST_F(HessianDfSaCasscfZ, TwoRootsEqualWeightsMatchReference) {
  check("h2o_631g_sa2");
}

TEST_F(HessianDfSaCasscfZ, TwoRootsUnequalWeightsMatchReference) {
  check("h2o_631g_sa2_unequal");
}

TEST_F(HessianDfSaCasscfZ, ThreeRootsEqualWeightsMatchReference) {
  check("h2o_631g_sa3");
}

TEST_F(HessianDfSaCasscfZ, ThreeRootsUnequalWeightsMatchReference) {
  check("h2o_631g_sa3_unequal");
}

// Weights equal for some pairs only, a root out of range, one root, and z of
// the wrong length are refused before anything runs.
TEST_F(HessianDfSaCasscfZ, RefusesBadInputs) {
  FBF_REQUIRE_GPU();
  load("h2o_631g_sa3");
  ASSERT_FALSE(HasFatalFailure());
  const std::vector<double> z_host = packed(0, "z", 1);
  const fb::Buffers::device<double> z = upload(*res_, z_host);
  const auto refused = [](const auto &r, std::string_view what) {
    ASSERT_FALSE(r) << what;
    EXPECT_EQ(r.error().kind, fb::ErrorKind::InvalidConfig) << what;
    EXPECT_NE(r.error().message.find("hessian::df_sacasscf_z"), std::string::npos) << what;
  };
  refused(terms({0.5, 0.25, 0.25}, 0, z), "mixed weights");
  refused(terms(weights_, 3, z), "root out of range");
  refused(terms(weights_, -1, z), "negative root");
  refused(terms({1.0}, 0, z), "one root");
  const fb::Buffers::device<double> short_z(z_host.size() - 1, res_->device());
  refused(terms(weights_, 0, short_z), "short z");
  // At unequal weights z also carries the roots' rotations.
  refused(terms({0.5, 0.3, 0.2}, 0, z), "z without rotations");
}

} // namespace
