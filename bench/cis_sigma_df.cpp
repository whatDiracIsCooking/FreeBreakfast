// cis::DfSigma's apply against the integral-direct cis::Sigma's, per block of
// k columns, over H2O / cc-pVDZ's RHF (the cis.json geometry), the fit over
// cc-pVDZ-JKFIT. The direct operator runs its columns one after another; the
// fitted one takes the block in one pass, so the per-column ratio is what
// the block buys.
//
//   free_breakfast_cis_sigma_df [reps = 20]
//
// On device 0, where the pipeline runs.
//
// Prints, for k = 1, 4 and 8, the milliseconds per apply of each and the
// direct / fitted ratio. Hold the card (devtools/lib.sh's with_gpu_card)
// while it runs, or another process's kernels are in the number.

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.cis.precondition;
import free_breakfast.cis.sigma;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

namespace {

namespace fb = free_breakfast;
namespace cis = free_breakfast::cis;

using Clock = std::chrono::steady_clock;

// H2O as tools/emit/cis_reference.py places it, in bohr.
constexpr std::array<std::tuple<std::string_view, double, double, double>, 3> kWater{{
    {"O", 0.0, 0.0, 0.2216},
    {"H", 0.0, 1.4309, -0.8867},
    {"H", 0.0, -1.4309, -0.8867},
}};

[[noreturn]] void fail(std::string_view what, const fb::Error &error) {
  std::println(std::cerr, "{}: {}", what, error.message);
  std::exit(1);
}

// Milliseconds per apply of k columns, over reps after two to warm up.
template <class Op>
double time_apply(fb::DeviceResources &res, Op &op, std::int32_t k, std::int32_t reps,
                  const double *x, double *y) {
  for (std::int32_t i = 0; i < 2; ++i) {
    if (const fb::Status s = op.apply(res, k, x, y); !s) {
      fail("apply", s.error());
    }
  }
  (void)res.sync();
  const auto start = Clock::now();
  for (std::int32_t i = 0; i < reps; ++i) {
    (void)op.apply(res, k, x, y);
  }
  (void)res.sync();
  return 1e3 * std::chrono::duration<double>(Clock::now() - start).count() / reps;
}

} // namespace

int main(int argc, char **argv) { // int-type-ok: main's signature is fixed
  const std::int32_t reps = argc > 1 ? std::stoi(argv[1]) : 20;
  const std::filesystem::path dir = std::filesystem::temp_directory_path() / "fb_cis_sigma_df";
  std::filesystem::create_directories(dir / "out");
  {
    std::ofstream xyz{dir / "water.xyz"};
    std::println(xyz, "3\nwater");
    for (const auto &[symbol, x, y, z] : kWater) {
      std::println(xyz, "{} {:.17g} {:.17g} {:.17g}", symbol, x * fb::kAngstromPerBohr<double>,
                   y * fb::kAngstromPerBohr<double>, z * fb::kAngstromPerBohr<double>);
    }
  }
  auto calc =
      fb::parse_config("[molecule]\nxyz = \"water.xyz\"\n[basis_set]\nbasis = \"cc-pvdz\"\n[scf]\n",
                       dir, dir / "out");
  if (!calc) {
    fail("config", calc.error());
  }
  fb::NodeResources node;
  if (const fb::Status ran = calc->run(node); !ran) {
    fail("scf", ran.error());
  }
  fb::DeviceResources &res = node.device(0);
  const auto &molecule = *calc->get<fb::MoleculeOutput>();
  const auto &basis = *calc->get<fb::BasisSetOutput>();
  const auto &transform = *calc->get<fb::SphericalTransformOutput>();
  const auto &pairs = *calc->get<fb::PrimitivePairClassesOutput>();
  const auto &direct = std::get<fb::IntegralDirect>(calc->get<fb::EriOutput>()->eri);
  const fb::ScfOutput &scf = *calc->get<fb::ScfOutput>();
  auto fitted = fb::EriStep{.mode = fb::EriMode::DensityFitting, .aux_basis = "cc-pvdz-jkfit"}.run(
      fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput>{
          molecule, basis, pairs},
      res);
  if (!fitted) {
    fail("fit", fitted.error());
  }
  const auto &fit = std::get<fb::DensityFitted>(fitted->eri);

  const std::int32_t n_occ = scf.n_alpha;
  const auto n_virt = static_cast<std::int32_t>(scf.c.n()) - n_occ;
  const std::int32_t n_ov = n_occ * n_virt;
  const auto n = static_cast<std::size_t>(n_ov);
  constexpr std::array<std::int32_t, 3> kBlocks{1, 4, 8};
  const std::int32_t widest = kBlocks.back();

  fb::Buffers::device<double> energies(scf.mo_energies.size(), res.device());
  (void)wwr::wwrMemcpyAsync(energies.data(), scf.mo_energies.data(),
                            scf.mo_energies.size() * sizeof(double), wwr::wwrMemcpyHostToDevice,
                            res.stream());
  fb::Buffers::device<double> delta(n, res.device());
  if (const fb::Status s = cis::diagonal(res, n_occ, n_virt, energies.data(), delta.data()); !s) {
    fail("diagonal", s.error());
  }
  auto sigma = cis::Sigma::create(res, basis.basis, transform.u, direct.j_classes, direct.k_classes,
                                  scf.c, n_occ, delta.data(), cis::Spin::Singlet);
  if (!sigma) {
    fail("Sigma::create", sigma.error());
  }
  auto df_sigma = cis::DfSigma::create(res, fit.b, fit.naux, scf.c, n_occ, delta.data(),
                                       cis::Spin::Singlet, widest);
  if (!df_sigma) {
    fail("DfSigma::create", df_sigma.error());
  }

  std::mt19937 gen{17};
  std::uniform_real_distribution<double> dist{-1.0, 1.0};
  std::vector<double> host(n * static_cast<std::size_t>(widest));
  for (double &v : host) {
    v = dist(gen);
  }
  fb::Buffers::device<double> x(host.size(), res.device());
  fb::Buffers::device<double> y(host.size(), res.device());
  (void)wwr::wwrMemcpyAsync(x.data(), host.data(), host.size() * sizeof(double),
                            wwr::wwrMemcpyHostToDevice, res.stream());
  (void)res.sync();

  std::println("H2O / cc-pVDZ singlets: nao {}, n_occ {}, n_ov {}, naux {} (cc-pVDZ-JKFIT), "
               "{} reps",
               scf.c.n(), n_occ, n_ov, fit.naux, reps);
  std::println("{:>3} {:>12} {:>12} {:>14}", "k", "direct ms", "DF ms", "direct / DF");
  for (const std::int32_t k : kBlocks) {
    const double t_direct = time_apply(res, *sigma, k, reps, x.data(), y.data());
    const double t_df = time_apply(res, *df_sigma, k, reps, x.data(), y.data());
    std::println("{:>3} {:>12.4f} {:>12.4f} {:>14.1f}", k, t_direct, t_df, t_direct / t_df);
  }
  std::filesystem::remove_all(dir);
  return 0;
}
