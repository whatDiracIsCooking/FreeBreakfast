/// @file
/// @brief The restricted SCF loop, RHF or ROHF, written against a Fock
/// builder: the Roothaan effective Fock matrix of an open shell, the energy,
/// the residual, convergence, DIIS, the eigensolve, the densities from the
/// occupied orbitals and the Molden write.
///
/// An internal partition (`module`, not `export module`): only this module's
/// implementation units import it, so nothing here reaches an importer of
/// free_breakfast.pipeline.scf. See docs/style.md, "Module-private code".

module free_breakfast.pipeline.scf:loop;

import std;
import free_breakfast.core.types;
import free_breakfast.core.constants;
import free_breakfast.core.error_handling;
import free_breakfast.core.log;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.recipes.matrix_ao;
import free_breakfast.recipes.blas;
import free_breakfast.recipes.diagonalize;
import free_breakfast.recipes.diis;
import free_breakfast.io.molden;
import :output;

#include "diagnostics.h"

namespace free_breakfast {

// Everything the loop holds across iterations, allocated once on res's device.
// The Fock builder's own state is the builder's.
struct ScfState {
  SphericalMatrixAO d;      // the density, the iterate; spin-summed, Da + Db
  SphericalMatrixAO f;      // the Fock matrix of d, as built; an open shell's Roothaan one
  SphericalMatrixAO f_diis; // DIIS's extrapolation of f
  SphericalMatrixAO e;      // F D S - S D F
  SphericalMatrixAO tmp;    // F D S; then X^T e X; then F' and its eigenvectors
  SphericalMatrixAO work;
  SphericalMatrixAO c;
  Buffers::device<double> eps;
};

// The open shell's iterate beside ScfState's, for n_alpha > n_beta: the spin
// densities, the spin Fock matrices the builder makes, of which ScfState's f is
// the Roothaan effective one, and the identity Pv = I - Da S is made from.
// None of it is allocated for a closed shell.
struct OpenShellState {
  SphericalMatrixAO da; // C[:, :n_alpha] C[:, :n_alpha]^T
  SphericalMatrixAO db; // C[:, :n_beta] C[:, :n_beta]^T
  SphericalMatrixAO fa; // H + J(Da + Db) - K(Da)
  SphericalMatrixAO fb; // H + J(Da + Db) - K(Db)
  Buffers::device<double> identity;
};

// E_elec = tr(D (H + F)) / 2, or for an open shell
// tr((Da + Db) H + Da Fa + Db Fb) / 2. Two dots, or three, into host scalars,
// each of which synchronizes the stream.
[[nodiscard]] Result<double> electronic_energy(DeviceResources &res, const ScfState &st,
                                               const OpenShellState *open,
                                               const SphericalMatrixAO &h) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto len = static_cast<int32_t>(h.num_elements());
  double dh = 0.0;
  double df = 0.0;
  // D, H and F are symmetric, so tr(D M) is their elementwise dot.
  FBF_TRY(gpu_status(wwr::dot<double, int32_t>(blas, len, st.d.data(), 1, h.data(), 1, &dh)));
  if (open == nullptr) {
    FBF_TRY(gpu_status(wwr::dot<double, int32_t>(blas, len, st.d.data(), 1, st.f.data(), 1, &df)));
    return 0.5 * (dh + df);
  }
  double dbfb = 0.0;
  FBF_TRY(gpu_status(
      wwr::dot<double, int32_t>(blas, len, open->da.data(), 1, open->fa.data(), 1, &df)));
  FBF_TRY(gpu_status(
      wwr::dot<double, int32_t>(blas, len, open->db.data(), 1, open->fb.data(), 1, &dbfb)));
  return 0.5 * (dh + df + dbfb);
}

// The Roothaan effective Fock matrix of an open shell into st.f, in PySCF's
// get_roothaan_fock convention, so that its eigenvalues are PySCF's ROHF
// orbital energies: with Fc = (Fa + Fb)/2 and the core, open and virtual
// projectors Pc = Db S, Po = (Da - Db) S and Pv = I - Da S,
//
//   A = Pc^T Fc Pc / 2 + Po^T Fc Po / 2 + Pv^T Fc Pv / 2
//     + Po^T Fb Pc + Po^T Fa Pv + Pv^T Fc Pc,     F = A + A^T.
//
// Fc on every diagonal block, Fb between core and open, Fa between open and
// virtual and Fc between core and virtual. It is made from the densities, not
// the orbitals, so the first build, whose densities are the guess's, needs no
// C. The projectors are this call's scratch; st.work is the products'.
[[nodiscard]] Status roothaan_fock(DeviceResources &res, ScfState &st, const OpenShellState &open,
                                   const SphericalMatrixAO &s) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto n = static_cast<int32_t>(s.n());
  const size_t nn = s.num_elements();
  const auto [fc, pc, po, pv, a] = res.device_regions<double>(nn, nn, nn, nn, nn);
  double *const work = st.work.data();
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n,
                                                &kHalf<double>, open.fa.data(), n, &kHalf<double>,
                                                open.fb.data(), n, fc, n)));
  FBF_TRY(gpu_status(gemm(blas, n, &kOne<double>, open.db.data(), s.data(), &kZero<double>, pc)));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n,
                                                &kOne<double>, open.da.data(), n, &kNegOne<double>,
                                                open.db.data(), n, work, n)));
  FBF_TRY(gpu_status(gemm(blas, n, &kOne<double>, work, s.data(), &kZero<double>, po)));
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(pv, open.identity.data(), nn * sizeof(double),
                                         wwr::wwrMemcpyDeviceToDevice, res.stream())));
  FBF_TRY(gpu_status(gemm(blas, n, &kNegOne<double>, open.da.data(), s.data(), &kOne<double>, pv)));
  // Fc Pc: A = Pc^T (Fc Pc) / 2 + Pv^T (Fc Pc).
  FBF_TRY(gpu_status(gemm(blas, n, &kOne<double>, fc, pc, &kZero<double>, work)));
  FBF_TRY(
      gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n, &kHalf<double>, pc, work, &kZero<double>, a)));
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n, &kOne<double>, pv, work, &kOne<double>, a)));
  // Fc Pv: A += Pv^T (Fc Pv) / 2.
  FBF_TRY(gpu_status(gemm(blas, n, &kOne<double>, fc, pv, &kZero<double>, work)));
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n, &kHalf<double>, pv, work, &kOne<double>, a)));
  // Every term with Po on the left: A += Po^T (Fc Po / 2 + Fb Pc + Fa Pv).
  FBF_TRY(gpu_status(gemm(blas, n, &kHalf<double>, fc, po, &kZero<double>, work)));
  FBF_TRY(gpu_status(gemm(blas, n, &kOne<double>, open.fb.data(), pc, &kOne<double>, work)));
  FBF_TRY(gpu_status(gemm(blas, n, &kOne<double>, open.fa.data(), pv, &kOne<double>, work)));
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_T>(blas, n, &kOne<double>, po, work, &kOne<double>, a)));
  // A + A^T is symmetric bit for bit, which the commutator's one product needs.
  return gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n, n,
                                               &kOne<double>, a, n, &kOne<double>, a, n,
                                               st.f.data(), n));
}

// The DIIS residual X^T (F D S - S D F) X into st.tmp, and its Frobenius norm.
// S D F = (F D S)^T for symmetric F, D and S, so one product and a geam.
[[nodiscard]] Result<double> commutator(DeviceResources &res, ScfState &st,
                                        const SphericalMatrixAO &s, const SphericalMatrixAO &x) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto n = static_cast<int32_t>(s.n());
  FBF_TRY(gpu_status(
      gemm(blas, n, &kOne<double>, st.f.data(), st.d.data(), &kZero<double>, st.work.data())));
  FBF_TRY(gpu_status(
      gemm(blas, n, &kOne<double>, st.work.data(), s.data(), &kZero<double>, st.tmp.data())));
  FBF_TRY(gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T, n, n,
                                                &kOne<double>, st.tmp.data(), n, &kNegOne<double>,
                                                st.tmp.data(), n, st.e.data(), n)));
  FBF_TRY(gpu_status(gemmwich(blas, n, x.data(), st.e.data(), st.work.data(), st.tmp.data())));
  double norm = 0.0;
  FBF_TRY(gpu_status(wwr::nrm2<double, int32_t>(blas, static_cast<int32_t>(st.tmp.num_elements()),
                                                st.tmp.data(), 1, &norm)));
  return norm;
}

// Solve F C = S C eps through X, @p fock being F: F' = X^T F X = C' eps C'^T,
// C = X C'. Then D = 2 C_occ C_occ^T from the n_alpha lowest columns; or, for
// an open shell, Da and Db from the n_alpha and n_beta lowest, and D = Da + Db.
[[nodiscard]] Status solve_fock(DeviceResources &res, ScfState &st, OpenShellState *open,
                                const SphericalMatrixAO &fock, const SphericalMatrixAO &x,
                                int32_t n_alpha, int32_t n_beta) {
  const wwr::wwrblasHandle_t blas = res.blas().get();
  const auto n = static_cast<int32_t>(x.n());
  FBF_TRY(gpu_status(gemmwich(blas, n, x.data(), fock.data(), st.work.data(), st.tmp.data())));
  // syevd's work is dead once it returns, so diagonalize takes it from the
  // shared workspace, not the loop's buffers.
  FBF_TRY(diagonalize<double>(res, n, st.tmp.data(), st.eps.data()));
  FBF_TRY(gpu_status(
      gemm(blas, n, &kOne<double>, x.data(), st.tmp.data(), &kZero<double>, st.c.data())));
  if (open == nullptr) {
    return gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(
        blas, n, n, n_alpha, &kTwo<double>, st.c.data(), st.c.data(), &kZero<double>, st.d.data()));
  }
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(blas, n, n, n_alpha, &kOne<double>,
                                                                st.c.data(), st.c.data(),
                                                                &kZero<double>, open->da.data())));
  FBF_TRY(gpu_status(gemm<wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_T>(blas, n, n, n_beta, &kOne<double>,
                                                                st.c.data(), st.c.data(),
                                                                &kZero<double>, open->db.data())));
  return gpu_status(wwr::geam<double, int32_t>(blas, wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, n, n,
                                               &kOne<double>, open->da.data(), n, &kOne<double>,
                                               open->db.data(), n, st.d.data(), n));
}

// The converged orbitals to @p path as Molden: C downloaded, every MO written,
// the n_beta lowest doubly occupied and the next n_alpha - n_beta singly.
[[nodiscard]] Status write_scf_molden(DeviceResources &res, const std::filesystem::path &path,
                                      const Molecule &molecule, const BasisSet &basis,
                                      const SphericalMatrixAO &c,
                                      const std::vector<double> &mo_energies, int32_t n_alpha,
                                      int32_t n_beta) {
  std::vector<double> host(c.num_elements());
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(host.data(), c.data(), host.size() * sizeof(double),
                                         wwr::wwrMemcpyDeviceToHost, res.stream())));
  FBF_TRY(res.sync());
  std::vector<double> occupations(mo_energies.size(), 0.0);
  std::fill_n(occupations.begin(), n_alpha, 1.0);
  std::fill_n(occupations.begin(), n_beta, 2.0);
  return write_molden(path, molecule.atoms(), basis, host, mo_energies, occupations);
}

// Restricted Hartree-Fock to self-consistency from the host density @p guess,
// rescaled to the molecule's electron count, with Pulay DIIS on the Fock
// matrix; on convergence the orbitals are written to @p molden if it is not
// empty. Closed-shell RHF when @p n_alpha == @p n_beta; otherwise ROHF, over
// one set of orbitals whose n_beta lowest are doubly occupied and the next
// n_alpha - n_beta singly. The options and the inputs arrive checked by the
// step.
//
// An open shell's iterate is Da and Db, and the Fock matrices Fa and Fb the
// builder makes from them; the loop assembles the Roothaan effective F from
// those (roothaan_fock), and from there runs as for a closed shell: DIIS and
// the commutator on (F, Da + Db), F's eigenvectors the orbitals. The guess,
// spin-summed, is split D/2 per spin. A closed shell allocates none of it and
// skips every step of it, so its path is the RHF loop's, operation for
// operation.
//
// @p builder makes the Fock matrix and owns how. Two members:
//
// - `Result<bool> build(res, d, c, h, f, open)` makes F = H + G(D) into f,
//   where c is the orbitals D came from, or null on the first build, whose D
//   is the guess. f holds the previous build's F, untouched by the loop, so a
//   builder may add to it. For an open shell f is open->fa, and the builder
//   makes Fb into open->fb, from D = Da + Db and open's da and db, which are
//   the densities D is the sum of; a closed shell's open is null. Returns
//   whether this F is exact for D: only such a build may be taken as
//   converged.
// - `bool reset_diis(double norm)` is told the commutator norm of the F just
//   built, and returns whether DIIS's history is stale and is to be dropped
//   after this iteration's extrapolation.
template <class FockBuilder>
[[nodiscard]] Result<ScfOutput>
scf_loop(DeviceResources &res, FockBuilder &builder, const Molecule &molecule,
         const BasisSet &basis, const SphericalMatrixAO &s, const SphericalMatrixAO &x,
         const SphericalMatrixAO &h, const std::vector<double> &guess, int32_t n_alpha,
         int32_t n_beta, int32_t max_iterations, double energy_tolerance,
         double commutator_tolerance, int32_t diis_history, const std::filesystem::path &molden) {
  const uint32_t n = basis.nao_sphe();
  const int32_t electrons = molecule.electron_count();
  ScfState st{
      .d = SphericalMatrixAO(res, basis),
      .f = SphericalMatrixAO(res, basis),
      .f_diis = SphericalMatrixAO(res, basis),
      .e = SphericalMatrixAO(res, basis),
      .tmp = SphericalMatrixAO(res, basis),
      .work = SphericalMatrixAO(res, basis),
      .c = SphericalMatrixAO(res, basis),
      .eps = Buffers::device<double>(n, res.device()),
  };
  Diis diis(res, std::size_t{n} * n, static_cast<size_t>(diis_history));

  // The guess is the loop's to upload, and to rescale: SAD places sum Z
  // electrons whatever the charge, and the SCF needs the molecule's own count.
  FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(st.d.data(), guess.data(), guess.size() * sizeof(double),
                                         wwr::wwrMemcpyHostToDevice, res.stream())));
  if (const int32_t z = molecule.total_nuclear_charge(); z != electrons && z > 0) {
    const double scale = static_cast<double>(electrons) / static_cast<double>(z);
    FBF_TRY(gpu_status(wwr::scal<double, int32_t>(
        res.blas().get(), static_cast<int32_t>(guess.size()), &scale, st.d.data(), 1)));
  }

  std::optional<OpenShellState> open_shell;
  if (n_alpha != n_beta) {
    open_shell = OpenShellState{
        .da = SphericalMatrixAO(res, basis),
        .db = SphericalMatrixAO(res, basis),
        .fa = SphericalMatrixAO(res, basis),
        .fb = SphericalMatrixAO(res, basis),
        .identity = Buffers::device<double>(std::size_t{n} * n, res.device()),
    };
    std::vector<double> identity(std::size_t{n} * n, 0.0);
    for (std::size_t i = 0; i < n; ++i) {
      identity[i * (n + 1)] = 1.0;
    }
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(open_shell->identity.data(), identity.data(),
                                           identity.size() * sizeof(double),
                                           wwr::wwrMemcpyHostToDevice, res.stream())));
    // The guess carries no spin: Da = Db = D/2, as PySCF's ROHF splits it.
    const auto ni = static_cast<int32_t>(n);
    FBF_TRY(gpu_status(wwr::geam<double, int32_t>(
        res.blas().get(), wwr::WWRBLAS_OP_N, wwr::WWRBLAS_OP_N, ni, ni, &kHalf<double>, st.d.data(),
        ni, &kZero<double>, st.d.data(), ni, open_shell->da.data(), ni)));
    FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(open_shell->db.data(), open_shell->da.data(),
                                           st.d.num_elements() * sizeof(double),
                                           wwr::wwrMemcpyDeviceToDevice, res.stream())));
    // identity is read by the upload above.
    FBF_TRY(res.sync());
  }
  OpenShellState *const open = open_shell.has_value() ? &*open_shell : nullptr;

  const double nuclear = nuclear_repulsion_energy(molecule);
  double previous = 0.0;
  for (int32_t iteration = 1; iteration <= max_iterations; ++iteration) {
    const bool full = FBF_TRY(builder.build(res, st.d, iteration == 1 ? nullptr : &st.c, h,
                                            open != nullptr ? open->fa : st.f, open));
    if (open != nullptr) {
      FBF_TRY(roothaan_fock(res, st, *open, s));
    }
    const double energy = nuclear + FBF_TRY(electronic_energy(res, st, open, h));
    const double norm = FBF_TRY(commutator(res, st, s, x));
    const double delta = energy - previous;
    previous = energy;
    FBF_LOG_INFO("scf: iteration {:3} {}  E = {:.12f}  dE = {:+.3e}  |[F,D]| = {:.3e}", iteration,
                 full ? "full" : "incr", energy, delta, norm);

    const bool converged =
        full && iteration > 1 && std::abs(delta) < energy_tolerance && norm < commutator_tolerance;
    const bool reset = builder.reset_diis(norm);
    if (!converged) {
      // With fewer than two pairs DIIS leaves its output alone: F as built.
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(st.f_diis.data(), st.f.data(),
                                             st.f.num_elements() * sizeof(double),
                                             wwr::wwrMemcpyDeviceToDevice, res.stream())));
      FBF_TRY(diis.push_and_extrapolate(res, st.f.data(), st.tmp.data(), st.f_diis.data()));
      if (reset) {
        diis.reset();
      }
    }
    // Converged: the canonical orbitals of the Fock matrix as built, not of an
    // extrapolation, and the density they give.
    FBF_TRY(solve_fock(res, st, open, converged ? st.f : st.f_diis, x, n_alpha, n_beta));
    if (converged) {
      std::vector<double> mo_energies(n);
      FBF_TRY(gpu_status(wwr::wwrMemcpyAsync(mo_energies.data(), st.eps.data(), n * sizeof(double),
                                             wwr::wwrMemcpyDeviceToHost, res.stream())));
      FBF_TRY(res.sync());
      FBF_LOG_INFO("scf: converged in {} iterations, E = {:.12f}", iteration, energy);
      if (!molden.empty()) {
        FBF_TRY(write_scf_molden(res, molden, molecule, basis, st.c, mo_energies, n_alpha, n_beta));
        FBF_LOG_INFO("scf: wrote {}", molden.string());
      }
      return ScfOutput{.energy = energy,
                       .iterations = iteration,
                       .n_alpha = n_alpha,
                       .n_beta = n_beta,
                       .mo_energies = std::move(mo_energies),
                       .c = std::move(st.c),
                       .d = std::move(st.d)};
    }
  }
  return err_max_iterations_exceeded(
      "scf: not converged in {} iterations (energy_tolerance {:g}, commutator_tolerance {:g})",
      max_iterations, energy_tolerance, commutator_tolerance);
}

} // namespace free_breakfast
