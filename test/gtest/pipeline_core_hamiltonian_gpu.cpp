// The CoreHamiltonian step against a direct core_hamiltonian +
// cartesian_to_spherical. The step's contract is test/pipeline/steps_host.cpp's; the
// rest builds H on the device, so needs a GPU.

#include <gtest/gtest.h>

#include "gpu_requirement.h"

import std;
import free_breakfast.core;
import free_breakfast.core.resources;
import free_breakfast.molecule;
import free_breakfast.basis;
import free_breakfast.integrals.one_electron.core_hamiltonian;
import free_breakfast.integrals.one_electron.launch; // kOneElectronMaxL
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.core_hamiltonian;

namespace {

namespace fb = free_breakfast;
using fb::Atom;
using fb::Element;

using CoreHamiltonianInput = fb::View<fb::MoleculeOutput, fb::BasisSetOutput,
                                      fb::SphericalTransformOutput, fb::PrimitivePairClassesOutput>;

std::vector<double> download(fb::DeviceResources &res, const fb::SphericalMatrixAO &m) {
  std::vector<double> host(m.num_elements());
  EXPECT_EQ(wwr::wwrMemcpyAsync(host.data(), m.data(), sizeof(double) * host.size(),
                                wwr::wwrMemcpyDeviceToHost, res.stream()),
            wwr::wwrSuccess);
  EXPECT_EQ(wwr::wwrStreamSynchronize(res.stream()), wwr::wwrSuccess);
  return host;
}

fb::MoleculeOutput water(fb::DeviceResources &res) {
  return {fb::Molecule(res, {Atom{{0.0, 0.0, 0.2217}, Element::O},
                             Atom{{0.0, 1.4309, -0.8867}, Element::H},
                             Atom{{0.0, -1.4309, -0.8867}, Element::H}})};
}

// One normalized shell of momentum l and exponent alpha at centre.
fb::Shell shell(std::int32_t l, double alpha, const fb::Vec<double, 3> &centre) {
  return fb::place(fb::ShellStamp(l, {{alpha, 1.0}}).primitive_normalized(), centre);
}

// H from the CoreHamiltonian step, its inputs built by the steps before it.
fb::Result<fb::CoreHamiltonianOutput> step_core_hamiltonian(fb::DeviceResources &res,
                                                            const fb::MoleculeOutput &molecule,
                                                            const fb::BasisSetOutput &basis) {
  auto transform = fb::SphericalTransformStep{}.run(fb::View<fb::BasisSetOutput>{basis}, res);
  auto pairs = fb::PrimitivePairClassesStep{}.run(fb::View<fb::BasisSetOutput>{basis}, res);
  EXPECT_TRUE(transform.has_value());
  EXPECT_TRUE(pairs.has_value());
  if (!transform || !pairs) {
    return fb::err_gpu("an input step failed");
  }
  return fb::CoreHamiltonianStep{}.run(CoreHamiltonianInput{molecule, basis, *transform, *pairs},
                                       res);
}

TEST(PipelineCoreHamiltonian, MatchesCoreHamiltonianThenCartesianToSpherical) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::MoleculeOutput molecule = water(res);
  const fb::Vec<double, 3> o = molecule.molecule.atoms()[0].position;
  const fb::Vec<double, 3> h1 = molecule.molecule.atoms()[1].position;
  const fb::Vec<double, 3> h2 = molecule.molecule.atoms()[2].position;
  // s, p, d and f on oxygen, an s on each hydrogen:
  // nao_sphe = 1 + 3 + 5 + 7 + 1 + 1.
  const fb::BasisSetOutput basis{
      fb::BasisSet({shell(0, 5.03, o), shell(1, 1.17, o), shell(2, 1.185, o), shell(3, 1.428, o),
                    shell(0, 0.62, h1), shell(0, 0.62, h2)},
                   {0, 4, 5, 6})};

  auto h = step_core_hamiltonian(res, molecule, basis);
  ASSERT_TRUE(h.has_value()) << h.error().message;
  EXPECT_EQ(h->h.n(), 18U);

  auto u = fb::upload_spherical_transform_matrix(res, basis.basis);
  auto classes = fb::primitive_pair_classes(res, basis.basis);
  ASSERT_TRUE(u.has_value());
  ASSERT_TRUE(classes.has_value());
  fb::MatrixAOPair direct(res, basis.basis, *u);
  ASSERT_TRUE(
      fb::core_hamiltonian(res, *classes, molecule.molecule, direct.cartesian()).has_value());
  ASSERT_TRUE(direct.cartesian_to_spherical(res).has_value());

  EXPECT_EQ(download(res, h->h), download(res, direct.spherical()));
}

TEST(PipelineCoreHamiltonian, ClassAboveFIsInvalidConfig) {
  FBF_REQUIRE_GPU();
  fb::DeviceResources res{0};
  const fb::MoleculeOutput molecule = water(res);
  const fb::BasisSetOutput basis{
      fb::BasisSet({shell(fb::kOneElectronMaxL + 1, 1.0, molecule.molecule.atoms()[0].position)})};

  auto h = step_core_hamiltonian(res, molecule, basis);
  ASSERT_FALSE(h.has_value());
  EXPECT_EQ(h.error().kind, fb::ErrorKind::InvalidConfig);
}

} // namespace
