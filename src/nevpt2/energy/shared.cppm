/// @file
/// @brief What the SC and PC class energies share: the slab walk's helpers,
/// the class metrics and Dyall Hamiltonians (make_*), Sijrs, and the active
/// operands every class reads.
///
/// An internal partition (`module`, not `export module`): energy.cpp and
/// pc.cpp import it, nothing reaches an importer of
/// free_breakfast.nevpt2.energy. It only declares; every definition is
/// energy.cpp's, where the launchers' bridge header is included, so no bridge
/// reaches a partition's global module fragment. See docs/style.md,
/// "Module-private code".

module free_breakfast.nevpt2.energy:shared;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.wwr;
import free_breakfast.core.resources;
import free_breakfast.recipes.einsum;
import free_breakfast.nevpt2.density_fitting;
import free_breakfast.nevpt2.energy; // EnergyInputs

namespace free_breakfast::nevpt2 {

/// @brief Everything a class reads of the active space, as einsum operands:
/// the integrals column-major, the RDMs row-major, delta the n x n identity
/// (owned here), and the orbital energies the host finish reads.
struct ActiveOperands {
  EinsumOperand h1e, h2e, dm1, dm2, dm3, f3ca, f3ac, delta;
  Buffers::device<double> delta_buf;
  std::vector<double> e_core, e_virt;
};

/// @brief The operands of @p in over @p blocks' sizes. Synchronizes once,
/// reading the orbital energies back.
Result<ActiveOperands> active_operands(DeviceResources &res, const MoBlocks &blocks,
                                       const EnergyInputs &in);

EinsumOperand column_major(const double *data, std::vector<int64_t> dims);
Result<Buffers::device<double>> device_zeros(DeviceResources &res, int64_t count);
Result<std::vector<double>> download(DeviceResources &res, const double *d, int64_t count);
int32_t batch_step(int32_t extent, int32_t batch);
std::vector<int64_t> slab_dims(const MoBlocks &blocks, ExtBlock block, int32_t b0, int32_t b1);
Buffers::device<double> slab_buffer(DeviceResources &res, const MoBlocks &blocks, ExtBlock block,
                                    int32_t step);
int32_t walk_extent(const MoBlocks &blocks, ExtBlock block);

Result<Buffers::device<double>> make_a16(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm3,
                                         const EinsumOperand &f3ac, const EinsumOperand &f3ca);
Result<Buffers::device<double>> make_a22(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm2,
                                         const EinsumOperand &dm3, const EinsumOperand &f3ac,
                                         const EinsumOperand &f3ca);
Result<Buffers::device<double>> make_dm3_h(DeviceResources &res, const EinsumOperand &dm2,
                                           const EinsumOperand &dm3, const EinsumOperand &delta);
Result<Buffers::device<double>> make_dm2_h(DeviceResources &res, const EinsumOperand &dm1,
                                           const EinsumOperand &dm2, const EinsumOperand &delta);
Result<Buffers::device<double>> make_hdm1(DeviceResources &res, const EinsumOperand &dm1,
                                          const EinsumOperand &delta);
Result<Buffers::device<double>> make_hdm2(DeviceResources &res, const EinsumOperand &dm1,
                                          const EinsumOperand &dm2, const EinsumOperand &delta);
Result<Buffers::device<double>> make_hdm3(DeviceResources &res, const EinsumOperand &dm1,
                                          const EinsumOperand &dm2, const EinsumOperand &dm3,
                                          const EinsumOperand &hdm2, const EinsumOperand &delta);
Result<Buffers::device<double>> make_a3(DeviceResources &res, const EinsumOperand &h1e,
                                        const EinsumOperand &h2e, const EinsumOperand &dm1,
                                        const EinsumOperand &dm2, const EinsumOperand &hdm1,
                                        const EinsumOperand &delta);
Result<Buffers::device<double>> make_k27(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm1,
                                         const EinsumOperand &dm2);
Result<std::pair<Buffers::device<double>, Buffers::device<double>>>
make_a7(DeviceResources &res, const EinsumOperand &h1e, const EinsumOperand &h2e,
        const EinsumOperand &dm1, const EinsumOperand &dm2, const EinsumOperand &dm3,
        const EinsumOperand &delta);
Result<Buffers::device<double>> make_a9(DeviceResources &res, const EinsumOperand &h1e,
                                        const EinsumOperand &h2e, const EinsumOperand &hdm2,
                                        const EinsumOperand &hdm3);
Result<Buffers::device<double>> make_a12(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm2,
                                         const EinsumOperand &dm3);
Result<Buffers::device<double>> make_a13(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm1,
                                         const EinsumOperand &dm2, const EinsumOperand &dm3,
                                         const EinsumOperand &delta);

/// @brief Sijrs: {norm, energy}. It reads no RDM, so it is PC's too.
Result<std::pair<double, double>> energy_sijrs(DeviceResources &res, const MoBlocks &blocks,
                                               const EnergyInputs &in, int32_t batch);

} // namespace free_breakfast::nevpt2
