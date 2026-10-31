/// @file
/// @brief What the SC and PC class energies and the SC reverse pass share: the
/// slab walk's helpers, the class metrics and Dyall Hamiltonians (make_*),
/// Sijrs, the active operands every class reads, and einsum's adjoint.
///
/// An internal partition (`module`, not `export module`): energy.cpp, pc.cpp
/// and reverse.cpp import it, nothing reaches an importer of
/// free_breakfast.nevpt2.energy. It only declares; every definition is
/// energy.cpp's, except upload (pc.cpp's) and einsum_adjoint (reverse.cpp's),
/// so no bridge header reaches a partition's global module fragment. See
/// docs/style.md, "Module-private code".

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

// Norms at or below this are an empty perturber space and contribute no
// energy: pyscf.mrpt.nevpt2's NUMERICAL_ZERO. The reverse pass gives such a
// perturber no multipliers, for the same reason.
constexpr double kNumericalZero = 1e-14;

EinsumOperand column_major(const double *data, std::vector<int64_t> dims);
Result<Buffers::device<double>> device_zeros(DeviceResources &res, int64_t count);
Result<std::vector<double>> download(DeviceResources &res, const double *d, int64_t count);
Result<Buffers::device<double>> upload(DeviceResources &res, const std::vector<double> &host);
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
// Srs's rm3, from its rm2; make_a7 builds it internally. Declared for the
// reverse sweep, which re-reads it.
Result<Buffers::device<double>> make_rm3(DeviceResources &res, const EinsumOperand &dm2,
                                         const EinsumOperand &dm3, const EinsumOperand &rm2,
                                         const EinsumOperand &delta);
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
// Sr's and Si's one-body Dyall blocks: SC's alone (PC reads S₁ h1 instead),
// declared for the reverse pass.
Result<Buffers::device<double>> make_a17(DeviceResources &res, const EinsumOperand &h1e_in,
                                         const EinsumOperand &h2e, const EinsumOperand &dm2,
                                         const EinsumOperand &dm3);
Result<Buffers::device<double>> make_a19(DeviceResources &res, const EinsumOperand &h1e_in,
                                         const EinsumOperand &h2e, const EinsumOperand &dm1,
                                         const EinsumOperand &dm2);
Result<Buffers::device<double>> make_a23(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm1,
                                         const EinsumOperand &dm2, const EinsumOperand &dm3);
Result<Buffers::device<double>> make_a25(DeviceResources &res, const EinsumOperand &h1e,
                                         const EinsumOperand &h2e, const EinsumOperand &dm1,
                                         const EinsumOperand &dm2);

/// @brief The adjoint of `einsum(subscripts, operands) -> out` (out_dims,
/// alpha): for every operand k whose @p bars[k] is not null,
/// bars[k] += alpha d<out_bar, einsum(...)>/d(operand k), queued on
/// res.stream().
///
/// @p subscripts, @p operands, @p out_dims and @p alpha are the forward
/// call's, unchanged: the bar on operand k is the same einsum with the output
/// and operand k swapping roles, so no derivative is written by hand. Each
/// bar has its operand's layout (its dims and strides, row-major when the
/// strides are empty), and a bar may be shared by several operands (a slab
/// read twice), each adding its own term. @p out_bar is packed row-major over
/// @p out_dims, as einsum's output is.
///
/// - A row-major or column-major operand's bar is written in place, the
///   output labels ordered so that einsum's packed row-major output is the
///   operand's layout.
/// - An operand that repeats a label (`jbij`, `mjjn`, `raai`) takes its bar
///   over its distinct labels into workspace, then scatters it onto the
///   diagonal as an einsum against @p delta, the identity over the repeated
///   label's extent (one delta per repetition, at most two). @p delta is read
///   only then.
/// - A two-axis view of a larger column-major array (strides {1, ld}: a run
///   of h1e_v's rows) takes its bar packed into workspace, then one in-place
///   geam adds it at the view.
///
/// Scratch is res.device_workspace()'s, so no argument may point into it,
/// and a caller holds no workspace region across the call.
///
/// @return Success; InvalidConfig for subscripts that do not parse into one
///         group per operand and bar, a label only operand k carries and only
///         once (its bar would be a broadcast), a layout outside the three
///         above, or einsum's own refusals; or the launch status.
Status einsum_adjoint(DeviceResources &res, std::string_view subscripts,
                      const std::vector<EinsumOperand> &operands, const std::vector<double *> &bars,
                      const double *out_bar, const std::vector<int64_t> &out_dims,
                      const EinsumOperand &delta, double alpha);

/// @brief Sijrs: {norm, energy}. It reads no RDM, so it is PC's too.
Result<std::pair<double, double>> energy_sijrs(DeviceResources &res, const MoBlocks &blocks,
                                               const EnergyInputs &in, int32_t batch);

} // namespace free_breakfast::nevpt2
