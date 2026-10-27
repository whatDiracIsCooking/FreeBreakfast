/// @file
/// @brief The Cartesian electron-nuclear potential (nuclear attraction) matrix
/// V on the GPU, over the v_{la}{lb} kernels FelizJueves generates.
///
/// The records come from free_breakfast.integrals.mmd and the nuclei from
/// free_breakfast.molecule; V is one kernel launch per shell-pair class into a
/// @c CartesianMatrixAO.

export module free_breakfast.integrals.one_electron.potential_en;

import std;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.integrals.mmd;
import free_breakfast.molecule;
import free_breakfast.integrals.one_electron.launch;
import free_breakfast.recipes.matrix_ao;

export namespace free_breakfast {

/// @brief The Cartesian nuclear attraction
/// V_mn = -Sum_C Z_C <m| 1/|r - C| |n> into @p v.
///
/// Zeroes @p v, then @ref add_nuclear_attraction. Point nuclei.
///
/// @return As @ref add_nuclear_attraction; a GpuRuntime error from the memset
///         too.
[[nodiscard]] Status nuclear_attraction(DeviceResources &res,
                                        std::span<const PrimitivePairClass> classes,
                                        const Molecule &molecule, CartesianMatrixAO &v);

/// @brief Add the Cartesian nuclear attraction into @p m, without zeroing it
/// first.
///
/// Launches the generated kernel of each class, which loops over the nuclei of
/// @p molecule serially inside each record and adds its contributions at (m, n)
/// and (n, m).
/// This is how @c core_hamiltonian forms T + V in place: @c kinetic zeroes the
/// matrix and writes T, and this adds V on top. @p classes must come from the
/// basis @p m was made for: a record's AO offsets are not checked against
/// m.n().
///
/// Queued on res.stream(), each class on its own stream, forked from it and
/// joined back into it (@c launch_each), with no host
/// synchronization. The device res.dev_idx() names is made current for the
/// call and the previous one restored.
///
/// @return Success; InvalidConfig if @p res is not on @p m's, @p classes' or
///         @p molecule's device, if a class is above f (or not la <= lb), or if
///         m.n() is too large for the kernels' 32-bit index (m.n()^2 above
///         INT32_MAX, about 46k AOs); or the GpuRuntime error of the first
///         fork, launch or join that failed, which leaves @p m unspecified.
[[nodiscard]] Status add_nuclear_attraction(DeviceResources &res,
                                            std::span<const PrimitivePairClass> classes,
                                            const Molecule &molecule, CartesianMatrixAO &m);

} // namespace free_breakfast
