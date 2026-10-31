/// @file
/// @brief The nuclear derivatives of the density-fitting integrals on the
/// GPU, by McMurchie-Davidson over kernels FelizJueves generates: the
/// gradients, SUM_PQ W_PQ d(P|Q)/dR and SUM_mnP Gam_mnP d(mn|P)/dR (over
/// gmetric<lP, lQ> and geri3<la, lb, lP>); the derivative integrals
/// themselves, d(P|Q)/dA and (nabla m n|P) (over ipmetric<lP, lQ> and
/// xeri3<la, lb, lP>); and over those, the density-fitted J and K
/// differentiated per nuclear coordinate at a fixed density, the
/// two-electron half of the skeleton Fock derivatives an RHF Hessian needs.
///
/// The gradients take their matrix in the spherical layout the energy builds
/// write (metric's, three_center's), convert it to the Cartesian one the
/// kernels contract against, and return the n_atom x 3 gradient on the host.
/// The derivative integrals come back spherical, on the device. The README
/// has the staging, the transforms and the J/K algebra.

export module free_breakfast.integrals.eri.density_fitting.gradient;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.resources;
import free_breakfast.basis;
import free_breakfast.integrals.mmd;

export namespace free_breakfast {

/// @brief SUM_PQ W_PQ d(P|Q)/dR_{A,k} of the auxiliary basis @p aux, for
/// any spherical naux x naux @p w on the device (column-major; no symmetry
/// assumed): G[A][k] at 3 A + k, one row per atom of @p aux.
///
/// @p aux must be made by @c place, so it knows which atom each shell sits on
/// (aux.natoms() rows). W is made Cartesian as U_aux W U_aux^T, then the
/// metric's records and aux shell pairs are staged as @c metric stages them,
/// each pair with its two atoms, and one gmetric<lP, lQ> launch per class that
/// has a pair adds into a zeroed device G, which is downloaded. The kernels'
/// atomic adds make a run reproducible to round-off, not bitwise.
///
/// Queued on res.stream(), and synchronized: the call returns when G is on the
/// host. The scratch comes from res.device_workspace(), so @p w must not point
/// into it. The device res.dev_idx() names is made current for the call and
/// the previous one restored.
///
/// @return The gradient (all zeros for an empty @p aux); InvalidConfig if
///         @p aux was not made by @c place, if an aux shell is above i, or if
///         aux.nao_cart() is too large for the kernels' 32-bit index; or the
///         GpuRuntime error of the first copy, launch or BLAS call that
///         failed.
[[nodiscard]] Result<std::vector<double>> metric_gradient(DeviceResources &res, const BasisSet &aux,
                                                          const double *w);

/// @brief SUM_mnP Gam_mnP d(mn|P)/dR_{A,k} of the orbital basis @p basis and
/// the auxiliary basis @p aux, for any spherical @p gamma on the device in
/// @c three_center's layout, element (m, n, P) at P + naux (n + nao m), P
/// fastest (no symmetry assumed): G[A][k] at 3 A + k, one row per atom.
///
/// @p classes must be @c primitive_pair_classes(res, basis), finished, and
/// both bases made by @c place over the same molecule. Gam is made Cartesian
/// on all three indices (one gemm with U_aux, a strided-batched gemm and a
/// gemm with U); mmd's records are regrouped by shell pair and staged with
/// their two exponents, the aux records and the cells as @c three_center
/// stages them, each cell with its three atoms; then one geri3<la, lb, lP>
/// launch per class that has a cell adds into a zeroed device G, which is
/// downloaded. The atomic adds make a run reproducible to round-off.
///
/// **Refused up front, not split.** The Cartesian Gam and its transform's
/// intermediate are scratch from res.device_workspace(); if they and the
/// staging need more bytes than the device has free (beyond what the
/// workspace already holds), the call is Unsupported, naming the bytes,
/// before anything is allocated.
///
/// Queued on res.stream(), and synchronized: the call returns when G is on the
/// host. @p gamma must not point into res.device_workspace(). The device
/// res.dev_idx() names is made current for the call and the previous one
/// restored.
///
/// @return The gradient (all zeros if either basis is empty); Unsupported if
///         an orbital shell is above f or an aux shell above i (naming the
///         shell), if the scratch does not fit (naming the bytes), or if
///         nao_cart^2 naux_cart is past the kernels' 32-bit index;
///         InvalidConfig if the bases were not made by @c place over one
///         molecule, or if @p classes do not hold @p basis's shell pairs; or
///         the GpuRuntime error of the first copy, launch or BLAS call that
///         failed.
[[nodiscard]] Result<std::vector<double>>
three_center_gradient(DeviceResources &res, const BasisSet &basis,
                      std::span<const PrimitivePairClass> classes, const BasisSet &aux,
                      const double *gamma);

/// @brief The derivative of the Coulomb metric of the auxiliary basis @p aux
/// on the centre of its first function, spherical, on the device:
/// M1[k][P][Q] = d(P|Q)/dA_k, A the centre of P, at Q + naux (P + naux k),
/// naux = aux.nao_sphe(), into @p m1 (3 naux^2 doubles).
///
/// The metric reads A - B only, so the derivative on Q's centre is
/// -M1[k][P][Q], which is M1[k][Q][P]: each slab is antisymmetric, and
/// d(P|Q)/dR_{X,k} = [P on X] M1[k][P][Q] + [Q on X] M1[k][Q][P].
///
/// @p aux must be made by @c place. The metric's pairs are staged as
/// @c metric_gradient stages them, one ipmetric<lP, lQ> launch per class
/// writes the Cartesian matrices, and each is made spherical as
/// U_aux^T M1 U_aux. Plain stores: a run is bit-reproducible.
///
/// Queued on res.stream(), and synchronized. The scratch comes from
/// res.device_workspace(), so @p m1 must not point into it. The device
/// res.dev_idx() names is made current for the call and the previous one
/// restored.
///
/// @return Success (nothing written for an empty @p aux); InvalidConfig if
///         @p aux was not made by @c place, or an aux shell is above i;
///         Unsupported if 3 aux.nao_cart()^2 is past the kernels' 32-bit
///         index or the scratch does not fit (naming the bytes); or the
///         GpuRuntime error of the first copy, launch or BLAS call that
///         failed.
[[nodiscard]] Status metric_derivative(DeviceResources &res, const BasisSet &aux, double *m1);

/// @brief The derivative three-center integrals over the aux functions on one
/// atom, spherical, on the device: X[k][m][n][P'] = (nabla_k m n|P) =
/// -d(mn|P)/dA_k, A the centre of m (libcint's int3c2e_ip1), for P the
/// P'-th aux function on atom @p atom, at P' + nblk (n + nao (m + nao k)),
/// nao = basis.nao_sphe() and nblk that atom's aux.nao_sphe() share, into
/// @p x (3 nao^2 nblk doubles).
///
/// Every centre's derivative follows: d(mn|P)/dR_{X,k} = -[m on X]
/// X[k][m][n][P] - [n on X] X[k][n][m][P] + [P on X] (X[k][m][n][P] +
/// X[k][n][m][P]), the last by translational invariance. The whole
/// 3 nao^2 naux tensor is never needed at once: a caller walks the atoms.
///
/// @p classes must be @c primitive_pair_classes(res, basis), finished, and
/// both bases made by @c place over one molecule. The cells are those of
/// @c three_center_gradient whose aux shell sits on @p atom; one
/// xeri3<la, lb, lP> launch per class writes the Cartesian block, which is
/// made spherical on its three indices (a gemm with the atom's block of
/// U_aux, a strided-batched gemm and a gemm with U). Plain stores.
///
/// Queued on res.stream(), and synchronized (also once after reading mmd's
/// records back). @p x must not point into res.device_workspace(). The
/// device res.dev_idx() names is made current for the call and the previous
/// one restored.
///
/// @return Success (nothing written if the atom has no aux function or
///         @p basis is empty); InvalidConfig if the bases were not made by
///         @c place over one molecule, @p atom is not one of them, or
///         @p classes do not hold @p basis's shell pairs; Unsupported if an
///         orbital shell is above f or an aux shell above i, if
///         3 nao_cart^2 nblk_cart is past the kernels' 32-bit index, or if
///         the scratch does not fit (naming the bytes); or the GpuRuntime
///         error of the first copy, launch or BLAS call that failed.
[[nodiscard]] Status three_center_derivative(DeviceResources &res, const BasisSet &basis,
                                             std::span<const PrimitivePairClass> classes,
                                             const BasisSet &aux, uint32_t atom, double *x);

/// @brief The density-fitted J and K differentiated by every nuclear
/// coordinate at a fixed closed-shell density, spherical, on the device:
/// the two-electron half of the RHF skeleton Fock derivatives,
/// F^(x) = h^x + J^(x) - 1/2 K^(x).
///
/// With D = 2 C_occ C_occ^T, J_mn = SUM_ls (mn|ls) D_ls and K_mn = SUM_ls
/// (ml|ns) D_ls over the fit (mn|ls) = SUM_PQ (mn|P) M^-1_PQ (Q|ls), and x
/// = 3 A + k the coordinate R_{A,k}: J^(x) = dJ/dx and K^(x) = dK/dx at fixed
/// D, the integrals and the metric differentiated, the fit's inverse
/// included:
///
///     J^(x) = T^x d + T M^-1 (u^x - M^x d)
///     K^(x) = 2 SUM_i (T^x_i c_i^T + c_i T^x_i^T - c_i M^x c_i^T)
///
/// with T_mn,P = (mn|P), d = M^-1 u, u = T^T vec D, u^x = T^x^T vec D, T_i
/// the nao x naux matrix (T_i)_mP = SUM_l (ml|P) C_li and c_i = T_i M^-1.
/// Each slab is symmetric.
///
/// @p b and @p l are the fit as @c density_fit leaves it and as
/// @c gradient::df_rhf takes it (B = L^-1 (P|mn), naux x nao^2 at
/// Q + naux (n + nao m); L the naux x naux lower triangle). @p c is nao x nmo
/// column-major over @p basis, its first @p nocc columns occupied (only they
/// are read). @p jx and @p kx receive 3 n_atom slabs of nao x nao each,
/// column-major, slab x at x nao^2. @p g, when not null, receives u^x - M^x d
/// as the naux x 3 n_atom column-major matrix, column x: the vectors the DF
/// Hessian's Coulomb cross term is the Gram matrix of after L^-1.
///
/// Built one aux atom at a time (@c three_center_derivative's blocks, never
/// the whole derivative tensor), against @c metric_derivative's M1, each
/// piece a gemm, gemv or axpy; the README has the algebra. Contracted with D,
/// 1/2 tr(D J^(x)) is @c gradient::df_rhf's Coulomb term and -1/4
/// tr(D K^(x)) its exchange term.
///
/// **Refused up front, not split.** The scratch comes from
/// res.device_workspace(): the fitted pairs (naux nocc nao), the metric
/// derivative (3 naux^2 and its Cartesian copy), and one aux atom's blocks
/// (about 9 nao^2 nblk beside the Cartesian 3 nao_cart^2 nblk_cart); past
/// the device's free memory it is Unsupported, naming the bytes, before
/// anything is allocated. None of @p b, @p l, @p c, @p jx, @p kx and @p g
/// may point into it.
///
/// Queued on res.stream(), and synchronized. The device res.dev_idx() names
/// is made current for the call and the previous one restored.
///
/// @return Success (all zeros for nocc = 0 or an empty basis); InvalidConfig
///         if the bases were not made by @c place over one molecule, if
///         @p nocc is outside [0, nao], or if @p classes do not hold
///         @p basis's shell pairs; Unsupported for a shell past the ladders,
///         an index past the kernels' or the BLAS calls' 32 bits, or scratch
///         past the device's free bytes; or the GpuRuntime error of the first
///         copy, launch or BLAS call that failed.
[[nodiscard]] Status df_jk_derivative(DeviceResources &res, const BasisSet &basis,
                                      std::span<const PrimitivePairClass> classes,
                                      const BasisSet &aux, const double *b, const double *l,
                                      const double *c, int32_t nocc, double *jx, double *kx,
                                      double *g);

} // namespace free_breakfast
