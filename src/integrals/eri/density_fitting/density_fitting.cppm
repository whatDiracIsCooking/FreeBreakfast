/// @file
/// @brief Convenience module that re-exports density fitting.
///
/// `import free_breakfast.integrals.eri.density_fitting;` gives the Coulomb
/// metric (P|Q) of an auxiliary basis, the three-center (mn|P), the fit B = L^-1 (mn|P) over it,
/// density-fitted J and K from B, and B in the MO basis. The sub-modules remain
/// importable individually.

export module free_breakfast.integrals.eri.density_fitting;

export import free_breakfast.integrals.eri.density_fitting.metric;
export import free_breakfast.integrals.eri.density_fitting.three_center;
export import free_breakfast.integrals.eri.density_fitting.fit;
export import free_breakfast.integrals.eri.density_fitting.jk;
export import free_breakfast.integrals.eri.density_fitting.mo;
