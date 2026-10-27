/// @file
/// @brief Convenience module that re-exports density fitting.
///
/// `import free_breakfast.integrals.eri.density_fitting;` gives the Coulomb
/// metric (P|Q) of an auxiliary basis, the fit B = L^-1 (mn|P) over it, and
/// density-fitted J and K from B. The sub-modules remain importable
/// individually.

export module free_breakfast.integrals.eri.density_fitting;

export import free_breakfast.integrals.eri.density_fitting.metric;
export import free_breakfast.integrals.eri.density_fitting.fit;
export import free_breakfast.integrals.eri.density_fitting.jk;
