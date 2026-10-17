/// @file
/// @brief Convenience module that re-exports the one-electron integral
/// matrices on the GPU: the overlap S, the kinetic energy T, the nuclear
/// attraction V, and the core Hamiltonian H = T + V.
///
/// `import free_breakfast.integrals.one_electron;` gives every build and the
/// device inputs they read. The sub-modules remain importable individually.

export module free_breakfast.integrals.one_electron;

export import free_breakfast.integrals.one_electron.upload;
export import free_breakfast.integrals.one_electron.overlap;
export import free_breakfast.integrals.one_electron.kinetic;
export import free_breakfast.integrals.one_electron.potential_en;
export import free_breakfast.integrals.one_electron.core_hamiltonian;
