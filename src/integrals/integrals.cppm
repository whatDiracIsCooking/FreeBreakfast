/// @file
/// @brief Convenience module that re-exports the integral layer.
///
/// `import free_breakfast.integrals;` gives the primitive-pair records, the
/// one-electron matrices, the Coulomb matrix J, and the ordered pair classes
/// and the exchange matrix K. The sub-modules remain
/// importable individually.

export module free_breakfast.integrals;

export import free_breakfast.integrals.mmd;
export import free_breakfast.integrals.one_electron;
export import free_breakfast.integrals.eri.integral_direct.coulomb;
export import free_breakfast.integrals.eri.integral_direct.exchange;
