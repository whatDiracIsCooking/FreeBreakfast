/// @file
/// @brief Convenience module that re-exports the integral layer.
///
/// `import free_breakfast.integrals;` gives the primitive-pair records and the
/// one-electron matrices. The sub-modules remain importable individually.

export module free_breakfast.integrals;

export import free_breakfast.integrals.mmd;
export import free_breakfast.integrals.one_electron;
