/// @file
/// @brief Convenience module that re-exports the NEVPT2 layer.
///
/// `import free_breakfast.nevpt2;` gives every nevpt2 module. The sub-modules
/// remain importable individually.

export module free_breakfast.nevpt2;

export import free_breakfast.nevpt2.density_fitting;
export import free_breakfast.nevpt2.rdm;
