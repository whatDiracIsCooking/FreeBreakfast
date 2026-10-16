/// @file
/// @brief Convenience module that re-exports the dense linear-algebra layer.
///
/// `import free_breakfast.linalg;` gives every linalg routine. The sub-modules
/// remain importable individually.

export module free_breakfast.linalg;

export import free_breakfast.linalg.diis;
export import free_breakfast.linalg.matrix_ao;
export import free_breakfast.linalg.orthogonalize;
