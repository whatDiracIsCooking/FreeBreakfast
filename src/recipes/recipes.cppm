/// @file
/// @brief Convenience module that re-exports the dense linear-algebra layer.
///
/// `import free_breakfast.recipes;` gives every recipes routine. The sub-modules
/// remain importable individually.

export module free_breakfast.recipes;

export import free_breakfast.recipes.blas;
export import free_breakfast.recipes.cg_unitary;
export import free_breakfast.recipes.cholesky;
export import free_breakfast.recipes.complex_cast;
export import free_breakfast.recipes.davidson;
export import free_breakfast.recipes.diagonalize;
export import free_breakfast.recipes.diis;
export import free_breakfast.recipes.einsum;
export import free_breakfast.recipes.expm;
export import free_breakfast.recipes.linear_operator;
export import free_breakfast.recipes.matrix_ao;
export import free_breakfast.recipes.norm;
export import free_breakfast.recipes.orthogonalize;
export import free_breakfast.recipes.polynomial_roots;
export import free_breakfast.recipes.sort;
