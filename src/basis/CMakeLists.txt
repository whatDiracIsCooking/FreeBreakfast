# angular_momentum.h and basis.h are the center-independent pieces, shared with
# CUDA / HIP #include-rs through this include path; the module partitions
# include them in their global module fragments and re-export the names.
# Both name the fixed-width integer types through integral_types.h.
free_breakfast_add_interface_library(
  NAME
  free_breakfast.basis.headers
  LINK_PUBLIC
  free_breakfast::core::types_headers
)

free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.basis
  PRIMARY_INTERFACE
  basis.cppm
  PARTITIONS
  shell.cppm
  ao.cppm
  harmonics.cppm
  basis_set.cppm
  shell_stamp.cppm
  basis_stamp.cppm
  IMPORT_STD
  LINK_PUBLIC
  free_breakfast::core::types
  # math::double_factorial_odd / math::factorial for the harmonic table, the
  # per-shell spherical scale and the primitive normalization.
  free_breakfast::core::primitives
  free_breakfast::basis::headers
  free_breakfast::molecule
  # :basis_stamp's place() returns a Result over a classified Error.
  free_breakfast::core::error_handling
  # Shell, ShellStamp and BasisSet's index accessors abort on an invariant
  # violation through check_or_abort.
  free_breakfast::core::diagnostics_macros
)
