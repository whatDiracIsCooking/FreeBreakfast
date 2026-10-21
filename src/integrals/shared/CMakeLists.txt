# Device headers more than one family of generated kernels includes.
# eval_boys.h is FelizJueves's device Boys routine, eval_boys<M>, which every
# unit with a fundamental includes (potential_en's and e_hermite's), written
# here by both tools/emit_*_kernels.py with the eval_boys_{M}.h it includes,
# one fixed fma chain per order M <= 20. It includes felizjueves/runtime.h,
# hence felizjueves::headers. A device library whose kernels include it links
# this PRIVATE.
free_breakfast_add_interface_library(
  NAME
  free_breakfast.integrals.shared.headers
  LINK_PUBLIC
  felizjueves::headers
)
