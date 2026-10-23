# constants.h includes wwr's complex.h for the numeric literals' complex
# specializations, which needs what vec.h's vector_types.h does in core/types:
# the WWR_GPU_BACKEND_* define, wwr's src/ include root and the vendor header's
# include path, all carried by wwr::vector_types::host. PUBLIC, since an
# importer naming kOne<wwrDoubleComplex> names the vendor type.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.core.constants
  PRIMARY_INTERFACE
  constants.cppm
  LINK_PUBLIC
  wwr::vector_types::host
)

# constants.h is plain C++ over that one wwr header, so a device translation
# unit (which cannot import a named module) #includes it directly through this
# target.
free_breakfast_add_interface_library(
  NAME
  free_breakfast.core.constants_header
  LINK_PUBLIC
  wwr::vector_types::host
)
