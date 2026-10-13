# Pure `import std` -- no wwr dependency. wwr.extension.common has its own
# NonCopyable, but a language-level mixin should not drag the GPU extension
# layer into every class that wants one, and wwr has no NonMovable or
# NonTransferable to pair with it.
free_breakfast_add_cxx_module_library(
  NAME
  free_breakfast.core.nontransferrable
  PRIMARY_INTERFACE
  nontransferrable.cppm
  IMPORT_STD
)
