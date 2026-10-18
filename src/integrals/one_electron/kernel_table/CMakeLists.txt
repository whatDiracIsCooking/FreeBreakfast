# kernel_table.h, which every family's launcher #includes to build its
# [la][lb] table from the family's kernel template. Header-only and device-side:
# no module re-exports it. basis::headers for kMaxAngularMomentum, the table's
# extent.
free_breakfast_add_interface_library(
  NAME
  free_breakfast.integrals.one_electron.kernel_table
  LINK_PUBLIC
  free_breakfast::basis::headers
)
