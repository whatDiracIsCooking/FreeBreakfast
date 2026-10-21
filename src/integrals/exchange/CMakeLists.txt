# The exchange build's hand-written device steps (helpers/).
add_subdirectory(helpers)

# The exchange kernels, k<la, lc, lb, ld>: one per K class through d, 81 in
# all (kernels/k/, written by tools/emit_exchange_kernels.py), each compiled on
# its own, so their compile time and memory are seen before any launcher
# exists. Device code in its own library, so its language and -x hip never
# reach a module compile. No launcher yet: the one that will index
# kernels/k/k.h's k_kernels joins these sources, as coulomb_launch.cu sits
# beside the J kernels. Relocatable device code, this library's default, so it
# must never pass NO_DEVICE_LINKING. The kernels include FelizJueves's
# felizjueves/runtime.h (felizjueves::headers, which also carries FJ_WAVEFRONT
# = WWR_WARP_SIZE) and the shared eval_boys.h (shared::headers); both PRIVATE.
free_breakfast_add_gpu_device_library(
  NAME
  free_breakfast.integrals.exchange.device
  SOURCES
  kernels/k/k_00_00.cu
  kernels/k/k_00_01.cu
  kernels/k/k_00_02.cu
  kernels/k/k_00_10.cu
  kernels/k/k_00_11.cu
  kernels/k/k_00_12.cu
  kernels/k/k_00_20.cu
  kernels/k/k_00_21.cu
  kernels/k/k_00_22.cu
  kernels/k/k_01_00.cu
  kernels/k/k_01_01.cu
  kernels/k/k_01_02.cu
  kernels/k/k_01_10.cu
  kernels/k/k_01_11.cu
  kernels/k/k_01_12.cu
  kernels/k/k_01_20.cu
  kernels/k/k_01_21.cu
  kernels/k/k_01_22.cu
  kernels/k/k_02_00.cu
  kernels/k/k_02_01.cu
  kernels/k/k_02_02.cu
  kernels/k/k_02_10.cu
  kernels/k/k_02_11.cu
  kernels/k/k_02_12.cu
  kernels/k/k_02_20.cu
  kernels/k/k_02_21.cu
  kernels/k/k_02_22.cu
  kernels/k/k_10_00.cu
  kernels/k/k_10_01.cu
  kernels/k/k_10_02.cu
  kernels/k/k_10_10.cu
  kernels/k/k_10_11.cu
  kernels/k/k_10_12.cu
  kernels/k/k_10_20.cu
  kernels/k/k_10_21.cu
  kernels/k/k_10_22.cu
  kernels/k/k_11_00.cu
  kernels/k/k_11_01.cu
  kernels/k/k_11_02.cu
  kernels/k/k_11_10.cu
  kernels/k/k_11_11.cu
  kernels/k/k_11_12.cu
  kernels/k/k_11_20.cu
  kernels/k/k_11_21.cu
  kernels/k/k_11_22.cu
  kernels/k/k_12_00.cu
  kernels/k/k_12_01.cu
  kernels/k/k_12_02.cu
  kernels/k/k_12_10.cu
  kernels/k/k_12_11.cu
  kernels/k/k_12_12.cu
  kernels/k/k_12_20.cu
  kernels/k/k_12_21.cu
  kernels/k/k_12_22.cu
  kernels/k/k_20_00.cu
  kernels/k/k_20_01.cu
  kernels/k/k_20_02.cu
  kernels/k/k_20_10.cu
  kernels/k/k_20_11.cu
  kernels/k/k_20_12.cu
  kernels/k/k_20_20.cu
  kernels/k/k_20_21.cu
  kernels/k/k_20_22.cu
  kernels/k/k_21_00.cu
  kernels/k/k_21_01.cu
  kernels/k/k_21_02.cu
  kernels/k/k_21_10.cu
  kernels/k/k_21_11.cu
  kernels/k/k_21_12.cu
  kernels/k/k_21_20.cu
  kernels/k/k_21_21.cu
  kernels/k/k_21_22.cu
  kernels/k/k_22_00.cu
  kernels/k/k_22_01.cu
  kernels/k/k_22_02.cu
  kernels/k/k_22_10.cu
  kernels/k/k_22_11.cu
  kernels/k/k_22_12.cu
  kernels/k/k_22_20.cu
  kernels/k/k_22_21.cu
  kernels/k/k_22_22.cu
  LINK_PRIVATE
  felizjueves::headers
  free_breakfast::integrals::shared::headers
)
