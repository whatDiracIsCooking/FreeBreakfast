/* fb_nuclei_view: a molecule's nuclei on the device, by pointer, for the
 * launch wrappers. Forked from src/molecule/nuclei_view.h, the same fields in
 * the same order, as plain C (rust/docs/gpu_layer.md, "Shared types"):
 * one_electron/potential_en/potential_en.h includes it, and
 * devtools/rust-bindings.py reads it into fb-integrals-sys's #[repr(C)]
 * mirror. */
#pragma once

#include <stdint.h>

/// A molecule's nuclei on the device: the arrays of a molecule, by pointer.
/// The struct itself is host memory, handed to a wrapper by address; the
/// arrays it points at are device memory.
typedef struct fb_nuclei_view {
  /// The number of nuclei, positive.
  int32_t n_nuc;
  /// Z_C, each nucleus's charge (n_nuc).
  const double *z;
  /// C, each nucleus's position in bohr (3 per nucleus, x y z).
  const double *c;
} fb_nuclei_view;
