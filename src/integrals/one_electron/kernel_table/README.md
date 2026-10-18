# Kernel table

The `[la][lb]` dispatch table every one-electron launcher reads, built at
compile time from the family's kernel template. Header-only, device-side:
`kernel_table.h`, target `free_breakfast::integrals::one_electron::kernel_table`.

## Design

FelizJueves emits each class of a family as one explicit specialization of a
kernel template, `overlap<0, 1>`, rather than one C name per class (`s_01`),
and writes the classes it emitted as `kernels/<family>_classes.h`
(`tools/emit_one_electron_kernels.py`). A launcher then:

- `static_assert`s `lists_every_class(<family>_classes)`, which holds only when
  the list is exactly every `la <= lb <= kMaxAngularMomentum`. Raising
  `kMaxAngularMomentum` without re-emitting is a compile error naming the fix.
- builds its table with `kernel_table<Kernel, Family>()`, where `Family` is a
  struct whose `static constexpr` member template `kernel<la, lb>()` returns
  `&family<la, lb>` (a struct rather than a template lambda, which nvcc warns
  on). The `la > lb` cells are null, as the host swaps every shell pair into
  `la <= lb`.

No kernel is named per class anywhere by hand. A class the list names but no
included unit defines fails to link: the template's primary
(`template <uint8_t La, uint8_t Lb>`) is declared once, in
`kernels/<family>.h`, and defined nowhere. A unit whose parameters drift from
that header fails to compile.
