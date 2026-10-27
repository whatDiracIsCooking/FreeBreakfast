# Localization

Orbital localization: post-SCF rotations of the occupied orbitals, within their
own span, into a chemically readable form. The density, the energy and every
observable are invariant under them; only the individual orbitals change. So a
localization runs after the SCF, not inside it.

Everything it exports is in **`namespace free_breakfast::localization`**, and
its launchers in **`free_breakfast::localization::device`** (`device::populations`,
`device::rotate_columns`). The namespace is the *layer*, and this layer holds
three methods, so a name the whole layer shares loses its prefix while a name
belonging to one method keeps one: `PipekMezeyInfo` is now
`localization::Info`, since `ibo_localize` and `livvo_localize` return it too,
but PM's own options stay qualified as `localization::PmOptions` and
`localization::PmExponent`, and the three entry points stay
`localization::pm_localize`, `localization::ibo_localize` and
`localization::livvo_localize`. `docs/style.md` "Namespaces, and when a name
needs a prefix" has the rule.

## Sub-modules

| Directory | Module | Role |
|-----------|--------|------|
| `pipek_mezey/` | `free_breakfast.localization.pipek_mezey` | Pipek-Mezey on the GPU (`localization::pm_localize`), maximizing the sum of squared (or fourth powers of) Mulliken atomic populations |
| `ibo/` | `free_breakfast.localization.ibo` | Intrinsic atomic orbitals over a minimal basis (`intrinsic_atomic_orbitals`), and intrinsic bond orbitals (`localization::ibo_localize`): Pipek-Mezey's sweep over the IAO populations |
| `livvo/` | `free_breakfast.localization.livvo` | Valence virtual orbitals (`valence_virtual_orbitals`): the part of the IAOs' span orthogonal to the occupied orbitals, m - n_occ of them; their localization over the IAO populations (`localization::livvo_localize`, the LIVVOs); and each orbital's valence character (`valence_character`) |

Pipek-Mezey came first because it needs nothing the SCF does not already have:
the overlap and the occupied orbitals. IBO adds only the MINAO overlaps, and
reuses its sweep. Foster-Boys (dipole integrals) and Edmiston-Ruedenberg (the
two-electron integrals in the MO basis) are not here.

`livvo` is the one exception to "occupied only": the valence virtual orbitals
(Derricotte & Evangelista, 2017) are the virtual half of the IAOs' span, the
valence-sized slice of the virtual space. It takes the IAOs as an argument
rather than building them, so it does not depend on `ibo`; its LIVVOs reuse
`localization::pm_localize` over the IAOs exactly as `localization::ibo_localize` does.

The pipeline step over them is `pipeline/localization`, `LocalizationMode::PipekMezey`
and `LocalizationMode::Ibo`.

## Files

| File | Role |
|------|------|
| `CMakeLists.txt` | Adds the sub-directories |
