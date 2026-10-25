# Localization

Orbital localization: post-SCF rotations of the occupied orbitals, within their
own span, into a chemically readable form. The density, the energy and every
observable are invariant under them; only the individual orbitals change. So a
localization runs after the SCF, not inside it.

## Sub-modules

| Directory | Module | Role |
|-----------|--------|------|
| `pipek_mezey/` | `free_breakfast.localization.pipek_mezey` | Pipek-Mezey on the GPU (`pipek_mezey_localize`), maximizing the sum of squared (or fourth powers of) Mulliken atomic populations |
| `ibo/` | `free_breakfast.localization.ibo` | Intrinsic atomic orbitals over a minimal basis (`intrinsic_atomic_orbitals`), and intrinsic bond orbitals (`ibo_localize`): Pipek-Mezey's sweep over the IAO populations |
| `livvo/` | `free_breakfast.localization.livvo` | Valence virtual orbitals (`valence_virtual_orbitals`): the part of the IAOs' span orthogonal to the occupied orbitals, m - n_occ of them; their localization over the IAO populations (`livvo_localize`, the LIVVOs); and each orbital's valence character (`valence_character`) |

Pipek-Mezey came first because it needs nothing the SCF does not already have:
the overlap and the occupied orbitals. IBO adds only the MINAO overlaps, and
reuses its sweep. Foster-Boys (dipole integrals) and Edmiston-Ruedenberg (the
two-electron integrals in the MO basis) are not here.

`livvo` is the one exception to "occupied only": the valence virtual orbitals
(Derricotte & Evangelista, 2017) are the virtual half of the IAOs' span, the
valence-sized slice of the virtual space. It takes the IAOs as an argument
rather than building them, so it does not depend on `ibo`; its LIVVOs reuse
`pipek_mezey_localize` over the IAOs exactly as `ibo_localize` does.

The pipeline step over them is `pipeline/localization`, `LocalizationMode::PipekMezey`
and `LocalizationMode::Ibo`.

## Files

| File | Role |
|------|------|
| `CMakeLists.txt` | Adds the sub-directories |
