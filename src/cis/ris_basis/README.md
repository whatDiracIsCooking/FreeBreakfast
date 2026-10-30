# RIS basis

The RIS minimal auxiliary basis (Zhou & Parker 2024, gpu4pyscf's
`tdscf/ris.py`): on every atom one uncontracted shell per emitted l, all of
exponent α = θ / R_A², R_A the atom's Ghosh (GB) radius in bohr. Fitted over,
it gives the cheap two-electron part of the RIS A matrix.

## Module

`free_breakfast.cis.ris_basis`; target `free_breakfast::cis::ris_basis`.
Host-only: no device library, no `DeviceResources`.

| Name | Role |
|---|---|
| `cis::RisShells` | What a heavy atom (Z > 1) gets: `S`, `Sp` or `Spd`. Hydrogen gets `s` alone under all three. gpu4pyscf's `fitting_basis` `'s'`, `'sp'`, `'spd'` |
| `cis::ris_basis(atoms, theta = 0.2, shells = RisShells::Sp)` | The aux `BasisSet` over `atoms` (pass `molecule.atoms()`), in atom order; `Result<BasisSet>`. `InvalidConfig` for a θ that is not finite and positive, or a Z outside 1..36 |
| `cis::RisOptions` | `theta` (0.2) and `shells` (`Sp`) together, with the preconditioner's exchange switch `exchange` (off) and its inner GMRES's `k_max_inner_iters` (10) and `k_inner_tol` (1e-2): what `cis::Options::ris` and `cis::RisPreconditioner::create` carry |

```cpp
import free_breakfast.cis.ris_basis;
namespace fb = free_breakfast;

const fb::BasisSet aux = FBF_TRY(fb::cis::ris_basis(molecule.atoms(), 0.2, fb::cis::RisShells::Sp));
// aux.nao_sphe() rows of B, as pipeline.eri's density fitting places them.
```

## Design

**The same kind of basis `pipeline.eri` fits over.** That step places an aux
stamp with `place(stamp.primitive_normalized(), molecule)`. This one builds
each shell as a one-primitive `ShellStamp` of coefficient 1, normalizes it
with `ShellStamp::primitive_normalized` and places it at the atom, so the
result is spherical, carries the per-atom shell map (`natoms`,
`atom_shell_offset`), and is fed to `metric` / `three_center` unchanged.
It has no `name()`: no SAD stamp or carried set stands behind it. Spherical
means H2O at `Spd` is 11 functions, where a Cartesian count would be 12.

**It takes the atoms, not the `Molecule`.** A `Molecule` uploads its nuclei at
construction, so it cannot be built without a device, and nothing here reads
the device side. Taking `std::span<const Atom>` keeps the function and its
test (`test/gtest/cis_ris_basis_host.cpp`) host-only; the caller passes
`molecule.atoms()`.

**Defaults are gpu4pyscf's.** θ = 0.2 and `Sp` are `TDDFT_ris`'s `theta` and
`J_fit`. (Its `K_fit` default is `S`, the caller's choice here.) gpu4pyscf
also gives a heavy atom its p and d shells only when the string asks, and
hydrogen never, which `RisShells` mirrors.

**Where it came from.** The issue (#196) asked for a port of the private
`cis/ris_precondition`'s `build_ris_aux_basis`. That source is not reachable
from this tree, so this was written from the issue's spec and gpu4pyscf's
public `gpu4pyscf/tdscf/ris.py` (`get_minimal_auxbasis`) and
`gpu4pyscf/tdscf/parameter.py`, as of commit `8d132c7` (2025-04-28).

## The radii

`kGhoshRadii` in `ris_basis.cpp` is gpu4pyscf's `parameter.py` `radii`,
verbatim, for H..Kr, in Angstrom. That file cites them as the GB radii of

> D. C. Ghosh, R. Biswas, T. Chakraborty, N. Islam, S. K. Rajak,
> *The wave mechanical evaluation of the absolute radii of atoms*,
> J. Mol. Struct. THEOCHEM **865**, 60–67 (2008).

The paper itself was not consulted. The values were cross-checked
independently instead: Ghosh's radius is the most probable radius of the
outermost orbital, r = n*² a₀ / Z_eff with Slater's Z_eff and n*, and
a₀ = 0.5292 Å reproduces the table to its last digit for H (0.5292), C
(0.6513), N (0.5428) and O (0.4652), and within 1.3e-4 Å, rounding,
for 35 of the 36 entries (n* = 3.7 for K..Kr, and aufbau 3d⁴4s² / 3d⁹4s²
for Cr and Cu). The exception is **As**: Slater's rules give 1.1500 Å, the
table 1.145. It is kept as gpu4pyscf has it, since matching gpu4pyscf is the
point of the port, and the difference is 0.4% in R, 0.9% in α.

The radius converts to bohr with `kBohrPerAngstrom` (CODATA 2018).
gpu4pyscf's own factor is 1.8897259885789 (CODATA 2014), so its exponents
differ from these by a relative 1.4e-7; the test checks gpu4pyscf's
documented exponents for C, H and O at that tolerance.
