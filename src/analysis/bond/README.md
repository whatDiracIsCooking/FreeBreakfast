# Bond-orbital analysis

Each orbital read the way Knizia reads the IBOs (J. Chem. Theory Comput. 9,
4834 (2013), section 3.2): the atoms it sits on, how much of it each holds,
and the angular character of each atom's share. One center is a core or a
lone pair, two a bond, three or more a multi-center bond like diborane's
B-H-B. "The carbon part of the CH bond orbitals has ~28% s-character" is a
center's `character`.

Host arithmetic only, no GPU and no kernels.

## Module

`free_breakfast.analysis.bond`, target `free_breakfast::analysis::bond`.

```cpp
struct BondCenter {
  uint32_t atom;
  double population;               // share of the orbital on the atom
  std::array<double, 4> character; // s, p, d, f shares of that, summing to one
};
struct BondOrbital {
  std::vector<BondCenter> centers; // by descending population
};

Result<std::vector<BondOrbital>> bond_orbitals(const BasisSet &minimal, int32_t n_orb,
                                               std::span<const double> weights,
                                               double threshold = 0.1);
```

`weights` is m x n_orb, column-major, m = `minimal.nao_sphe()`: each orbital's
weight on each function of an orthonormal, atom-owned minimal basis, which is
what `analysis/population`'s `orbital_weights` returns over the IAOs. The
module takes the span, not the IAOs, so it depends on neither
`analysis/population` nor `localization/ibo`:

```cpp
const std::vector<double> weights =
    FBF_TRY(orbital_weights(res, basis, minao, n_occ, s, c_ibo, iao.data()));
const std::vector<BondOrbital> bonds = FBF_TRY(bond_orbitals(minao, n_occ, weights));
```

## The rules

- An atom's **population** in an orbital is the orbital's weights summed over
  the atom's functions; its **character** is the same sum split by each
  function's angular momentum, over the population.
- The **centers** are the atoms holding at least `threshold` of the orbital,
  largest first. The largest is a center whatever the threshold, so no
  orbital has none.
- The default **threshold, 0.1**, is a reading choice, not the paper's: it
  keeps SO3's 15% sulfur share of a polar pi bond as a second center, and
  drops the few-percent tails a localized orbital always has. A caller
  wanting the tails passes a lower one.

Over the canonical orbitals this reads delocalized orbitals as many-center;
it is meant for localized ones.

## Tests

`test/gtest/pipeline_localization_gpu.cpp`:

- on the host, over hand-written weights on a two-atom s/p minimal basis:
  the centers, their order and character, both ends of the threshold, and
  the refusals;
- through the Localization step's IBOs of water (STO-3G, cc-pVDZ) and NH4+
  (cc-pVDZ), both exponents: every IBO is on the heavy atom alone or a bond
  to one hydrogen, one bond per hydrogen; each center's population is the
  step's own, each character sums to one, a hydrogen's is all s, and the
  symmetry-equivalent bonds have the same heavy-atom hybrid.

## Files

| File | Role |
|---|---|
| `bond.cppm` | Module interface: `BondCenter`, `BondOrbital`, `bond_orbitals` |
| `bond.cpp` | The per-atom, per-l sums and the threshold |
| `CMakeLists.txt` | The module |
