# Bond-orbital analysis

Each orbital read the way Knizia reads the IBOs (J. Chem. Theory Comput. 9,
4834 (2013), section 3.2): the atoms it sits on, how much of it each holds,
and the angular character of each atom's share. One center is a core or a
lone pair, two a bond, three or more a multi-center bond like diborane's
B-H-B. "The carbon part of the CH bond orbitals has ~28% s-character" is a
center's `character`, and each orbital is read as sigma or pi from those
characters, the rule Derricotte and Evangelista give for LIVVOs (J. Chem.
Theory Comput. 13, 5984 (2017), section 2.2).

Host arithmetic only, no GPU and no kernels.

## Module

`free_breakfast.analysis.bond`, target `free_breakfast::analysis::bond`.

```cpp
struct BondCenter {
  uint32_t atom;
  double population;               // share of the orbital on the atom
  std::array<double, 4> character; // s, p, d, f shares of that, summing to one
};
enum class BondKind : uint8_t { Sigma, Pi };
struct BondOrbital {
  std::vector<BondCenter> centers; // by descending population
  BondKind kind;
};

Result<std::vector<BondOrbital>> bond_orbitals(const BasisSet &minimal, int32_t n_orb,
                                               std::span<const double> weights,
                                               double threshold = 0.1,
                                               double pi_threshold = 0.9);
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

Nothing in it is particular to the IBOs: any orbitals' weights over the IAOs
read the same way, occupied or virtual (the LIVVOs).

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
- An orbital is **pi** when every one of its centers is almost entirely p,
  its `character[1]` at least `pi_threshold`, and **sigma** otherwise. Only
  the centers count, so a tail under `threshold` cannot make a pi bond
  sigma. Over IBOs at the default exponent of four, which separates sigma
  from pi rather than giving banana bonds, this splits a double bond into
  its sigma and pi halves.
- The rule applies to **one-center** orbitals unchanged: a core or an s-rich
  lone pair is sigma, and a lone pair that is almost pure p, like water's
  out-of-plane one, is **pi**.
- The default **pi_threshold, 0.9**, is a reading choice like `threshold`:
  over MINAO a pi orbital's centers are all p, a carbon sp2 hybrid in a
  sigma bond 62-74% p, and a hydrogen 0%, so it sits well clear of both.

Over the canonical orbitals this reads delocalized orbitals as many-center;
it is meant for localized ones.

## Tests

`test/gtest/pipeline_localization_gpu.cpp`:

- on the host, over hand-written weights on a two-atom s/p minimal basis:
  the centers, their order and character, both ends of the threshold, and
  the refusals; then the sigma / pi kind, either side of `pi_threshold`, of
  a pi bond, a bond with one s-bearing end, and a pure-p and a half-s lone
  pair;
- through the Localization step's IBOs of water (STO-3G, cc-pVDZ) and NH4+
  (cc-pVDZ), both exponents: every IBO is on the heavy atom alone or a bond
  to one hydrogen, one bond per hydrogen; each center's population is the
  step's own, each character sums to one, a hydrogen's is all s, and the
  symmetry-equivalent bonds have the same heavy-atom hybrid;
- through the IBOs of ethylene and benzene (cc-pVDZ, exponent four,
  geometries inline): one pi bond and five sigma bonds beside two sigma
  cores, and three pi orbitals beside twelve sigma bonds and six cores.
  Benzene's pi IBOs are three-center (half on one carbon, a fifth on each
  neighbour), and pi all the same.

## Files

| File | Role |
|---|---|
| `bond.cppm` | Module interface: `BondCenter`, `BondKind`, `BondOrbital`, `bond_orbitals` |
| `bond.cpp` | The per-atom, per-l sums, the threshold and the kind |
| `CMakeLists.txt` | The module |
