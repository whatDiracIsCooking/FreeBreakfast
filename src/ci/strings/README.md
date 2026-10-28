# Strings

The occupation strings of one spin for a determinant CI: every way of putting
N electrons in n orbitals, their addresses, and what a†_p a_q does to each.
`ci::StringSet` is host-side and over plain integers; `ci::DeviceStringSet`
is one with its strings and lists uploaded, the single device copy that the
GPU sigma, RDMs and S² all read.

## Module

`free_breakfast.ci.strings`, also re-exported by `free_breakfast.ci`; target
`free_breakfast::ci::strings`.

| Name | Role |
|---|---|
| `ci::StringSet::create(n_orbitals, n_electrons)` | Enumerate the strings and build their replacement lists; `Result<StringSet>` |
| `size()` | C(n, N); addresses are [0, size()) |
| `strings()` | Every string's bit pattern, by address |
| `address(string)` | A pattern's address, or -1 when it is not one of the set's |
| `replacements(address)`, `replacements_per_string()` | The N (n - N + 1) nonzero a†_p a_q \|I> for one string, as `ci::Replacement{target, sign, p, q}` |
| `ci::DeviceStringSet::create(res, n_orbitals, n_electrons)` | A `ci::StringSet` with `strings()` and every list uploaded to `res`'s device; `Result<DeviceStringSet>` |
| `device_strings()`, `device_replacements()`, `dev_idx()` | The device copies, and the device they are on |

## Design

**A string is a `uint64_t` bit pattern**, bit k set for orbital k occupied, so
n is at most 64. `create` also rejects a set whose strings, or whose
replacements in all, an `int32_t` cannot count: addresses and list offsets
are `int32_t`, which is what the GPU tables will index with, and no CI vector
over such a set would fit in memory anyway.

**Lexical order is ascending integer order.** The address of a string whose
occupied orbitals are o_0 < o_1 < ... < o_{N-1} is the sum of C(o_k, k + 1)
(the combinatorial number system), which ranks the patterns in ascending
numeric order. That is PySCF's `cistring` order, so CI vectors and the
`direct_spin1` golden data compare element for element with no permutation.
`ci::StringSet` keeps the N × n table of those binomials, so `address` is one
addition per electron.

**The sign convention** orders the creation operators by ascending orbital,
\|I> = a†_{o_0} a†_{o_1} ... \|vac>. Then a†_p a_q \|I> carries (-1) to the
number of orbitals occupied in I strictly between p and q.

**Each list has the same length**, N (n - N + 1): the N diagonal terms p = q
(target I, sign +1), then each occupied q times each empty p. The fixed stride
is what lets the lists be one flat array, on the host now and on the device
later. The order is PySCF's `gen_linkstr_index`: diagonals over the occupied q
ascending, then for each occupied q ascending, each empty p ascending.

The edge cases fall out without special-casing: N = 0 is the vacuum alone,
one string with an empty list; N = n is one string whose list is only its
diagonal.

**A link is eight bytes**, `ci::Replacement{int32_t target; int16_t sign;
int8_t p, q;}`, `alignas(8)`, so a gather reads one in a single load. An
orbital fits an `int8_t` because n ≤ 64, and the target needs the full 31
bits `create`'s count check allows. It carries (p, q) rather than a pair
index: σ wants the packed pair, the RDMs p · n + q and S²'s β partner
q · n + p, and each kernel forms its own from p and q, so one table serves
all three. The definition is `replacement.h`, `#include`d by the `.cu`
gathers and re-exported by the module (the `atom.h` arrangement), so host and
device read the same type.

**One device copy, borrowed.** `ci::DeviceStringSet` is a `ci::StringSet`,
so the host-side code that takes one (`ci::solve`'s P-space, the test oracles)
takes it too, and it adds the uploaded tables. `ci::Sigma`, `ci::Rdm` and
`ci::SpinSquare` hold pointers into them rather than uploading their own, so
a CASSCF's two sigmas, its RDMs and every macro-iteration's S² read one copy;
for N_α = N_β the caller passes the one set as both spins and there is one
copy for both. A set must outlive what is created from it.

## Tests

`test/gtest/ci_strings_host.cpp`. Addressing is checked as a bijection onto
[0, C(n, N)) for every n ≤ 10 and N ≤ n, by walking every n-bit pattern
(each string lands on the next address in turn, every other pattern on -1)
and round-tripping address → string → address. The replacement lists are
checked against a†_p a_q applied to the bit pattern operator by operator, a
code path that shares nothing with the module's sign rule, for every (p, q)
and every string with n ≤ 8: each surviving term appears exactly once with its
target and sign, and nothing else appears. Further cases cover N = 0, N = n,
n = 64 and the rejected shapes. The device copy is exercised by every GPU CI
test, which builds its sets as `ci::DeviceStringSet`s.

## Files

| File | Role |
|------|------|
| `replacement.h` | `ci::Replacement`, the one definition the module and the `.cu` gathers share |
| `strings.cppm` | Module interface: `ci::StringSet`, `ci::DeviceStringSet`, and `ci::Replacement` re-exported |
| `strings.cpp` | Enumeration, addressing and the replacement lists; `ci::DeviceStringSet`'s upload |
| `CMakeLists.txt` | The module `free_breakfast.ci.strings`, and `free_breakfast::ci::strings::headers` for `replacement.h` |
