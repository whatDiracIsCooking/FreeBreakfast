# Strings

The occupation strings of one spin for a determinant CI: every way of putting
N electrons in n orbitals, their addresses, and what a†_p a_q does to each.
Host-side and over plain integers; the GPU copies of the tables come with the
GPU sigma.

## Module

`free_breakfast.ci.strings`, also re-exported by `free_breakfast.ci`; target
`free_breakfast::ci::strings`.

| Name | Role |
|---|---|
| `ci::StringSet::create(n_orbitals, n_electrons)` | Enumerate the strings and build their replacement lists; `Result<StringSet>` |
| `size()` | C(n, N); addresses are [0, size()) |
| `strings()` | Every string's bit pattern, by address |
| `address(string)` | A pattern's address, or -1 when it is not one of the set's |
| `replacements(address)`, `replacements_per_string()` | The N (n - N + 1) nonzero a†_p a_q \|I> for one string, as `ci::Replacement{target, p, q, sign}` |

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

## Tests

`test/gtest/ci_strings_host.cpp`. Addressing is checked as a bijection onto
[0, C(n, N)) for every n ≤ 10 and N ≤ n, by walking every n-bit pattern
(each string lands on the next address in turn, every other pattern on -1)
and round-tripping address → string → address. The replacement lists are
checked against a†_p a_q applied to the bit pattern operator by operator, a
code path that shares nothing with the module's sign rule, for every (p, q)
and every string with n ≤ 8: each surviving term appears exactly once with its
target and sign, and nothing else appears. Further cases cover N = 0, N = n,
n = 64 and the rejected shapes.

## Files

| File | Role |
|------|------|
| `strings.cppm` | Module interface: `ci::StringSet`, `ci::Replacement` |
| `strings.cpp` | Enumeration, addressing and the replacement lists |
| `CMakeLists.txt` | The module `free_breakfast.ci.strings` |
