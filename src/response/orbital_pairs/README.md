# Orbital pairs

The packed order of a CASSCF's non-redundant orbital pairs, on the GPU: a full
nmo × nmo matrix's pairs into the packed vector, and a packed vector back out
to the antisymmetric κ that C exp(κ) rotates by. The order is
[`../README.md`](../README.md)'s "Orbitals and the packed order", and
everything over the pairs shares it: `response::Fock::gradient`,
`response::Hessian`'s products and diagonal, and `mcscf::solve`'s step.

## Module

`free_breakfast.response.orbital_pairs`, target
`free_breakfast::response::orbital_pairs`, re-exported by
`free_breakfast.response`.

```cpp
Status response::pack_orbital_pairs(DeviceResources &res, int32_t nmo, int32_t ncore,
                                    int32_t ncas, const double *full, double *packed);
Status response::unpack_orbital_pairs(DeviceResources &res, int32_t nmo, int32_t ncore,
                                      int32_t ncas, const double *packed, double *kappa);
```

`full` and `kappa` are nmo × nmo column-major device matrices; `packed` is
ncore (nmo − ncore) + ncas (nmo − nocc) doubles on the device. Both queue on
`res.stream()` and do not synchronize. Both refuse a negative ncore or ncas,
and ncore + ncas past nmo, as InvalidConfig.

## What it does

**Pack is two 2D copies.** The packed vector is two column-major blocks of the
nmo × nmo matrix back to back: rows [ncore, nmo) of columns [0, ncore), then
rows [nocc, nmo) of columns [ncore, nocc). Each block is one
`wwrMemcpy2DAsync`, so there is no gather kernel. A block with no rows or no
columns is skipped, because a zero-extent copy does not behave the same on
both runtimes.

**Unpack is a memset, the same two copies the other way, and two geams.** The
copies write the lower blocks, κ_pq = x_pq. Each upper block, κ_qp = −x_pq, is
the negated transpose of a lower block it shares no element with, written by
one `geam` in its in-place mode (B is C at β = 0), so nothing past κ is read.
`res.blas()` must be in host pointer mode (its default). `packed` and `kappa`
must not overlap.

It is its own module, not part of `fock/`, because it has nothing to do with
the Fock matrices: the Hessian packs and unpacks every product through it, and
`mcscf::solve` unpacks its step and packs the CI's response through it.

## Not done

- **Active–active pairs.** The two blocks are CAS's; RAS and GAS rotate
  between active subspaces too, and need a third.

## Tests

`test/gtest/response_orbital_pairs_gpu.cpp` (suite `ResponseOrbitalPairs`),
linked into `free_breakfast_tests_gpu`: the packed order element by element
against a host enumeration of the pairs, and unpack as its antisymmetric
inverse, over empty and non-empty core and active spaces; and the refusals.
