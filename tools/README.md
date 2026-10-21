# `tools/`

Standalone Python utilities around the C++ tree. Not tests, not a package, not
on the push gate.

```
bse_download.py   Fetch basis sets from the Basis Set Exchange as the JSON
                  src/io/basis reads. Standard library only.
atomic_cd/        aCD / acCD atomic Cholesky decomposition over PySCF
                  integrals; writes acCD auxiliary shells as BSE JSON.
emit_one_electron_kernels.py
                  Regenerate src/integrals/one_electron/{overlap,kinetic,
                  potential_en}/kernels/, and the mmd record bound
                  src/integrals/mmd/kernels/pair_bound/, from the
                  FelizJueves submodule. Standard library only.
emit_coulomb_kernels.py
                  Regenerate src/integrals/coulomb/kernels/{dq,e_hermite,
                  jab}/ from the FelizJueves submodule. Standard library
                  only.
emit_exchange_kernels.py
                  Regenerate src/integrals/exchange/kernels/kb/ from the
                  FelizJueves submodule. Standard library only.
extract_integral_golden.py
                  Copy the libcint golden data test/gtest/data/{one_electron,
                  coulomb,exchange}/ hold out of FelizJueves's fixtures.
                  Standard library only.
```

Re-run the last four after bumping `deps/FelizJueves`; all write tracked files,
so the diff is the review.

## Running

`bse_download.py` needs nothing beyond Python:

```sh
tools/bse_download.py cc-pVDZ def2-SVP -o data/basis
tools/bse_download.py --list
```

`atomic_cd` needs PySCF and numpy, which sit in the opt-in `tools` dependency
group in `pyproject.toml` rather than in `dev`, so the image and the push gate
do not carry them:

```sh
uv run --group tools python -m tools.atomic_cd.cli C sto-3g
uv run --group tools python -m tools.atomic_cd.cli C cc-pVDZ --threshold 1e-8 \
    --bse-out c_ccpvdz_accd.bse.json
```

The two compose: an orbital basis fetched by `bse_download.py` can be fed back
in with `--bse-in`, which makes the positional basis name a display label only.

```sh
tools/bse_download.py cc-pVDZ --elements C -o /tmp/basis
uv run --group tools python -m tools.atomic_cd.cli C cc-pVDZ \
    --bse-in /tmp/basis/cc-pVDZ.json
```

Each module's docstring carries the derivation and conventions;
`atomic_cd/__init__.py` lists them in dependency order.
