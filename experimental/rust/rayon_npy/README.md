# rust/rayon_npy — one `.npy`, many writing threads

A column-major (`fortran_order: True`) float64 matrix written to a single `.npy`
file by rayon threads at once, each writing its own columns. Pure Rust, no
free_breakfast code and no GPU: no C ABI, no CMake target, no link record.

## How

The header of a 2-D `<f8` array depends only on its shape, so
`NpyWriter::create(path, rows, cols)` writes it (version 1.0, padded so the data
starts on a 64-byte boundary, as numpy pads) and `set_len`s the file to its final
size before any value exists. Column `j` then starts at a fixed offset,
`header + j * rows * 8`, and in column-major order a block of columns is one
contiguous range. Each thread writes its block there with `pwrite`
(`FileExt::write_all_at`). That call takes an explicit offset and never touches
the file's shared cursor, so writes to disjoint ranges need no lock, and
`NpyWriter` is `Sync` with `write_columns(&self, first_col, values)`.

| Item | What |
|---|---|
| `NpyWriter` | `create`, `rows`, `cols`, `write_columns` (whole columns, range-checked), `finish` (`fdatasync`; dropping closes without it) |
| `write_par(path, rows, cols, cols_per_chunk, values)` | A matrix already in memory: one rayon task per block of columns, written straight from `values`, no copy |
| `fill_par(path, rows, cols, cols_per_chunk, fill)` | A matrix never held whole: `fill(first_col, block)` computes each block on whichever thread writes it, into one reused buffer per thread |
| `read(path)` | Reads a 2-D Fortran-order `<f8` `.npy` back, for the tests and the demo |
| `header(rows, cols)` | The header bytes, public so the siblings write the identical file |

Neither helper syncs, just as `np.save` doesn't. A column nobody writes reads
back as zeros, because the file is sparse until it is written. Unix only
(`pwrite`), and little-endian only (values go out as their native bytes; a
big-endian build is a `compile_error!`).

## Build and run

The toolchain is the image's `/opt/rust` (`docker/Dockerfile.base`), or
`~/.cargo` from rustup in a container built before that. `rayon` comes from
crates.io. `Cargo.lock` is not tracked, as in `../gemm`.

```bash
cd experimental/rust/rayon_npy
cargo test --release          # round trips at every chunking, a 16-thread
                              # one-column-per-task run, empty shapes, refusals
cargo run --release -- 4096 8192 16 target/demo.npy
                              # npy-demo [ROWS [COLS [COLS_PER_CHUNK [PATH]]]]
```

Nothing in ctest or either gate runs cargo. This directory is checked by hand only.

## Checked (2026-11-11)

- `cargo test --release`: 7 tests pass. `cargo clippy --all-targets` and
  `cargo fmt --check` are clean.
- `np.load` of the demo's 4096 x 8192 file gives a Fortran-contiguous float64
  array equal to the expected values, and the file is **byte-identical** to
  `np.save(np.asfortranarray(expected))`, header included.

## Measured: threads do not speed up the write itself

`npy-demo 4096 8192 C` (0.27 GB). The fill is trivial (`i + j * rows`), so this
times the writes, from `create` to the last `pwrite`, with no sync, on the
container's ext4 workspace mount (NVMe), on a machine other worktrees share:

| threads | 16 columns per task | 256 columns per task |
|---|---|---|
| 1 | 1.78 GB/s | 1.32 GB/s |
| 2 | 1.62 | 1.60 |
| 4 | 1.81 | 0.71 |
| 8 | 1.74 | 0.74 |
| 16 | 1.57 | 1.54 |
| 24 | 1.00 | 1.42 |

The rate stays flat at roughly 1.5–1.8 GB/s. The slower lines are noise from the
shared box. One explanation, not confirmed by a trace: the kernel serializes
buffered writes to one ext4 file on its inode lock, so the copies into the page
cache happen one at a time, whatever the thread count. What this crate
parallelizes is the work in `fill`. Only one thread is ever writing, but the
others can compute their blocks meanwhile. The file layout is what makes that
possible, because no thread waits for another's columns.

Two pitfalls showed up in these numbers:

- **The filesystem matters more than the threads.** On the container's overlay
  root (`/tmp`) every thread count ran at 0.07 GB/s. `/dev/shm` is 64 MB in the
  container, too small for the demo.
- **Overwriting a large file costs its truncation.** Timing `create` over the
  previous run's 0.27 GB file gave 0.1–0.35 GB/s at 24 threads. The demo now
  unlinks the file before each timed run.

Two siblings write the same file, byte for byte, in other ways:

- [`../rayon_npy_mmap`](../rayon_npy_mmap/README.md) stores into a shared
  mapping, and that does scale: 3.6 GB/s on 8 threads. Like this crate's
  numbers, that is the rate of handing the data to the page cache, not of
  reaching the disk.
- [`../rayon_npy_direct`](../rayon_npy_direct/README.md) writes with `O_DIRECT`,
  bypassing the page cache. On this box's shared SSD it is bounded by the device
  at about 0.08 GB/s. Timed through `fdatasync`, all three writers tie at that
  rate.

Untried: one file per thread joined afterwards.
