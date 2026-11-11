# rust/rayon_npy_mmap — the same `.npy`, filled in place through a mapping

The sibling of [`../rayon_npy`](../rayon_npy/README.md). It writes the same file,
a column-major float64 `.npy`, byte for byte, with many rayon threads. The
difference is that the threads **store into a shared mapping of the file** rather
than calling `pwrite`. `../rayon_npy` measured flat: 1.6–1.9 GB/s at every thread
count. This one scales to about 3.6 GB/s on 8 threads.

## How

`NpyMap::create(path, rows, cols)` does four things:

1. Writes `../rayon_npy`'s header (`fb_rayon_npy::header`).
2. `set_len`s the file to its final size.
3. `fallocate`s the data range.
4. Maps the whole file `MAP_SHARED`.

`data_mut()` is then the matrix as one `&mut [f64]` over the page cache.
`par_chunks_mut` hands each rayon task a disjoint block of whole columns, and
`fill` computes its values straight into the file's pages.

| Item | What |
|---|---|
| `NpyMap` | `create`, `rows`, `cols`, `data_mut`, `finish` (`msync`); dropping it unmaps without syncing, as `np.save` doesn't sync |
| `fill_par(path, rows, cols, cols_per_chunk, fill)` | Same signature as `../rayon_npy`'s. Each block of the file itself goes to `fill`, with no buffer and no copy |
| `write_par(path, rows, cols, cols_per_chunk, values)` | A matrix in memory, copied in by block |
| `read` | Re-exported from `../rayon_npy` |

**Why it scales where `pwrite` doesn't:** a buffered `write()` to one ext4 file
takes that file's inode lock for the whole copy, so the writers take turns. A
store to a mapped page takes no such lock. The first touch of each page faults,
and faults on a shared mapping run concurrently.

**Safety:** blocks don't overlap because the borrow checker guarantees it, not by
convention. The only `unsafe` is the mapping itself and the cast of the bytes to
`f64` (alignment asserted).

**The cost:** an I/O error after `create` can't come back as an `Err` through a
store to memory; the kernel raises `SIGBUS` instead. The `fallocate` in `create`
moves the common case, a full disk, into `create`'s `Err`. A filesystem without
`fallocate` gets a sparse file and keeps that exposure.

## Build and run

As `../rayon_npy` (cargo by hand; `rayon`, `memmap2` and `libc` from crates.io):

```bash
cd experimental/rust/rayon_npy_mmap
cargo test --release      # byte-identical to ../rayon_npy's file, round trips at
                          # every chunking, 16 threads on 3-row columns (tasks
                          # sharing a page), empty shapes, refusals
cargo run --release -- 4096 32768 16 target/demo.npy 3
      # npy-mmap-demo [ROWS [COLS [COLS_PER_CHUNK [PATH [REPS]]]]]
      # times both crates at each thread count, best of REPS
```

## Measured (2026-11-11)

`npy-mmap-demo 4096 32768 C` (1.07 GB), best of 3. The fill is trivial (`i + j *
rows`), so the write is what is timed, from `create` through unmap, with no sync.
The file is on the container's ext4 workspace mount (NVMe), on a 24-thread box
other worktrees share. Each method's widest run is read back and checked value by
value.

| threads | `pwrite`, C = 16 | `mmap`, C = 16 | `pwrite`, C = 256 | `mmap`, C = 256 |
|---|---|---|---|---|
| 1 | 1.91 GB/s | 1.20 GB/s | 1.86 | 1.15 |
| 2 | 1.99 | 1.90 | 1.84 | 1.79 |
| 4 | 1.92 | 3.11 | 1.73 | 2.88 |
| 8 | 1.85 | **3.61** | 1.67 | **3.60** |
| 16 | 1.70 | 3.61 | 1.61 | 3.54 |
| 24 | 1.63 | 3.29 | 1.64 | 3.44 |

From 8 threads up the mapping is **3× its own single thread and 1.9× `pwrite`'s
best**. On one thread it is the slower of the two, because a fault per 4 KiB page
costs more than one bulk copy. So `../rayon_npy` remains the choice for a single
writer.

### Where the ceiling is

A per-phase timing of the 1 GB run (a temporary hack, since removed):

- `create` (header, `set_len`, `fallocate`): about 1 ms.
- Unmapping: 55–70 ms at every thread count.
- Filling the pages: 873 ms on 1 thread, 226 ms at its best on 8, then
  250–265 ms at 16 and 24.

Two attempts to raise the plateau changed nothing:

- **`madvise(MADV_POPULATE_WRITE)`** on each block before filling it, which
  replaces one fault per page with one call per block: 231 ms at 8 threads, the
  same.
- **No `fallocate`** (a sparse file, with blocks allocated at writeback): 252 ms
  at 8 threads, the same.

So the limit is not fault overhead and not block allocation, but the kernel
inserting about 260 000 new pages into one file's page cache. That index is
still per-file, which is the likely cap (not traced). Spreading across files
would avoid it, but that is not one `.npy`.

Untried: `O_DIRECT` with aligned blocks, which skips the page cache and so
measures the device rather than memory.
