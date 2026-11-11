# rust/rayon_npy_direct — the same `.npy`, written with `O_DIRECT`

The third sibling, after [`../rayon_npy`](../rayon_npy/README.md) (`pwrite`) and
[`../rayon_npy_mmap`](../rayon_npy_mmap/README.md) (a shared mapping). It writes
the same file, byte for byte, from many rayon threads, with each write going from
the thread's buffer straight to the device, bypassing the page cache.

**Result:** on this machine's disk `O_DIRECT` is bounded by the device, at about
0.08 GB/s. Threads take it from 0.04 to 0.09 GB/s, which matches `dd`'s rate on
the same filesystem. Once the data must actually reach the disk, all three
writers run at that same rate at every thread count. The mapping's 3.6 GB/s is a
page-cache figure: it measures memory, not the disk.

## How

`O_DIRECT` needs every write's file offset, length and buffer address aligned to
the device's logical block, taken here as 4 KiB (`ALIGN`). Whole-column blocks
are almost never aligned: a column is `rows * 8` bytes, and the data starts
after a 64- or 128-byte header. So tasks split the **file** rather than the
matrix:

- Task `s` owns bytes `[s * stripe, (s + 1) * stripe)` of the file, which hold a
  contiguous run of column-major elements.
- Stripe 0 also carries the header.
- Each thread fills a reused, page-aligned `Vec<Page>` (`#[repr(align(4096))]`,
  so the aligned allocation needs no `unsafe`) and writes it in one `pwrite`.

| Item | What |
|---|---|
| `fill_par(path, rows, cols, stripe_bytes, fill)` | **A different `fill` from the siblings':** `fill(first, values)` fills the flat column-major elements `first..first + values.len()`, element `k` being row `k % rows` of column `k / rows`. A range may start and end mid-column |
| `write_par(path, rows, cols, stripe_bytes, values)` | A matrix in memory, copied into the aligned buffers (`values` itself is not page-aligned in general) |
| `read` | Re-exported from `../rayon_npy` |

Before the first write, the file is extended to its length rounded up to 4 KiB
and `fallocate`d. It is truncated to its exact length at the end. As I read
ext4's `ext4_dio_write_checks`, a direct write runs under the inode lock
*shared* only when it neither extends the file nor writes into a hole.
Otherwise it takes the lock exclusive and the writers take turns. **That is from
the kernel source, not from a measurement here:** this disk is too slow for the
lock to show. A filesystem without `fallocate` is refused rather than silently
serialized.

When `fill_par` returns, the data has been handed to the device, not
necessarily through its write cache. The metadata that converts the allocated
blocks to written is not synced, so `fdatasync` is still needed for a crash.

## Build and run

As the siblings (cargo by hand; `rayon` and `libc` from crates.io, the two
sibling crates by path, the mapping one only for the demo's comparison):

```bash
cd experimental/rust/rayon_npy_direct
cargo test --release      # byte-identical to ../rayon_npy's file over 6 shapes x 3
                          # stripe sizes (data ending in stripe 0, on a page,
                          # mid-page), 16 threads on one-page stripes, empty
                          # shapes, refusals. Scratch under target/, not /tmp:
                          # the container's /tmp is overlayfs
cargo run --release -- 4096 8192 1024 target/demo.npy 1 1
      # npy-direct-demo [ROWS [COLS [STRIPE_KIB [PATH [REPS [SYNC]]]]]]
      # all three writers per thread count; SYNC=1 times each through fdatasync
```

## Measured (2026-11-11)

The file is on the container's ext4 workspace, a Samsung SSD 980 1TB (DRAM-less,
69% full) that every worktree on the box shares. It wrote 131 MB in a 5-second
window while this was idle, 38% busy. Baseline for 1 GiB on the same
filesystem: `dd bs=1M oflag=direct` 85.7 MB/s; `dd bs=1M conv=fdatasync`
74.8 MB/s.

**No sync**, 4096 x 32768 (1.07 GB), 1 MiB per task (32 columns for the column
writers), best of 3:

| threads | `pwrite` | `mmap` | `O_DIRECT` |
|---|---|---|---|
| 1 | 1.62 GB/s | 0.88 | 0.04 |
| 2 | 1.21 | 1.50 | 0.07 |
| 4 | 1.24 | 2.13 | 0.07 |
| 8 | 1.08 | 2.81 | 0.08 |
| 16 | 1.60 | 3.40 | 0.08 |
| 24 | 1.58 | 3.35 | 0.09 |

**Through `fdatasync`**, 4096 x 8192 (0.27 GB), 1 MiB per task, one run each (to
spare the disk):

| threads | `pwrite` | `mmap` | `O_DIRECT` |
|---|---|---|---|
| 1 | 0.07 GB/s | 0.07 | 0.08 |
| 4 | 0.08 | 0.08 | 0.08 |
| 24 | 0.08 | 0.09 | 0.07 |

The rows left out (2, 8 and 16 threads) are the same, 0.06–0.09.

What the tables say:

- **`O_DIRECT` reaches the device's rate and stops there.** Threads help from 1
  to about 4, because one thread keeps only one 1 MiB write in flight and more
  threads deepen the queue. After that the disk is the limit.
- **With a sync, the comparison is fair, and all three tie.** The two
  page-cache writers' speed without a sync was the speed of handing the data to
  the kernel. That is real when the caller does not wait for the disk, as
  `np.save` does not.
- **`O_DIRECT` would only earn its keep on a device faster than memory
  bandwidth into the page cache** (a fast, idle NVMe or an array), or when the
  page cache must not be polluted. Neither applies here. The box's other SSD
  (970 EVO) is not mounted in the container, so it is untried.
