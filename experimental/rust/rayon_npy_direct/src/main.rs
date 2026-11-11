//! `npy-direct-demo [ROWS [COLS [STRIPE_KIB [PATH [REPS [SYNC]]]]]]`: writes a
//! ROWS x COLS column-major float64 `.npy` with a trivial fill by all three
//! writers (`../rayon_npy`'s `pwrite`, `../rayon_npy_mmap`'s mapping, this
//! crate's `O_DIRECT`) at each thread count, best of REPS, each task about
//! STRIPE_KIB of the file. With SYNC = 1 every run also includes an
//! `fdatasync`, so the page-cache writers are timed to the device too, as
//! `O_DIRECT` is by construction. The file is unlinked before each run and
//! checked value by value after each method's widest run.

use rayon::ThreadPoolBuilder;
use std::fs::File;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// `(path, rows, cols, stripe_bytes)`; each method turns the stripe into its
/// own task size.
type Writer = fn(&Path, usize, usize, usize) -> io::Result<()>;

fn flat(first: usize, out: &mut [f64]) {
    for (t, x) in out.iter_mut().enumerate() {
        *x = (first + t) as f64;
    }
}

fn cols_per_task(rows: usize, stripe: usize) -> usize {
    (stripe / (rows * 8).max(1)).max(1)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let num = |k: usize, default: usize| {
        args.get(k)
            .map(|a| {
                a.parse()
                    .unwrap_or_else(|_| panic!("argument {} is not a count: {a}", k + 1))
            })
            .unwrap_or(default)
    };
    let rows = num(0, 4096);
    let cols = num(1, 32768);
    let stripe = num(2, 1024) * 1024;
    let path = args
        .get(3)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("fb_rayon_npy_direct_demo.npy"));
    let reps = num(4, 3);
    let sync = num(5, 0) != 0;
    let gb = (rows * cols * 8) as f64 / 1e9;
    println!(
        "{rows} x {cols} float64 ({gb:.2} GB), {} KiB per task ({} columns for the column \
         writers), best of {reps}, {}, to {}",
        stripe / 1024,
        cols_per_task(rows, stripe),
        if sync { "through fdatasync" } else { "no sync" },
        path.display()
    );

    let methods: [(&str, Writer); 3] = [
        ("pwrite", |p, r, c, s| {
            fb_rayon_npy::fill_par(p, r, c, cols_per_task(r, s), |j, b| flat(j * r, b))
        }),
        ("mmap", |p, r, c, s| {
            fb_rayon_npy_mmap::fill_par(p, r, c, cols_per_task(r, s), |j, b| flat(j * r, b))
        }),
        ("O_DIRECT", |p, r, c, s| {
            fb_rayon_npy_direct::fill_par(p, r, c, s, flat)
        }),
    ];

    let max = rayon::current_num_threads();
    let mut sweep: Vec<usize> = [1, 2, 4, 8, 16].into_iter().filter(|&t| t < max).collect();
    sweep.push(max);
    let mut head = "threads".to_string();
    for (name, _) in &methods {
        head += &format!("  {name:>19}");
    }
    println!("{head}");
    for &threads in &sweep {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool");
        let mut line = format!("{threads:>7}");
        for (_, write) in methods {
            let best = (0..reps)
                .map(|_| {
                    let _ = std::fs::remove_file(&path);
                    let start = Instant::now();
                    pool.install(|| write(&path, rows, cols, stripe))
                        .and_then(|()| match sync {
                            true => File::open(&path)?.sync_data(),
                            false => Ok(()),
                        })
                        .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                    start.elapsed().as_secs_f64()
                })
                .fold(f64::INFINITY, f64::min);
            line += &format!("  {:7.1} ms {:5.2} GB/s", best * 1e3, gb / best);
            if threads == max {
                check(&path, rows, cols);
            }
        }
        println!("{line}");
    }
    println!("widest run of each method read back exact");
    let _ = std::fs::remove_file(&path);
}

fn check(path: &Path, rows: usize, cols: usize) {
    let (r, c, values) =
        fb_rayon_npy::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!((r, c), (rows, cols), "shape read back");
    let wrong = values
        .iter()
        .enumerate()
        .filter(|&(k, &x)| x != k as f64)
        .count();
    assert_eq!(wrong, 0, "{wrong} values differ on read-back");
}
