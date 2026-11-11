//! `npy-mmap-demo [ROWS [COLS [COLS_PER_CHUNK [PATH [REPS]]]]]`: fills a
//! ROWS x COLS column-major float64 `.npy` with a trivial `fill` (so the
//! write is what is timed), by `../rayon_npy`'s `pwrite` and by this crate's
//! mapping, at each thread count, best of REPS. The file is unlinked before
//! each run and checked value by value after each method's widest run.

use rayon::ThreadPoolBuilder;
use std::io;
use std::path::{Path, PathBuf};
use std::time::Instant;

type Writer = fn(&Path, usize, usize, usize, &(dyn Fn(usize, &mut [f64]) + Sync)) -> io::Result<()>;

fn value(i: usize, j: usize, rows: usize) -> f64 {
    (i + j * rows) as f64
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
    let cpc = num(2, 16);
    let path = args
        .get(3)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("fb_rayon_npy_mmap_demo.npy"));
    let reps = num(4, 3);
    let gb = (rows * cols * 8) as f64 / 1e9;
    println!(
        "{rows} x {cols} float64 ({gb:.2} GB), {cpc} columns per task, best of {reps}, to {}",
        path.display()
    );

    let fill = move |first: usize, block: &mut [f64]| {
        for (jj, col) in block.chunks_mut(rows).enumerate() {
            for (i, x) in col.iter_mut().enumerate() {
                *x = value(i, first + jj, rows);
            }
        }
    };
    let methods: [(&str, Writer); 2] = [
        ("pwrite", |p, r, c, k, f| {
            fb_rayon_npy::fill_par(p, r, c, k, f)
        }),
        ("mmap", |p, r, c, k, f| {
            fb_rayon_npy_mmap::fill_par(p, r, c, k, f)
        }),
    ];

    let max = rayon::current_num_threads();
    let mut sweep: Vec<usize> = [1, 2, 4, 8, 16].into_iter().filter(|&t| t < max).collect();
    sweep.push(max);
    println!("threads  {:>16}  {:>16}", methods[0].0, methods[1].0);
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
                    pool.install(|| write(&path, rows, cols, cpc, &fill))
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
        .filter(|&(k, &x)| x != value(k % rows, k / rows, rows))
        .count();
    assert_eq!(wrong, 0, "{wrong} values differ on read-back");
}
