//! `npy-demo [ROWS [COLS [COLS_PER_CHUNK [PATH]]]]`: writes a ROWS x COLS
//! column-major float64 `.npy` with `fill_par`, on one thread and then on
//! rayon's default pool, reads each back and checks every value, and prints
//! the write rate of each.

use std::path::PathBuf;
use std::time::Instant;

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
    let cols = num(1, 8192);
    let cpc = num(2, 16);
    let path = args
        .get(3)
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("fb_rayon_npy_demo.npy"));
    let gb = (rows * cols * 8) as f64 / 1e9;
    println!(
        "{rows} x {cols} float64 ({gb:.2} GB), {cpc} columns per task, to {}",
        path.display()
    );

    let fill = |first: usize, block: &mut [f64]| {
        for (jj, col) in block.chunks_mut(rows).enumerate() {
            for (i, x) in col.iter_mut().enumerate() {
                *x = value(i, first + jj, rows);
            }
        }
    };
    let max = rayon::current_num_threads();
    let mut sweep: Vec<usize> = [1, 2, 4, 8, 16].into_iter().filter(|&t| t < max).collect();
    sweep.push(max);
    for threads in sweep {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("thread pool");
        // Unlinked first so the timed region does not include truncating the
        // previous run's pages.
        let _ = std::fs::remove_file(&path);
        let start = Instant::now();
        pool.install(|| fb_rayon_npy::fill_par(&path, rows, cols, cpc, fill))
            .unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let secs = start.elapsed().as_secs_f64();

        let (r, c, values) =
            fb_rayon_npy::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!((r, c), (rows, cols), "shape read back");
        let wrong = values
            .iter()
            .enumerate()
            .filter(|&(k, &x)| x != value(k % rows, k / rows, rows))
            .count();
        assert_eq!(wrong, 0, "{wrong} values differ on read-back");
        println!(
            "{threads:>3} thread(s): {:8.1} ms, {:6.2} GB/s, read back exact",
            secs * 1e3,
            gb / secs
        );
    }
}
