//! What first-touching a fresh slab costs on this machine: the cost of
//! budget mode's touch of B, without the rest of the prototype. No device,
//! no config.
//!
//! ```sh
//! cargo run --release --example touch_probe              # 4 GiB, 8 threads
//! cargo run --release --example touch_probe -- 7.27 16   # gly10's B, 16 threads
//! ```
//!
//! Each row maps a fresh anonymous slab, advises huge pages or not, writes
//! one byte per page from one or more threads, and unmaps it. The first two
//! rows are the same run twice. If the second is much faster, the first
//! paid for something that a slab mapped later in the process does not: on
//! a VM, the host backing guest memory on its first touch. Freshly freed
//! pages are usually reused first, but that is not guaranteed, and a guest
//! that reports free pages to its host gets no speedup either way. If both
//! rows are equally slow, the cost is in the guest's own zeroing.
//!
//! The rows after that show how the touch scales with threads. On bare metal
//! more threads help with 4 KiB pages and change little with huge pages. On
//! the VM page-v0.0.4 measured, they made it slower.

use std::ffi::c_void;
use std::time::Instant;

const PROT_READ: i32 = 1;
const PROT_WRITE: i32 = 2;
const MAP_PRIVATE: i32 = 0x02;
const MAP_ANONYMOUS: i32 = 0x20;
const MADV_HUGEPAGE: i32 = 14;
const SMALL: usize = 4 << 10;
const HUGE: usize = 2 << 20;

unsafe extern "C" {
    fn mmap(addr: *mut c_void, len: usize, prot: i32, flags: i32, fd: i32, off: i64)
    -> *mut c_void;
    fn munmap(addr: *mut c_void, len: usize) -> i32;
    fn madvise(addr: *mut c_void, len: usize, advice: i32) -> i32;
}

/// Map `bytes`, touch every page from `threads` threads, unmap; the seconds
/// from the map to the last touch.
fn touch(bytes: usize, huge: bool, threads: usize) -> f64 {
    let t0 = Instant::now();
    // SAFETY: a fresh private anonymous mapping; nothing else refers to it.
    let p = unsafe {
        mmap(
            std::ptr::null_mut(),
            bytes,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    assert!(p as isize != -1, "mmap of {bytes} bytes failed");
    if huge {
        // SAFETY: the whole mapping just made; advice changes no contents.
        unsafe { madvise(p, bytes, MADV_HUGEPAGE) };
    }
    let step = if huge { HUGE } else { SMALL };
    let chunk = (bytes / threads).next_multiple_of(step);
    let base = p as usize;
    std::thread::scope(|s| {
        for t in 0..threads {
            let (lo, hi) = (t * chunk, ((t + 1) * chunk).min(bytes));
            s.spawn(move || {
                for off in (lo..hi).step_by(step) {
                    // SAFETY: off < bytes, inside the mapping; threads write
                    // disjoint ranges.
                    unsafe { ((base + off) as *mut u8).write_volatile(1) };
                }
            });
        }
    });
    let secs = t0.elapsed().as_secs_f64();
    // SAFETY: the mapping made above, no longer referenced.
    unsafe { munmap(p, bytes) };
    secs
}

fn main() {
    let mut args = std::env::args().skip(1);
    let gib: f64 = args.next().map_or(4.0, |a| a.parse().expect("GiB"));
    let threads: usize = args.next().map_or(8, |a| a.parse().expect("threads"));
    let bytes = ((gib * (1u64 << 30) as f64) as usize).next_multiple_of(HUGE);
    println!("{gib} GiB a slab, {threads} threads where more than one");
    let rows = [
        ("2 MiB, 1 thread (first)", true, 1),
        ("2 MiB, 1 thread (second)", true, 1),
        ("4 KiB, 1 thread", false, 1),
        ("4 KiB, threads", false, threads),
        ("2 MiB, threads", true, threads),
    ];
    for (name, huge, n) in rows {
        let s = touch(bytes, huge, n);
        println!("{name:<26} {s:7.2} s  {:6.2} GiB/s", gib / s);
    }
}
