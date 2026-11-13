//! No launch wrapper under rust/ allocates, synchronizes, records or waits on
//! an event, copies, sets or reads the current device, or opens a profiler
//! range: a wrapper only enqueues (`docs/gpu_layer.md`, "Layering"). The
//! grep the design promises, over every `.cu` and `.cuh` under rust/, with
//! comments removed, in every spelling a call can take: the vendors'
//! (`cudaMalloc`, `hipMalloc`), wwr's C++ (`wwrMalloc`) and its C ABI's
//! (`wwr_malloc`). Tier 3: it reads files and nothing else.

use std::path::{Path, PathBuf};

/// The forbidden calls, after a `cuda`, `hip` or `wwr` prefix (camel case).
const CAMEL: &[&str] = &[
    // Allocation.
    "Malloc",
    "Free",
    "HostAlloc",
    "HostRegister",
    "MemPool",
    // Synchronization, and the event calls.
    "DeviceSynchronize",
    "StreamSynchronize",
    "StreamQuery",
    "StreamWaitEvent",
    "Event",
    // Copies, a read-back among them.
    "Memcpy",
    // The current device.
    "SetDevice",
    "GetDevice",
    "DeviceReset",
    "DeviceSetLimit",
    // Capture: the orchestrator's.
    "StreamBeginCapture",
    "StreamEndCapture",
];

/// The same calls in the C ABI's spelling, after `wwr_`.
const SNAKE: &[&str] = &[
    "malloc",
    "free",
    "host_alloc",
    "host_register",
    "mem_pool",
    "device_synchronize",
    "stream_synchronize",
    "stream_query",
    "stream_wait_event",
    "event",
    "memcpy",
    "set_device",
    "get_device",
    "device_reset",
    "device_set_limit",
    "stream_begin_capture",
    "stream_end_capture",
];

/// Profiler ranges, NVTX's and rocTX's and wwr's.
const RANGES: &[&str] = &["nvtxRange", "roctxRange", "wwrTxRange", "wwr_tx_range"];

/// `source` with `//` and `/* */` comments blanked, newlines kept.
fn strip_comments(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let mut rest = source;
    while !rest.is_empty() {
        if let Some(r) = rest.strip_prefix("//") {
            let end = r.find('\n').unwrap_or(r.len());
            rest = &r[end..];
        } else if let Some(r) = rest.strip_prefix("/*") {
            let end = r.find("*/").map_or(r.len(), |e| e + 2);
            out.extend(r[..end].chars().filter(|&c| c == '\n'));
            rest = &r[end..];
        } else if rest.starts_with('"') {
            // A string literal is kept whole, so a "//" inside one is not a
            // comment.
            let end = rest[1..].find(['"', '\n']).map_or(rest.len(), |e| e + 2);
            out.push_str(&rest[..end]);
            rest = &rest[end..];
        } else {
            let c = rest.chars().next().unwrap();
            out.push(c);
            rest = &rest[c.len_utf8()..];
        }
    }
    out
}

/// Each forbidden call in `source`, with its line number.
fn violations(source: &str) -> Vec<(usize, String)> {
    let mut found = Vec::new();
    for (i, line) in strip_comments(source).lines().enumerate() {
        for word in line.split(|c: char| !(c.is_ascii_alphanumeric() || c == '_')) {
            let camel = ["cuda", "hip", "wwr"].iter().any(|p| {
                word.strip_prefix(p)
                    .is_some_and(|rest| CAMEL.iter().any(|n| rest.starts_with(n)))
            });
            let snake = word
                .strip_prefix("wwr_")
                .is_some_and(|rest| SNAKE.iter().any(|n| rest.starts_with(n)));
            if camel || snake || RANGES.iter().any(|r| word.starts_with(r)) {
                found.push((i + 1, word.to_owned()));
            }
        }
    }
    found
}

/// Every `.cu` and `.cuh` under `dir`, skipping build and target trees.
fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy();
        if path.is_dir() {
            if !(name.starts_with("build") || name == "target" || name.starts_with('.')) {
                sources(&path, out);
            }
        } else if name.ends_with(".cu") || name.ends_with(".cuh") {
            out.push(path);
        }
    }
}

#[test]
fn launch_wrappers_only_enqueue() {
    let rust = Path::new(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut files = Vec::new();
    sources(&rust, &mut files);
    files.sort();
    // At least the merge sort, so a walk that found nothing fails.
    assert!(
        files
            .iter()
            .any(|f| f.ends_with("recipes/kernels/sort/merge_sort.cu")),
        "{files:?}"
    );
    let mut failures = Vec::new();
    for file in &files {
        let source = std::fs::read_to_string(file).unwrap();
        for (line, call) in violations(&source) {
            failures.push(format!("{}:{line}: {call}", file.display()));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// The grep sees each kind of call in each spelling, even in a string with a
/// `//` in it, and nothing in a comment, a launch, or the status query.
#[test]
fn the_grep_sees_what_it_forbids() {
    let source = "\
        cudaMalloc(&p, n);\n\
        x = hipMallocAsync(&p, n, s);\n\
        wwr::wwrStreamSynchronize(s);\n\
        hipEventRecord(e, s);\n\
        cudaMemcpyAsync(d, h, n, k, s);\n\
        wwr_memcpy(d, h, n, k);\n\
        wwr_stream_synchronize(s);\n\
        cudaSetDevice(0);\n\
        nvtxRangePushA(\"x\");\n\
        // cudaMalloc in a comment\n\
        /* hipFree\n in a block */\n\
        puts(\"cudaFree // in a string\");\n\
        kernel<<<g, b, 0, s>>>(p);\n\
        return wwr_get_last_error();\n\
        wwrGetLastError();\n";
    let words: Vec<String> = violations(source).into_iter().map(|(_, w)| w).collect();
    assert_eq!(
        words,
        [
            "cudaMalloc",
            "hipMallocAsync",
            "wwrStreamSynchronize",
            "hipEventRecord",
            "cudaMemcpyAsync",
            "wwr_memcpy",
            "wwr_stream_synchronize",
            "cudaSetDevice",
            "nvtxRangePushA",
            "cudaFree",
        ]
    );
}
