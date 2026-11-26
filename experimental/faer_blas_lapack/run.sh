#!/usr/bin/env bash
# run.sh: build both images, then run eigh-bench over every matrix in
# results/matrices/ at 1, 4 and 12 threads, faer and LAPACK interleaved at
# each count. A t-thread run is pinned to CPUs 0..t-1: on this 5900X those
# are distinct physical cores (12..23 are their SMT siblings), so no run
# shares a core. Results go to results/run/ (an earlier run/ is moved aside
# to run.<timestamp>/); compare.py reads them. Host side: needs docker only.
set -euo pipefail

here=$(cd "$(dirname "$0")" && pwd)
results="$here/results"
matrices_dir="$results/matrices"
threads_list=${THREADS:-"1 4 12"}

shopt -s nullglob
# Smallest file first, so the quick cases report before the slow ones.
mapfile -t matrices < <(ls -Sr "$matrices_dir"/*.npy 2>/dev/null)
if ((${#matrices[@]} == 0)); then
    echo "run.sh: no matrices in $matrices_dir (README.md, \"Matrices\")" >&2
    exit 2
fi
in_container=()
for m in "${matrices[@]}"; do
    in_container+=("/results/matrices/$(basename "$m")")
done

for backend in faer lapack; do
    docker build -q -t "fb-eigh:$backend" -f "$here/Dockerfile.$backend" "$here" >/dev/null
done

if [[ -d $results/run ]]; then
    mv "$results/run" "$results/run.$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$results/run"
echo "load before: $(cut -d' ' -f1-3 /proc/loadavg)" | tee "$results/run/host.txt"

for t in $threads_list; do
    for backend in faer lapack; do
        docker run --rm \
            --cpuset-cpus="0-$((t - 1))" \
            --user "$(id -u):$(id -g)" \
            -e OPENBLAS_NUM_THREADS="$t" \
            -e RAYON_NUM_THREADS="$t" \
            -v "$results:/results" \
            "fb-eigh:$backend" --threads "$t" --out /results/run "${in_container[@]}" \
            2>&1 | tee -a "$results/run/$backend.log"
    done
done
echo "load after: $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$results/run/host.txt"
