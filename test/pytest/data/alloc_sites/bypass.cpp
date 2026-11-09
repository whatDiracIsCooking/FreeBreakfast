// Every allocation the pool cap cannot see, one per line, for
// devtools/alloc-sites.py check. test_alloc_sites.py pins the flagged lines;
// it reads this file and is never compiled.
module free_breakfast.fixture;

import free_breakfast.core.wwr;

namespace free_breakfast::fixture {

Status bypasses(DeviceResources &res, void **p, const size_t bytes) {
  FBF_TRY(gpu_status(wwr::wwrMalloc(p, bytes)));
  FBF_TRY(gpu_status(wwr::wwrMallocAsync(p, bytes, res.stream())));
  FBF_TRY(gpu_status(wwr::wwrMallocHost(p, bytes)));
  FBF_TRY(gpu_status(wwr::wwrHostAlloc(p, bytes, 0)));
  cudaMalloc(p, bytes);
  cudaMallocManaged(p, bytes);
  cudaHostAlloc(p, bytes, 0);
  hipMalloc(p, bytes);
  hipHostMalloc(p, bytes);
  Buffers::pinned<double> pinned(bytes / sizeof(double));
  StreamBuffers::pinned<double> streamed(bytes / sizeof(double), res.device_stream());
  auto counted = res.counted_pinned<double>("fixture", bytes / sizeof(double));
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  FBF_TRY(gpu_status(wwr::wwrMemGetInfo(&free_bytes, &total_bytes)));
  // wwr::wwrMalloc(p, bytes) in a comment is prose.
  const char *quoted = "cudaMalloc(p, bytes)";
  Buffers::device<double> pooled(bytes / sizeof(double), res.device());
  return {};
}

} // namespace free_breakfast::fixture
