# System Environment

Queries system hardware at startup: CPU cores, GPU devices, host RAM, and disk
space.

## Module

`free_breakfast.core.sys_env` — target `free_breakfast::core::sys_env`, also
re-exported by `free_breakfast.core`.

## Usage

```cpp
import free_breakfast.core.sys_env;

if (auto env = free_breakfast::query_sys_env()) {
  free_breakfast::log_sys_env(*env);
}
```

## Files

| File | Role |
|------|------|
| `sys_env.cppm` | Module interface exporting `SysEnv`, `GpuInfo`, `query_sys_env()`, `available_gpus()`, `logical_cpu_cores()`, `log_sys_env()`. The GPU roster is `vector<GpuInfo>` — plain data (index, name, memory, compute capability), deliberately not `Device` (`free_breakfast.core.wwr`), which owns a stream and a memory pool; a startup inventory snapshot must not create one per device just to report what is on the box |
| `sys_env.cpp` | Implementation: reads `/proc/cpuinfo` and `/proc/meminfo`; enumerates GPUs through WarpWraps' backend-neutral `wwrGetDeviceCount` / `wwrGetDeviceProperties`, so the same code serves a CUDA and a HIP build (graceful: no device yields an empty roster, not an error) |
| `CMakeLists.txt` | Build definition for the module |

Linux-only: the CPU and RAM figures come from `/proc`. Where `/proc/cpuinfo`
carries no `physical id` / `core id` lines (as on most aarch64 kernels),
`cpu_physical_cores` falls back to the logical count rather than failing the
query, and a debug-level line says so.
