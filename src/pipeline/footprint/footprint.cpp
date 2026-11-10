/// @file
/// @brief check_plan and check_pinned_node: the plan fold held to the
///        `[memory]` caps, and the node's RAM held to the pinned one.

module free_breakfast.pipeline.footprint;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.log;

#include "diagnostics.h"

namespace free_breakfast {

// Bytes in GiB, beside the exact count in a message.
[[nodiscard]] constexpr double plan_gib(const size_t bytes) {
  return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0);
}

Result<PlanFootprint> check_plan(std::span<const StepFootprint> steps, const PlanCaps &caps) {
  std::vector<MemoryFootprint> footprints;
  footprints.reserve(steps.size());
  for (const StepFootprint &step : steps) {
    // Counting it as zero would pass a plan its step may not fit, so a step
    // with no footprint yet is refused under a cap, never skipped.
    if (!step.footprint) {
      return err_config("pre-flight: [{}] has no footprint yet, so the plan cannot be held to "
                        "its [memory] caps; drop the [memory] table to run it uncapped",
                        step.name);
    }
    footprints.push_back(*step.footprint);
  }
  const PlanFootprint plan = plan_footprint(footprints);
  if (steps.empty()) {
    return plan;
  }

  const size_t device_peak = plan.footprint.device.peak();
  const std::string_view device_step = steps[plan.device_peak_step].name;
  if (caps.device_room != 0 && device_peak > caps.device_room) {
    return err_config("pre-flight: [{}] reaches the plan's device peak of {} bytes ({:.3f} GiB), "
                      "past the {} bytes ({:.3f} GiB) max_device_mem_per_gpu = {} leaves the "
                      "plan: the cap less the reserve outside the pool, the fragmentation "
                      "headroom, what the device's resources hold in the pool themselves "
                      "(DeviceResources::kOwnPoolBytes) and, on a pool that rounds reuse, its "
                      "rounding",
                      device_step, device_peak, plan_gib(device_peak), caps.device_room,
                      plan_gib(caps.device_room), caps.max_device_mem_per_gpu);
  }
  const size_t pinned_peak = plan.footprint.pinned.peak();
  const std::string_view pinned_step = steps[plan.pinned_peak_step].name;
  if (caps.max_pinned_mem_per_gpu != 0 && pinned_peak > caps.max_pinned_mem_per_gpu) {
    return err_config("pre-flight: [{}] reaches the plan's pinned peak of {} bytes ({:.3f} GiB), "
                      "past max_pinned_mem_per_gpu = {} ({:.3f} GiB)",
                      pinned_step, pinned_peak, plan_gib(pinned_peak), caps.max_pinned_mem_per_gpu,
                      plan_gib(caps.max_pinned_mem_per_gpu));
  }

  if (caps.device_room != 0) {
    FBF_LOG_INFO("pre-flight: the plan fits: device peak {} bytes ({:.3f} GiB) at [{}], of the {} "
                 "bytes max_device_mem_per_gpu = {} leaves it",
                 device_peak, plan_gib(device_peak), device_step, caps.device_room,
                 caps.max_device_mem_per_gpu);
  }
  if (caps.max_pinned_mem_per_gpu != 0) {
    FBF_LOG_INFO("pre-flight: the plan fits: pinned peak {} bytes ({:.3f} GiB) at [{}], of "
                 "max_pinned_mem_per_gpu = {}",
                 pinned_peak, plan_gib(pinned_peak), pinned_step, caps.max_pinned_mem_per_gpu);
  }
  // The workspace is held at the plan's maximum through every step, so what
  // the peak step leaves of it is held for nothing there: a large gap is a
  // step's one-off carving that belongs in its scratch.
  FBF_LOG_INFO("pre-flight: device workspace {} bytes held throughout; the peak step [{}] uses "
               "{} of it",
               plan.footprint.device.workspace, device_step,
               footprints[plan.device_peak_step].device.workspace);
  FBF_LOG_INFO("pre-flight: pinned workspace {} bytes held throughout; the peak step [{}] uses "
               "{} of it",
               plan.footprint.pinned.workspace, pinned_step,
               footprints[plan.pinned_peak_step].pinned.workspace);

  // What the cap leaves each adaptive step above its least: the room less
  // what is held at its point in the fold, its own least scratch included.
  if (caps.device_room != 0) {
    size_t persistent = 0;
    for (size_t k = 0; k < steps.size(); ++k) {
      const auto &device = footprints[k].device;
      persistent += device.persistent;
      if (!steps[k].adaptive_unit) {
        continue;
      }
      const size_t held = persistent + device.scratch + plan.footprint.device.workspace;
      const size_t above = caps.device_room - held;
      const size_t unit = *steps[k].adaptive_unit;
      if (above < unit) {
        FBF_LOG_WARN("pre-flight: [{}] fits only at its least: the cap leaves it {} bytes above "
                     "its least scratch, under the {} one more unit of its adaptive part takes",
                     steps[k].name, above, unit);
      } else {
        FBF_LOG_INFO("pre-flight: [{}] can take {} bytes ({:.3f} GiB) above its least scratch, "
                     "{} units of its adaptive part",
                     steps[k].name, above, plan_gib(above), unit != 0 ? above / unit : 0);
      }
    }
  }
  return plan;
}

Status check_pinned_node(const size_t max_pinned_mem_per_gpu, const size_t devices,
                         const size_t mem_available) {
  // devices * cap > available, without the product's overflow.
  if (devices != 0 && max_pinned_mem_per_gpu > mem_available / devices) {
    return err_config("pre-flight: {} devices, each allowed max_pinned_mem_per_gpu = {} bytes "
                      "({:.3f} GiB) of pinned memory, can pin more than the node's MemAvailable "
                      "of {} bytes ({:.3f} GiB); pinned memory is not pageable",
                      devices, max_pinned_mem_per_gpu, plan_gib(max_pinned_mem_per_gpu),
                      mem_available, plan_gib(mem_available));
  }
  return {};
}

} // namespace free_breakfast
