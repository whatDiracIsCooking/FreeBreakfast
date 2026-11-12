/// @file
/// @brief IrcStep: geometry::irc down each side of the transition state over
/// the chain OptimizeStep reruns, one ChainEvaluator per side, and the
/// trajectories and history it writes.

module free_breakfast.pipeline.optimize;

import std;
import free_breakfast.core.types;
import free_breakfast.core.error_handling;
import free_breakfast.core.constants; // kAngstromPerBohr
import free_breakfast.core.log;
import free_breakfast.core.resources;
import free_breakfast.io.base;
import free_breakfast.io.json;

#include "diagnostics.h"

namespace free_breakfast {

Result<IrcOutput> IrcStep::run(View<MoleculeOutput, HessianOutput> in, NodeResources &node) const {
  if (node.device_count() == 0) {
    return err_gpu("[{}] needs a GPU device, and none is visible", kName);
  }
  const Molecule &ts = in.get<MoleculeOutput>().molecule;
  const HessianOutput &start = in.get<HessianOutput>();
  const size_t n = 3 * ts.atoms().size();
  if (start.hessian.size() != n * n) {
    return err_config("[{}] starts from the [{}] of the molecule's {} atoms, {} x {}, and it holds "
                      "{} elements",
                      kName, HessianStep::kName, ts.atoms().size(), n, n, start.hessian.size());
  }
  // The frequencies come ascending, an imaginary one negative: the first is
  // the mode left along.
  const std::vector<double> &frequencies = start.vibrations.frequencies;
  const auto imaginary =
      static_cast<size_t>(std::ranges::count_if(frequencies, [](double f) { return f < 0.0; }));
  if (imaginary == 0) {
    return err_config("[{}] starts from a transition state, and the [{}] has no imaginary "
                      "frequency{}: the molecule is not a saddle point",
                      kName, HessianStep::kName,
                      frequencies.empty()
                          ? std::string{}
                          : std::format(" (the lowest is {:.2f} cm^-1)", frequencies.front()));
  }
  if (imaginary > 1 && !allow_multiple_imaginary) {
    std::string listed;
    for (size_t j = 0; j < imaginary; ++j) {
      listed += std::format("{}{:.2f}i", j == 0 ? "" : ", ", -frequencies[j]);
    }
    return err_config("[{}] leaves a transition state along its one imaginary mode, and the [{}] "
                      "has {} ({} cm^-1): set [{}] allow_multiple_imaginary = true to leave along "
                      "the largest",
                      kName, HessianStep::kName, imaginary, listed, kName);
  }
  if (start.vibrations.modes.size() < n) {
    return err_config("[{}] leaves along the [{}]'s imaginary mode, and it holds no modes", kName,
                      HessianStep::kName);
  }
  const std::span<const double> mode(start.vibrations.modes.data(), n);
  FBF_LOG_INFO("irc: leaving along the {:.2f}i cm^-1 mode ({} imaginary), step {} amu^1/2 bohr",
               -frequencies.front(), imaginary, options.step_size);

  json doc = {{"step_size", options.step_size},
              {"imaginary_frequency", frequencies.front()},
              {"imaginary_count", imaginary}};
  const auto write_history = [&]() -> Status {
    if (!history.empty()) {
      FBF_TRY(write_json(history, doc));
    }
    return {};
  };
  IrcOutput out;
  struct Side {
    std::string_view name;
    double sign;
    const std::filesystem::path &trajectory;
    std::optional<geometry::IrcPath> &path;
  };
  std::vector<Side> sides;
  if (direction != IrcDirection::Backward) {
    sides.push_back({"forward", 1.0, forward_trajectory, out.forward});
  }
  if (direction != IrcDirection::Forward) {
    sides.push_back({"backward", -1.0, backward_trajectory, out.backward});
  }
  for (const Side &side : sides) {
    // A fresh evaluator per side: its first evaluation, at the TS, starts
    // the chain from scratch, so the backward side's orbitals are the TS's,
    // not those of the forward side's end.
    const ChainEvaluator evaluate = FBF_TRY(chain_evaluator(chain, node, ts, kName));
    json record = {{"points", json::array()}};
    std::string frames;
    const auto on_point = [&](const geometry::IrcPoint &point) -> Status {
      const double s = side.sign * point.arc_length;
      const size_t k = record["points"].size();
      json rows = json::array();
      for (size_t a = 0; a + 2 < point.x.size(); a += 3) {
        rows.push_back({point.x[a] * kAngstromPerBohr<double>,
                        point.x[a + 1] * kAngstromPerBohr<double>,
                        point.x[a + 2] * kAngstromPerBohr<double>});
      }
      json entry = {{"arc_length", s},
                    {"energy", point.energy},
                    {"gradient_norm", point.gradient_norm},
                    {"max_force", point.max_force},
                    {"sphere_iterations", point.sphere_iterations},
                    {"tangential_gradient", point.tangential_gradient},
                    {"geometry", std::move(rows)}};
      entry.update(evaluate.record(evaluate.evaluations() - 1));
      record["points"].push_back(std::move(entry));
      FBF_LOG_INFO("irc: {} point {:>3} s {:+.4f} E {:.10f} |g| {:.3e} ({} evaluations)", side.name,
                   k, s, point.energy, point.gradient_norm, point.sphere_iterations);
      frames +=
          xyz_frame(placed(ts.atoms(), point.x), std::format("{} point {} s = {:.6f} E = {:.12f}",
                                                             side.name, k, s, point.energy));
      if (!side.trajectory.empty()) {
        FBF_TRY(write_file(side.trajectory, frames));
      }
      doc[side.name] = record;
      return write_history();
    };
    std::vector<double> leave(mode.begin(), mode.end());
    for (double &value : leave) {
      value *= side.sign;
    }
    Result<geometry::IrcPath> walk =
        geometry::irc(ts.atoms(), start.hessian, leave, evaluate, options, on_point);
    if (!walk) {
      record["error"] = walk.error().message;
      doc[side.name] = record;
      const Status written = write_history();
      if (!written) {
        FBF_LOG_ERROR("irc: the failed walk's history is not written: {}", written.error().message);
      }
      return std::unexpected(walk.error());
    }
    record["stop"] = geometry::irc_stop_name(walk->stop);
    doc[side.name] = record;
    FBF_TRY(write_history());
    FBF_LOG_INFO("irc: {} side stopped ({}) after {} points, E {:.10f} -> {:.10f}", side.name,
                 geometry::irc_stop_name(walk->stop), walk->points.size() - 1,
                 walk->points.front().energy, walk->points.back().energy);
    side.path = std::move(*walk);
  }
  return out;
}

MemoryFootprint IrcStep::footprint(const FootprintDimensions &dims) const {
  return chain.footprint(dims);
}

} // namespace free_breakfast
