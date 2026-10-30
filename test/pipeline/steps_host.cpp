// The contract of every step in the registry, in Calculation's order: each is
// a PipelineStep of the run shape the planner schedules it in, reading and
// writing exactly the products it declares. All static_asserts, verified by
// compiling this file; running a step is its test/gtest/*_gpu.cpp's, and
// needs a GPU.

#include <gtest/gtest.h>

import std;
import free_breakfast.pipeline.step;
import free_breakfast.pipeline.config;

namespace {

namespace fb = free_breakfast;

// --- The setup chain --------------------------------------------------------

static_assert(fb::PipelineStep<fb::MoleculeStep>);
static_assert(fb::kRunShape<fb::MoleculeStep> == fb::RunShape::Device);
static_assert(std::same_as<fb::StepInput<fb::MoleculeStep>, fb::View<>>);
static_assert(std::same_as<fb::StepOutput<fb::MoleculeStep>, fb::MoleculeOutput>);

static_assert(fb::PipelineStep<fb::BasisSetStep>);
static_assert(fb::kRunShape<fb::BasisSetStep> == fb::RunShape::Host);
static_assert(std::same_as<fb::StepInput<fb::BasisSetStep>, fb::View<fb::MoleculeOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::BasisSetStep>, fb::BasisSetOutput>);

static_assert(fb::PipelineStep<fb::SphericalTransformStep>);
static_assert(fb::kRunShape<fb::SphericalTransformStep> == fb::RunShape::Device);
static_assert(
    std::same_as<fb::StepInput<fb::SphericalTransformStep>, fb::View<fb::BasisSetOutput>>);
static_assert(
    std::same_as<fb::StepOutput<fb::SphericalTransformStep>, fb::SphericalTransformOutput>);

static_assert(fb::PipelineStep<fb::PrimitivePairClassesStep>);
static_assert(fb::kRunShape<fb::PrimitivePairClassesStep> == fb::RunShape::Device);
static_assert(
    std::same_as<fb::StepInput<fb::PrimitivePairClassesStep>, fb::View<fb::BasisSetOutput>>);
static_assert(
    std::same_as<fb::StepOutput<fb::PrimitivePairClassesStep>, fb::PrimitivePairClassesOutput>);

static_assert(fb::PipelineStep<fb::EriStep>);
static_assert(fb::kRunShape<fb::EriStep> == fb::RunShape::Device);
static_assert(
    std::same_as<fb::StepInput<fb::EriStep>,
                 fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::PrimitivePairClassesOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::EriStep>, fb::EriOutput>);
static_assert(!fb::LoadableStep<fb::EriStep>);
static_assert(fb::EriStep{}.mode == fb::EriMode::IntegralDirect);

static_assert(fb::PipelineStep<fb::OverlapStep>);
static_assert(fb::kRunShape<fb::OverlapStep> == fb::RunShape::Device);
static_assert(std::same_as<fb::StepInput<fb::OverlapStep>,
                           fb::View<fb::BasisSetOutput, fb::SphericalTransformOutput,
                                    fb::PrimitivePairClassesOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::OverlapStep>, fb::OverlapOutput>);

static_assert(fb::PipelineStep<fb::OrthogonalizerStep>);
static_assert(fb::kRunShape<fb::OrthogonalizerStep> == fb::RunShape::Device);
static_assert(std::same_as<fb::StepInput<fb::OrthogonalizerStep>,
                           fb::View<fb::BasisSetOutput, fb::OverlapOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::OrthogonalizerStep>, fb::OrthogonalizerOutput>);

static_assert(fb::PipelineStep<fb::CoreHamiltonianStep>);
static_assert(fb::kRunShape<fb::CoreHamiltonianStep> == fb::RunShape::Device);
static_assert(std::same_as<fb::StepInput<fb::CoreHamiltonianStep>,
                           fb::View<fb::MoleculeOutput, fb::BasisSetOutput,
                                    fb::SphericalTransformOutput, fb::PrimitivePairClassesOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::CoreHamiltonianStep>, fb::CoreHamiltonianOutput>);

// The host work beside the device chain, and the one step with a load.
static_assert(fb::PipelineStep<fb::GuessStep>);
static_assert(fb::kRunShape<fb::GuessStep> == fb::RunShape::HostSubflow);
static_assert(
    std::same_as<fb::StepInput<fb::GuessStep>, fb::View<fb::MoleculeOutput, fb::BasisSetOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::GuessStep>, fb::GuessOutput>);
static_assert(fb::LoadableStep<fb::GuessStep>);
static_assert(fb::kLoadShape<fb::GuessStep> == fb::RunShape::Host);
static_assert(std::same_as<fb::LoadInput<fb::GuessStep>, fb::View<fb::BasisSetOutput>>);

// --- The methods ------------------------------------------------------------

static_assert(fb::PipelineStep<fb::ScfStep>);
static_assert(fb::kRunShape<fb::ScfStep> == fb::RunShape::Device);
static_assert(
    std::same_as<fb::StepInput<fb::ScfStep>,
                 fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
                          fb::EriOutput, fb::OverlapOutput, fb::OrthogonalizerOutput,
                          fb::CoreHamiltonianOutput, fb::GuessOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::ScfStep>, fb::ScfOutput>);

static_assert(fb::PipelineStep<fb::LocalizationStep>);
static_assert(fb::kRunShape<fb::LocalizationStep> == fb::RunShape::Device);
static_assert(std::same_as<
              fb::StepInput<fb::LocalizationStep>,
              fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::OverlapOutput, fb::ScfOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::LocalizationStep>, fb::LocalizationOutput>);

static_assert(fb::PipelineStep<fb::CasciStep>);
static_assert(fb::kRunShape<fb::CasciStep> == fb::RunShape::Device);
static_assert(std::same_as<
              fb::StepInput<fb::CasciStep>,
              fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
                       fb::PrimitivePairClassesOutput, fb::CoreHamiltonianOutput, fb::ScfOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::CasciStep>, fb::CasciOutput>);

static_assert(fb::PipelineStep<fb::CasscfStep>);
static_assert(fb::kRunShape<fb::CasscfStep> == fb::RunShape::Device);
static_assert(
    std::same_as<fb::StepInput<fb::CasscfStep>,
                 fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
                          fb::EriOutput, fb::CoreHamiltonianOutput, fb::ScfOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::CasscfStep>, fb::CasscfOutput>);

static_assert(fb::PipelineStep<fb::FciStep>);
static_assert(fb::kRunShape<fb::FciStep> == fb::RunShape::Device);
static_assert(std::same_as<
              fb::StepInput<fb::FciStep>,
              fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
                       fb::PrimitivePairClassesOutput, fb::CoreHamiltonianOutput, fb::ScfOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::FciStep>, fb::FciOutput>);

static_assert(fb::PipelineStep<fb::Nevpt2Step>);
static_assert(fb::kRunShape<fb::Nevpt2Step> == fb::RunShape::Device);
static_assert(std::same_as<fb::StepInput<fb::Nevpt2Step>,
                           fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::EriOutput,
                                    fb::CoreHamiltonianOutput, fb::ScfOutput, fb::CasscfOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::Nevpt2Step>, fb::Nevpt2Output>);

static_assert(fb::PipelineStep<fb::CisStep>);
static_assert(fb::kRunShape<fb::CisStep> == fb::RunShape::Device);
static_assert(
    std::same_as<fb::StepInput<fb::CisStep>,
                 fb::View<fb::MoleculeOutput, fb::BasisSetOutput, fb::SphericalTransformOutput,
                          fb::PrimitivePairClassesOutput, fb::EriOutput, fb::ScfOutput>>);
static_assert(std::same_as<fb::StepOutput<fb::CisStep>, fb::CisOutput>);

} // namespace
