/// @file
/// @brief The TOML front end: a config file read into the pipeline it asks
///        for, and the registry of every step a config may name.
///
/// One table per step, named by the step's `kName`; its keys are the step's
/// options, spelled as its members. A table is a request: the steps producing
/// what it reads are filled in, each read from an empty table, so only a step
/// with a required option must be written out. A lone `from` key loads the
/// step's product from that file instead, for a step with a `load`.
/// Re-exports the steps and the pipeline it returns.

export module free_breakfast.pipeline.config;

import std;
import free_breakfast.core.error_handling;
export import free_breakfast.pipeline.pipeline;
export import free_breakfast.pipeline.molecule;
export import free_breakfast.pipeline.basis_set;
export import free_breakfast.pipeline.spherical_transform;
export import free_breakfast.pipeline.primitive_pair_classes;
export import free_breakfast.pipeline.eri;
export import free_breakfast.pipeline.overlap;
export import free_breakfast.pipeline.orthogonalizer;
export import free_breakfast.pipeline.core_hamiltonian;
export import free_breakfast.pipeline.guess;
export import free_breakfast.pipeline.scf;
export import free_breakfast.pipeline.localization;
export import free_breakfast.pipeline.casci;
export import free_breakfast.pipeline.casscf;
export import free_breakfast.pipeline.fci;
export import free_breakfast.pipeline.nevpt2;
export import free_breakfast.pipeline.cis;
export import free_breakfast.pipeline.gradient;
export import free_breakfast.pipeline.casscf_gradient;
export import free_breakfast.pipeline.nevpt2_gradient;
export import free_breakfast.pipeline.optimize;

export namespace free_breakfast {

/// @brief The pipeline over every step a config file may name.
using Calculation =
    Pipeline<MoleculeStep, BasisSetStep, SphericalTransformStep, PrimitivePairClassesStep, EriStep,
             OverlapStep, OrthogonalizerStep, CoreHamiltonianStep, GuessStep, ScfStep,
             LocalizationStep, CasciStep, CasscfStep, FciStep, Nevpt2Step, CisStep, GradientStep,
             CasscfGradientStep, Nevpt2GradientStep, OptimizeStep>;

/// @brief Parse a config from an in-memory TOML document and plan the
/// pipeline it asks for.
///
/// Every path in it is an input, and a relative one is resolved against
/// @p input_dir (empty: left relative to the working directory). Every output
/// a planned step can write is written into @p output_dir, named after the
/// step: `scf.molden`. Nothing is created or read on disk.
///
/// A syntactically invalid document is ErrorKind::IO. A document naming no
/// step, an unknown table or key, a value of the wrong type, or a missing
/// required key -- in a written table or in one a filled-in step was read
/// from -- is ErrorKind::InvalidConfig naming it. So is an @p output_dir that
/// is @p input_dir, or holds an input file, compared as the directories they
/// name rather than as spelt. A `[nevpt2]`, a `[casscf_gradient]` or a
/// `[nevpt2_gradient]` under an integral-direct `[eri]`, written or filled
/// in, is ErrorKind::Unsupported naming `[eri]`, and so is a `[cis]` under a
/// density-fitted one. A `[nevpt2_gradient]` over a `[casscf]` of more than
/// one root, a molecule of multiplicity other than 1, or `variant = "pc"` is
/// ErrorKind::Unsupported too, and one whose `root` is not 0
/// ErrorKind::InvalidConfig.
[[nodiscard]] Result<Calculation> parse_config(std::string_view toml_text,
                                               const std::filesystem::path &input_dir,
                                               const std::filesystem::path &output_dir);

/// @brief parse_config of the file at @p path, then @p output_dir created if
/// it does not exist, so a run's paths fail before it starts. A file that
/// cannot be read, or a directory that cannot be created, is ErrorKind::IO.
[[nodiscard]] Result<Calculation> read_config(const std::filesystem::path &path,
                                              const std::filesystem::path &input_dir,
                                              const std::filesystem::path &output_dir);

} // namespace free_breakfast
