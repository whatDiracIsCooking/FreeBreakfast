/// @file
/// @brief The routine spike (#1246): a check-before-invoke CRTP base, an
/// allocation key minted only at a root, and mcscf::root_rotation ported
/// onto both. README.md has the questions and the measured answers.

export module free_breakfast.experimental.routine;

export import :key;
export import :base;
export import :root_rotation;
