# ---------------------------------------------------------------------------
# Project source
# ---------------------------------------------------------------------------
# Add your own domain code as sibling directories here and wire them in below.
#
# core/ is the starting set, not a fixture: one module library that imports the
# WarpWraps layer and `std`, so a fresh clone has something that actually
# exercises the whole chain -- the three cmake/ macros, C++23 named modules,
# `import std`, and the GPU backend switch. Delete it once this project has real
# modules of its own, or grow it; nothing else depends on it staying small.
add_subdirectory(core)
add_subdirectory(atom)
add_subdirectory(molecule)
add_subdirectory(basis)
add_subdirectory(io)
add_subdirectory(sad)
add_subdirectory(recipes)
# After recipes: one_electron writes into recipes.matrix_ao's CartesianMatrixAO.
add_subdirectory(integrals)
# Before pipeline: its Localization step wraps localization.pipek_mezey.
add_subdirectory(localization)
# Before pipeline: its Localization step reports IAO charges through it.
add_subdirectory(analysis)
# Over core alone for now; recipes, which the CI solver will need, is above.
add_subdirectory(ci)
# After integrals, whose J and K it contracts; before mcscf, whose loop steps
# over its gradient and Hessian, and before pipeline, which it must never
# import.
add_subdirectory(response)
# After integrals, whose J, K and active-space integrals it contracts, and
# response; before pipeline, which it must never import.
add_subdirectory(mcscf)
# After integrals, whose density-fitted MO transform it reads, and ci, whose
# strings its RDMs read; before pipeline, which it must never import.
add_subdirectory(nevpt2)
# After recipes, whose Davidson it preconditions, and integrals, whose J and K
# its sigma will contract; before pipeline, which it must never import.
add_subdirectory(cis)
# Host-only, over core alone; before pipeline, whose optimize step will drive
# it.
add_subdirectory(geometry)
# After the modules its steps wrap: the step contract itself is a leaf over
# core, but the setup steps that follow it import integrals and recipes.
add_subdirectory(pipeline)

# The free-breakfast executable: the one place a value-tier Error becomes an
# exit code. Not a module library, so it sets the module-scanning properties
# itself, as the experimental/ benches do.
add_executable(free_breakfast_main main.cpp)
set_target_properties(
  free_breakfast_main PROPERTIES OUTPUT_NAME free-breakfast CXX_MODULE_STD 1
                                 CXX_SCAN_FOR_MODULES ON
)
target_link_libraries(
  free_breakfast_main
  PRIVATE free_breakfast::core
          free_breakfast::core::resources
          free_breakfast::core::diagnostics_macros
          free_breakfast::pipeline::config
)
