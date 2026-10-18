// free_breakfast.core -- the aggregating module for this project's core.
//
// `import free_breakfast.core;` pulls in everything re-exported here. Add an
// `export import` line below for each module that is part of the public
// surface, and leave the rest to be imported directly.
//
// This is the starting set: it exists so a fresh clone has a target that
// actually compiles a C++23 module interface, resolves `import std`, and links
// the WarpWraps layer for whichever backend the build selected. Replace its
// contents with real modules.
export module free_breakfast.core;

export import :version;

// Vocabulary with no runtime of its own: fixed-width integers, the builtin
// vector-type mapping and the scalar concepts (re-exported from wwr), typed
// numeric / physical constants, the math and integer-rounding primitives in
// free_breakfast::math, the launch-shape constants (kWarpSize, NumWarps), and
// the copy/move mixins.
export import free_breakfast.core.types;
export import free_breakfast.core.constants;
export import free_breakfast.core.primitives;
export import free_breakfast.core.kernel;
export import free_breakfast.core.nontransferrable;

// Separate modules rather than partitions: each has consumers that want it
// without the rest (log has no reason to drag in the GPU layer), and the
// macros in diagnostics/diagnostics.h name them individually. error_handling
// also carries check_or_abort, the abort tier.
export import free_breakfast.core.error_handling;
export import free_breakfast.core.log;

// The interface to WarpWraps (the submodule, backend-neutral -- the same
// imports resolve against src/cuda under a CUDA build and src/hip under a HIP
// one): gpu_status, this build's GPU error policies, and wwr's RAII wrappers
// bound to them. free_breakfast.core.wwr re-exports every wwr module the tree
// uses, so a consumer of free_breakfast.core gets them without naming wwr
// itself.
export import free_breakfast.core.wwr;

// MirroredBuffer: a device and a pinned buffer paired for h2d / d2h. Its own
// module, not a partition of core.wwr, because it is built from Buffers, which
// core.wwr's primary interface defines.
export import free_breakfast.core.mirrored_buffer;

// Deliberately absent: free_breakfast.core.resources (NodeResources). It
// re-exports taskflow, which every importer of this module would then compile
// against. Import it explicitly where it is used.
