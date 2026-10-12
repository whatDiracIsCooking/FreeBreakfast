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

// The GPU layer comes from the WarpWraps submodule and is backend-neutral: the
// same two imports resolve against src/cuda under a CUDA build and src/hip
// under a HIP one. Re-exported so a consumer of free_breakfast.core gets the
// buffer types without naming wwr itself.
export import wwr.extension.common;
export import wwr.extension.memory_buffer;
