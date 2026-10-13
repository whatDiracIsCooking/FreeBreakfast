/// @file
/// @brief The `import` front end for the atom types: re-exports Element and
///        Atom from the headers.
///
/// element.h / atom.h hold the single definitions (for CUDA `#include`). This
/// module includes them in its global module fragment and re-exports the names.
///
/// The re-export must be written *inside* `namespace free_breakfast`
/// (not as `export using free_breakfast::Element;` at global scope):
/// a name brought in from the global module fragment is only usable by importers
/// when the re-exporting using-declaration sits in the namespace that owns the
/// name. At global scope importers see the name but cannot form a declaration
/// with it ("declaration is not visible"). Because we re-export the header's own
/// entity, `import`ers and CUDA `#include`rs share one type, not two.
module;

#include "atom.h"

export module free_breakfast.atom;

export namespace free_breakfast {

using free_breakfast::Atom;
using free_breakfast::Element;

} // namespace free_breakfast
