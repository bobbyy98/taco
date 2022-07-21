#ifndef TACO_WORKSPACE_REWRITER_H
#define TACO_WORKSPACE_REWRITER_H

#include <map>
#include <vector>

namespace taco {
  class TensorVar;

  namespace ir {
    class Stmt;
    class Expr;
  } // namespace ir

  /// Rewrite a post-lowered IR statement to take into account multidimensional temporaries.
  /// Replaces Dimension GetProperty nodes that correspond to temporary workspaces with
  /// their corresponding dimension found in the temporarySizeMap.
  ir::Stmt rewriteTemporaryGP(const ir::Stmt &stmt, std::vector<TensorVar> whereTemps, std::map<TensorVar, std::vector<ir::Expr>> temporarySizeMap);

} // namespace taco
#endif
