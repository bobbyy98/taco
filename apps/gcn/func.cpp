#include "func.h"

using namespace taco;

IndexVar i, j, k, l, m, n;

struct ReLuAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr> &regions) {
    return regions[0];
  }
};

// ReLU: y = max(x, 0)
struct ReLULower {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    taco_iassert(v.size() == 1) << "Operator needs exactly one operand";
    return ir::Max::make(v[0], ir::Literal::zero(v[0].type()));
  }
};

Func reluOp() {
  Func relu("relu", ReLULower(), ReLuAlgebra());
  return relu;
}

template <typename T>
class ReLU {
 public:
  void forward(taco::Tensor<T> &output, taco::Tensor<T> &input) {
    Func relu = reluOp();
    output(i, j) = relu(input(i, j));
  }
};

template <typename T>
Tensor<T> reluTensor(Tensor<T> A) {
  Func relu = reluOp();
  Tensor<T> result({A.getDimensions()}, A.getFormat());
  result(i, j) = relu(A(i, j));
  return result;
}

struct InverseSqrtAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr> &regions) {
    return regions[0];
  }
};

// InverseSqrt: y = 1 / sqrt(x)
struct InverseSqrtLower {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    taco_iassert(v.size() == 1) << "Operator needs exactly one operand";

    return ir::Div::make(ir::Literal::make(1), ir::Sqrt::make(v[0]));
  }
};

Func inverseSqrtOp() {
  Func inverseSqrt("inverseSqrt", InverseSqrtLower(), InverseSqrtAlgebra());
  return inverseSqrt;
}