#ifndef SPARSE_ML_FUNC_H
#define SPARSE_ML_FUNC_H

#include "common.h"

using namespace taco;

struct ReLuAlgebra;
struct ReLULower;
Func reluOp();

template <typename T>
class ReLU;

struct InverseSqrtAlgebra;
struct InverseSqrtLower;
Func inverseSqrtOp();

template <typename T>
Tensor<T> inverseSqrtTensor(Tensor<T> input, Tensor<T> &result) {
  IndexVar i;
  Func inverseSqrt = inverseSqrtOp();
  result(i) = inverseSqrt(input(i));
  return result;
}

#endif  // SPARSE_ML_FUNC_H
