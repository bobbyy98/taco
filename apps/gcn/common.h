#ifndef COMMON_H
#define COMMON_H

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <optional>
#include <random>
#include <vector>

#include "taco.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/tensor_operator.h"
#include "taco/tensor.h"

template <typename T>
void print(std::vector<T> const &input) {
  for (auto &i : input) {
    std::cout << i << " ";
  }
  std::cout << std::endl;
}

#endif  // COMMON_H
