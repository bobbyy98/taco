//
// Created by Bobby Yan on 8/1/22.
//

#ifndef SPARSE_ML_MODULE_H
#define SPARSE_ML_MODULE_H

#include <taco/tensor.h>

#include <string>

template <typename T>
class Module {
 public:
  bool training = true;
  std::string name;
  virtual void forward(taco::Tensor<float> &output,
                       taco::Tensor<float> &input) = 0;
  virtual void backward(taco::Tensor<float> &grad_input,
                        taco::Tensor<float> &grad_output) = 0;
  virtual void update(float learning_rate) = 0;
  virtual void save(std::string filename) = 0;
  virtual void load(std::string filename) = 0;
};

#endif  // SPARSE_ML_MODULE_H
