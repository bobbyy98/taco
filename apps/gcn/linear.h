//
// Created by Bobby Yan on 8/1/22.
//

#ifndef SPARSE_ML_LINEAR_H
#define SPARSE_ML_LINEAR_H

#include "module.h"

using namespace taco;

template <typename T>
class Linear : public Module<T> {
  Linear(int in_channels, int out_channels)
      : in_channels(in_channels), out_channels(out_channels) {
    this->name = "Linear";
    this->weight = Tensor<T>::empty({out_channels, in_channels});
    if (bias) {
      this->bias = Tensor<T>::empty({out_channels});
    }
  }
  void forward(Tensor<T> &output, Tensor<T> &input) {
    // weight: (out_channels, in_channels)
    // input: (batch_size, in_channels)
    // output: (batch_size, out_channels)
    IndexVar i, j, k;
    output(i, k) = input(i, j) * this->weight(k, j);
    if (bias) {
      output(i, k) += this->bias(k);
    }
  }

  void load_weights(std::string &filename, Format &format) {
    //  TODO: implement this
    this->weight = taco::read(filename, format);
  }

  void load_bias(std::string &filename, Format &format) {
    this->bias = taco::read(filename, format);
  }

  std::string name;
  int in_channels;
  int out_channels;
  Tensor<T> weight;
  Tensor<T> bias;
};

#endif  // SPARSE_ML_LINEAR_H
