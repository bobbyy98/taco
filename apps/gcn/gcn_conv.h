//
// Created by Bobby Yan on 8/7/22.
//

#ifndef SPARSE_ML_GCN_CONV_H
#define SPARSE_ML_GCN_CONV_H

#include "linear.h"
#include "module.h"

using namespace taco;

template <typename T>
Tensor<T> gcn_norm(Tensor<T> edge_index, bool add_self_loops = true) {
  //  TODO: implement this
}

template <typename T>
class GCNConv : public Module<T> {
 public:
  /**
   * in_channels (int): Size of each input sample, or :obj:`-1` to derive the
   * size from the first input(s) to the forward method.
   *
   * out_channels (int): Size of each output sample.
   */
  int in_channels, out_channels;
  Linear<T> lin;
  bool add_self_loops;
  /**
   * normalize (bool, optional): Whether to add self-loops and compute
   * symmetric normalization coefficients on the fly.
   * (default: :obj:`True`)
   */
  bool normalize = true;
  /**
   * bias (bool, optional): If set to :obj:`False`, the layer will not learn
   * an additive bias. (default: :obj:`True`)
   */
  bool has_bias = false;
  /**
   * Forward function
   * args: x (Tensor): Input tensor.
   * edge_index
   * edge_weight (optional)
   */
  Tensor<T> forward(Tensor<T> &x, Tensor<T> &edge_index,
                    Tensor<T> &edge_weight = Tensor<T>()) {
    //  normalize if needed
    if (normalize) {
      edge_index = gcn_norm(edge_index, add_self_loops);
    }
    Tensor<T> lin_output = Tensor<T>({x.getDimension(0), out_channels}, COO(2));
    lin.forward(lin_output, x);

    if (has_bias) {
      // TODO: implement this
    }
  }
};

#endif  // SPARSE_ML_GCN_CONV_H
