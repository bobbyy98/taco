#include <chrono>
#include <filesystem>
#include <iostream>

#include "common.h"
#include "func.h"
#include "module.h"

using namespace taco;
using namespace std;
using namespace std::chrono;

auto dataset = "pubmed";
bool VERBOSE = false;

auto build_path = std::__fs::filesystem::current_path();
auto apps_path = build_path.string() + "/../../apps";
// Concat the current path with the relative path to the data folder
auto weights_dir_path = apps_path + "/weights/" + dataset + "_gcn";
auto all_datasets_dir_path = apps_path + "/datasets";
auto dataset_dir_path = all_datasets_dir_path + "/" + dataset;

auto edge_index_path = dataset_dir_path + "/edge_index.mtx";
auto adj_matrix_path = dataset_dir_path + "/adj_matrix.tns";
auto data_x_path = dataset_dir_path + "/x.mtx";

template <typename T>
Tensor<T> dot(Tensor<T> A, Tensor<T> B) {
  assert(A.getDimension(1) == B.getDimension(0));
  Tensor<T> result({A.getDimension(0), B.getDimension(1)}, A.getFormat());
  IndexVar i, j, k;
  result(i, j) = A(i, k) * B(k, j);
  return result;
}

template <typename T>
Tensor<T> dot_with_format(Tensor<T> A, Tensor<T> B, Format format) {
  assert(A.getDimension(1) == B.getDimension(0));
  Tensor<T> result({A.getDimension(0), B.getDimension(1)}, format);
  IndexVar i, j, k;
  result(i, j) = A(i, k) * B(k, j);
  return result;
}

template <typename T>
Tensor<T> reluTensor(Tensor<T> A) {
  Func relu = reluOp();
  Tensor<T> result({A.getDimensions()}, A.getFormat());
  IndexVar i, j;
  result(i, j) = relu(A(i, j));
  return result;
}

template <typename T>
Tensor<T> gcn(Tensor<T> &A, Tensor<T> &H, Tensor<T> &W) {
  IndexVar i, j, k, l;
  assert(A.getDimension(0) == A.getDimension(1));

  auto gcn_start_time = high_resolution_clock::now();
  auto last_stop_time = gcn_start_time;

  // Create identity matrix of same shape as A
  Tensor<T> I({A.getDimensions()}, A.getFormat());
  for (int ii = 0; ii < A.getDimension(0); ++ii) {
    I.insert({ii, ii}, (T)1);
  }

  Tensor<T> A_hat({A.getDimensions()}, A.getFormat());

  A_hat(i, j) = A(i, j) + I(i, j);

  if (VERBOSE) {
    auto A_hat_compile_start_time = high_resolution_clock::now();
    A_hat.compile();
    auto A_hat_compile_stop_time = high_resolution_clock::now();
    A_hat.assemble();
    auto A_hat_assemble_stop_time = high_resolution_clock::now();
    A_hat.compute();
    auto A_hat_compute_stop_time = high_resolution_clock::now();
    auto A_hat_compile_duration = duration_cast<microseconds>(
        A_hat_compile_stop_time - A_hat_compile_start_time);
    auto A_hat_assemble_duration = duration_cast<microseconds>(
        A_hat_assemble_stop_time - A_hat_compile_stop_time);
    auto A_hat_compute_duration = duration_cast<microseconds>(
        A_hat_compute_stop_time - A_hat_assemble_stop_time);
    cout << "A_hat compile time: "
         << (double)A_hat_compile_duration.count() / 1000 << " milliseconds"
         << endl;
    cout << "A_hat assemble time: "
         << (double)A_hat_assemble_duration.count() / 1000 << " milliseconds"
         << endl;
    cout << "A_hat compute time: "
         << (double)A_hat_compute_duration.count() / 1000 << " milliseconds"
         << endl;
  }

  Tensor<T> A_hat_sumAlongRows({A_hat.getDimension(1)}, dense);
  A_hat_sumAlongRows(j) = taco::sum(i, A_hat(i, j));

  if (VERBOSE) {
    auto A_hat_sumAlongRows_compile_start_time = high_resolution_clock::now();
    A_hat_sumAlongRows.compile();
    auto A_hat_sumAlongRows_compile_stop_time = high_resolution_clock::now();
    A_hat_sumAlongRows.assemble();
    auto A_hat_sumAlongRows_assemble_stop_time = high_resolution_clock::now();
    A_hat_sumAlongRows.compute();
    auto A_hat_sumAlongRows_compute_stop_time = high_resolution_clock::now();
    auto A_hat_sumAlongRows_compile_duration =
        duration_cast<microseconds>(A_hat_sumAlongRows_compile_stop_time -
                                    A_hat_sumAlongRows_compile_start_time);
    auto A_hat_sumAlongRows_assemble_duration =
        duration_cast<microseconds>(A_hat_sumAlongRows_assemble_stop_time -
                                    A_hat_sumAlongRows_compile_stop_time);
    auto A_hat_sumAlongRows_compute_duration =
        duration_cast<microseconds>(A_hat_sumAlongRows_compute_stop_time -
                                    A_hat_sumAlongRows_assemble_stop_time);
    cout << "A_hat_sumAlongRows compile time: "
         << (double)A_hat_sumAlongRows_compile_duration.count() / 1000
         << " milliseconds" << endl;
    cout << "A_hat_sumAlongRows assemble time: "
         << (double)A_hat_sumAlongRows_assemble_duration.count() / 1000
         << " milliseconds" << endl;
    cout << "A_hat_sumAlongRows compute time: "
         << (double)A_hat_sumAlongRows_compute_duration.count() / 1000
         << " milliseconds" << endl;
  }

  Tensor<T> d({A_hat.getDimension(1)}, {Dense});
  inverseSqrtTensor(A_hat_sumAlongRows, d);

  if (VERBOSE) {
    auto d_compile_start_time = high_resolution_clock::now();
    d.compile();
    auto d_compile_stop_time = high_resolution_clock::now();
    d.assemble();
    auto d_assemble_stop_time = high_resolution_clock::now();
    d.compute();
    auto d_compute_stop_time = high_resolution_clock::now();
    auto d_compile_duration =
        duration_cast<microseconds>(d_compile_stop_time - d_compile_start_time);
    auto d_assemble_duration =
        duration_cast<microseconds>(d_assemble_stop_time - d_compile_stop_time);
    auto d_compute_duration =
        duration_cast<microseconds>(d_compute_stop_time - d_assemble_stop_time);
    cout << "d compile time: " << (double)d_compile_duration.count() / 1000
         << " milliseconds" << endl;
    cout << "d assemble time: " << (double)d_assemble_duration.count() / 1000
         << " milliseconds" << endl;
    cout << "d compute time: " << (double)d_compute_duration.count() / 1000
         << " milliseconds" << endl;
  }

  Tensor<T> result2({A.getDimensions()}, {Dense, Compressed});
  result2(i, j) = d(j) * A_hat(i, j) * d(i);

  if (VERBOSE) {
    auto result2_compile_start_time = high_resolution_clock::now();
    result2.compile();
    auto result2_compile_stop_time = high_resolution_clock::now();
    result2.assemble();
    auto result2_assemble_stop_time = high_resolution_clock::now();
    result2.compute();
    auto result2_compute_stop_time = high_resolution_clock::now();
    auto result2_compile_duration = duration_cast<microseconds>(
        result2_compile_stop_time - result2_compile_start_time);
    auto result2_assemble_duration = duration_cast<microseconds>(
        result2_assemble_stop_time - result2_compile_stop_time);
    auto result2_compute_duration = duration_cast<microseconds>(
        result2_compute_stop_time - result2_assemble_stop_time);
    cout << "result2 compile time: "
         << (double)result2_compile_duration.count() / 1000 << " milliseconds"
         << endl;
    cout << "result2 assemble time: "
         << (double)result2_assemble_duration.count() / 1000 << " milliseconds"
         << endl;
    cout << "result2 compute time: "
         << (double)result2_compute_duration.count() / 1000 << " milliseconds"
         << endl;
  }

  auto result3 = dot_with_format(H, W, {Dense, Dense});

  if (VERBOSE) {
    cout << "H dimensions: " << endl;
    print(H.getDimensions());
    cout << "W dimensions: " << endl;
    print(W.getDimensions());
  }

  if (VERBOSE) {
    auto result3_compile_start_time = high_resolution_clock::now();
    result3.compile();
    auto result3_compile_stop_time = high_resolution_clock::now();
    result3.assemble();
    auto result3_assemble_stop_time = high_resolution_clock::now();
    result3.compute();
    auto result3_compute_stop_time = high_resolution_clock::now();
    auto result3_compile_duration = duration_cast<microseconds>(
        result3_compile_stop_time - result3_compile_start_time);
    auto result3_assemble_duration = duration_cast<microseconds>(
        result3_assemble_stop_time - result3_compile_stop_time);
    auto result3_compute_duration = duration_cast<microseconds>(
        result3_compute_stop_time - result3_assemble_stop_time);
    cout << "result3 compile time: "
         << (double)result3_compile_duration.count() / 1000 << " milliseconds"
         << endl;
    cout << "result3 assemble time: "
         << (double)result3_assemble_duration.count() / 1000 << " milliseconds"
         << endl;
    cout << "result3 compute time: "
         << (double)result3_compute_duration.count() / 1000 << " milliseconds"
         << endl;
  }

  auto result4 = dot_with_format(result2, result3, {Dense, Compressed});

  if (VERBOSE) {
    auto result4_compile_start_time = high_resolution_clock::now();
    result4.compile();
    auto result4_compile_stop_time = high_resolution_clock::now();
    result4.assemble();
    auto result4_assemble_stop_time = high_resolution_clock::now();
    result4.compute();
    auto result4_compute_stop_time = high_resolution_clock::now();

    auto result4_compile_duration = duration_cast<microseconds>(
        result4_compile_stop_time - result4_compile_start_time);
    auto result4_assemble_duration = duration_cast<microseconds>(
        result4_assemble_stop_time - result4_compile_stop_time);
    auto result4_compute_duration = duration_cast<microseconds>(
        result4_compute_stop_time - result4_assemble_stop_time);
    cout << "result4 compile time: "
         << (double)result4_compile_duration.count() / 1000 << " milliseconds"
         << endl;
    cout << "result4 assemble time: "
         << (double)result4_assemble_duration.count() / 1000 << " milliseconds"
         << endl;
    cout << "result4 compute time: "
         << (double)result4_compute_duration.count() / 1000 << " milliseconds"
         << endl;
    cout << "============================================================"
         << endl;
  }

  return result4;
}

int main() {
  Format csr({Dense, Sparse});

  // Tensor<double> data_x = read(data_x_path, csr);
  Tensor<double> data_x = read(data_x_path, {Dense, Dense});

  Tensor<double> edge_index = read(edge_index_path, COO(2));

  auto start = high_resolution_clock::now();

  Tensor<double> adj_matrix = read(adj_matrix_path, COO(2));

  auto stop = high_resolution_clock::now();
  auto duration_adj_matrix = duration_cast<milliseconds>(stop - start);
  std::cout << "Adj matrix loading duration: " << duration_adj_matrix.count()
            << " ms" << std::endl;

  std::cout << "Adj matrix dimensions: " << std::endl;
  print(adj_matrix.getDimensions());

  // Load saved weights for inference
  auto gcn_conv1_weights_path = weights_dir_path + "/conv1.lin.weight.mtx";
  auto gcn_conv2_weights_path = weights_dir_path + "/conv2.lin.weight.mtx";
  Tensor<double> gcn_conv1_weights_t = read(gcn_conv1_weights_path, COO(2));
  Tensor<double> gcn_conv2_weights_t = read(gcn_conv2_weights_path, COO(2));
  // Tensor<double> gcn_conv1_weights_t = read(gcn_conv1_weights_path, csr);
  // Tensor<double> gcn_conv2_weights_t = read(gcn_conv2_weights_path, csr);
  // Tensor<double> gcn_conv1_weights(gcn_conv1_weights_t.getDimensions(), csr);
  // Tensor<double> gcn_conv2_weights(gcn_conv1_weights_t.getDimensions(), csr);
  Tensor<double> gcn_conv1_weights(gcn_conv1_weights_t.getDimensions(),
                                   {Dense, Dense});
  Tensor<double> gcn_conv2_weights(gcn_conv1_weights_t.getDimensions(),
                                   {Dense, Dense});
  gcn_conv1_weights = gcn_conv1_weights_t.transpose({1, 0});
  gcn_conv2_weights = gcn_conv2_weights_t.transpose({1, 0});
  adj_matrix.pack();
  data_x.pack();
  gcn_conv1_weights.pack();
  gcn_conv2_weights.pack();

  start = high_resolution_clock::now();

  auto H1 = gcn(adj_matrix, data_x, gcn_conv1_weights);
  auto H1_relu = reluTensor(H1);
  auto H2 = gcn(adj_matrix, H1_relu, gcn_conv2_weights);
  auto H2_relu = reluTensor(H2);
  H2_relu.evaluate();

  stop = high_resolution_clock::now();

  auto taco_duration_ms = duration_cast<milliseconds>(stop - start);
  std::cout << "TACO duration (milliseconds): " << taco_duration_ms.count()
            << std::endl;
}
