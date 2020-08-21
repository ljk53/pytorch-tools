#include <iostream>
#include <torch/script.h>
#include "common.h"

constexpr int BATCH_SIZE = 32;
constexpr int SEQUENCE_LENGTH = 512;

void index_over_matrix_torch(int count) {
  auto matrix = torch::full(
      {BATCH_SIZE, SEQUENCE_LENGTH},
      0,
      torch::TensorOptions().dtype(torch::kInt32)
  );

  benchmark("index_over_matrix_torch", count, BATCH_SIZE * SEQUENCE_LENGTH, [&]() {
    for (int row_index = 0; row_index < BATCH_SIZE; ++row_index) {
      for (int column_index = 0; column_index < SEQUENCE_LENGTH; ++column_index) {
        // matrix.index({row_index}).index({column_index});
        at::indexing::get_item(at::indexing::get_item(matrix, {row_index}), {column_index});
      }
    }
  });
}

void index_over_matrix_torch_nograd(int count) {
  torch::NoGradGuard nograd_guard;

  auto matrix = torch::full(
      {BATCH_SIZE, SEQUENCE_LENGTH},
      0,
      torch::TensorOptions().dtype(torch::kInt32)
  );

  benchmark("index_over_matrix_torch_nograd", count, BATCH_SIZE * SEQUENCE_LENGTH, [&]() {
    for (int row_index = 0; row_index < BATCH_SIZE; ++row_index) {
      for (int column_index = 0; column_index < SEQUENCE_LENGTH; ++column_index) {
        // matrix.index({row_index}).index({column_index});
        at::indexing::get_item(at::indexing::get_item(matrix, {row_index}), {column_index});
      }
    }
  });
}

void index_over_matrix_torch_nograd_novar(int count) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;

  auto matrix = torch::full(
      {BATCH_SIZE, SEQUENCE_LENGTH},
      0,
      torch::TensorOptions().dtype(torch::kInt32)
  );

  benchmark("index_over_matrix_torch_nograd_novar", count, BATCH_SIZE * SEQUENCE_LENGTH, [&]() {
    for (int row_index = 0; row_index < BATCH_SIZE; ++row_index) {
      for (int column_index = 0; column_index < SEQUENCE_LENGTH; ++column_index) {
        // matrix.index({row_index}).index({column_index});
        at::indexing::get_item(at::indexing::get_item(matrix, {row_index}), {column_index});
      }
    }
  });
}

void index_over_matrix_torch_nograd_novar_select(int count) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;

  auto matrix = torch::full(
      {BATCH_SIZE, SEQUENCE_LENGTH},
      0,
      torch::TensorOptions().dtype(torch::kInt32)
  );

  benchmark("index_over_matrix_torch_nograd_novar_select", count, BATCH_SIZE * SEQUENCE_LENGTH, [&]() {
    for (int row_index = 0; row_index < BATCH_SIZE; ++row_index) {
      for (int column_index = 0; column_index < SEQUENCE_LENGTH; ++column_index) {
        matrix.select(0, row_index).select(0, column_index);
      }
    }
  });
}

void index_over_matrix_torch_nograd_novar_select_inline(int count) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;

  auto matrix = torch::full(
      {BATCH_SIZE, SEQUENCE_LENGTH},
      0,
      torch::TensorOptions().dtype(torch::kInt32)
  );

  auto select = [&](const torch::Tensor& self, int64_t dim, int64_t index) -> torch::Tensor {
    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();
    auto storage_offset = self.storage_offset() + index * strides[dim];
    sizes.erase(sizes.begin() + dim);
    strides.erase(strides.begin() + dim);
    return self.as_strided(sizes, strides, storage_offset);
  };

  benchmark("index_over_matrix_torch_nograd_novar_select_inline", count, BATCH_SIZE * SEQUENCE_LENGTH, [&]() {
    for (int row_index = 0; row_index < BATCH_SIZE; ++row_index) {
      for (int column_index = 0; column_index < SEQUENCE_LENGTH; ++column_index) {
        select(select(matrix, 0, row_index), 0, column_index);
      }
    }
  });
}

int main() {
  report_header();
  index_over_matrix_torch(200);
  index_over_matrix_torch_nograd(200);
  index_over_matrix_torch_nograd_novar(200);
  index_over_matrix_torch_nograd_novar_select(200);
  index_over_matrix_torch_nograd_novar_select_inline(200);
  return 0;
}
