#include <chrono>
#include <iostream>

#include <torch/script.h>

using namespace std::chrono;

void report(const char* name, high_resolution_clock::time_point start, int count) {
  duration<double> time_span =
      duration_cast<duration<double>>(high_resolution_clock::now() - start);
  std::cout << std::setw(30) << std::fixed << std::setprecision(2)
            << name
            << "\tthroughput: " << std::setw(10) << count / time_span.count() << " samples/sec"
            << "\tduration: " << std::setw(15) << time_span.count() / count * 1e9 << " ns"
            << std::endl;
}

void add_s1_grad(int count) {
  auto a = torch::ones({1}, at::TensorOptions().requires_grad(true));
  auto b = torch::ones({1}, at::TensorOptions().requires_grad(true));

  high_resolution_clock::time_point start = high_resolution_clock::now();
  auto c = a;
  for (int i = 0; i < count; ++i) {
    c = at::add(c, b);
  }
  report("add_s1_grad", start, count);
}

void add_s1_nograd(int count) {
  torch::NoGradGuard guard;
  auto a = torch::ones({1});
  auto b = torch::ones({1});

  high_resolution_clock::time_point start = high_resolution_clock::now();
  auto c = a;
  for (int i = 0; i < count; ++i) {
    c = at::add(c, b);
  }
  report("add_s1_nograd", start, count);
}

void add_s1_nograd_novar(int count) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;
  auto a = torch::ones({1});
  auto b = torch::ones({1});

  high_resolution_clock::time_point start = high_resolution_clock::now();
  auto c = a;
  for (int i = 0; i < count; ++i) {
    c = at::add(c, b);
  }
  report("add_s1_nograd_novar", start, count);
}

void mm_s1000_grad(int count) {
  auto a = torch::ones({1000, 1000}, at::TensorOptions().requires_grad(true));
  auto b = torch::ones({1000, 1000}, at::TensorOptions().requires_grad(true));

  high_resolution_clock::time_point start = high_resolution_clock::now();
  auto c = a;
  for (int i = 0; i < count; ++i) {
    c = at::mm(c, b);
  }
  report("mm_s1000_grad", start, count);
}

void mm_s1000_nograd_novar(int count) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;
  auto a = torch::ones({1000, 1000}, at::TensorOptions().requires_grad(true));
  auto b = torch::ones({1000, 1000}, at::TensorOptions().requires_grad(true));

  high_resolution_clock::time_point start = high_resolution_clock::now();
  auto c = a;
  for (int i = 0; i < count; ++i) {
    c = at::mm(c, b);
  }
  report("mm_s1000_nograd_novar", start, count);
}

int main() {
  add_s1_grad(1000000);
  add_s1_nograd(1000000);
  add_s1_nograd_novar(1000000);
  mm_s1000_grad(100);
  mm_s1000_nograd_novar(100);
  return 0;
}
