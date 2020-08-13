#include <chrono>
#include <iostream>

#include <torch/script.h>

using namespace std::chrono;

void report_header() {
  std::cout << std::setw(35) << "name"
            << std::setw(25) << "samples/sec"
            << std::setw(25) << "ns"
            << std::endl;
}

void report(const std::string& name, high_resolution_clock::time_point start, int count) {
  duration<double> time_span =
      duration_cast<duration<double>>(high_resolution_clock::now() - start);
  std::cout << std::setw(35) << std::fixed << std::setprecision(2)
            << name
            << std::setw(25) << count / time_span.count()
            << std::setw(25) << time_span.count() / count * 1e9
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

void add_s1_nograd_outplace(int count) {
  torch::NoGradGuard nograd_guard;
  auto a = torch::ones({1});
  auto b = torch::ones({1});
  auto c = torch::empty({1});

  high_resolution_clock::time_point start = high_resolution_clock::now();
  for (int i = 0; i < count; ++i) {
    at::add_out(c, a, b);
    std::swap(a, c);
  }
  report("add_s1_nograd_outplace", start, count);
}

void add_s1_nograd_outplace_novar(int count) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;
  auto a = torch::ones({1});
  auto b = torch::ones({1});
  auto c = torch::empty({1});

  high_resolution_clock::time_point start = high_resolution_clock::now();
  for (int i = 0; i < count; ++i) {
    at::add_out(c, a, b);
    std::swap(a, c);
  }
  report("add_s1_nograd_outplace_novar", start, count);
}

void mm_sN_grad(int count, int N) {
  auto a = torch::ones({N, N}, at::TensorOptions().requires_grad(true));
  auto b = torch::ones({N, N}, at::TensorOptions().requires_grad(true));

  high_resolution_clock::time_point start = high_resolution_clock::now();
  for (int i = 0; i < count; ++i) {
    auto c = at::mm(a, b);
  }
  report("mm_s" + std::to_string(N) + "_grad", start, count);
}

void mm_sN_nograd_novar(int count, int N) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;
  auto a = torch::ones({N, N}, at::TensorOptions().requires_grad(false));
  auto b = torch::ones({N, N}, at::TensorOptions().requires_grad(false));

  high_resolution_clock::time_point start = high_resolution_clock::now();
  for (int i = 0; i < count; ++i) {
    auto c = at::mm(a, b);
  }
  report("mm_s" + std::to_string(N) + "_nograd_novar", start, count);
}

void mm_sN_nograd_novar_outplace(int count, int N) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;
  auto a = torch::ones({N, N}, at::TensorOptions().requires_grad(false));
  auto b = torch::ones({N, N}, at::TensorOptions().requires_grad(false));
  auto c = torch::empty({N, N}, at::TensorOptions().requires_grad(false));

  high_resolution_clock::time_point start = high_resolution_clock::now();
  for (int i = 0; i < count; ++i) {
    at::mm_out(c, a, b);
  }
  report("mm_s" + std::to_string(N) + "_nograd_novar_outplace", start, count);
}

int main() {
  report_header();

  add_s1_nograd_outplace_novar(1000000);
  add_s1_nograd_outplace(1000000);
  add_s1_nograd(1000000);
  add_s1_grad(1000000);

  for (int N : {64, 256}) {
    std::cout << std::endl;
    int count = (1024 / N) * (1024 / N) * (1024 / N) * 10;
    mm_sN_nograd_novar_outplace(count, N);
    mm_sN_nograd_novar(count, N);
    mm_sN_grad(count, N);
  }

  return 0;
}
