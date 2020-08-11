#include <chrono>
#include <iostream>

#include <torch/script.h>

using namespace std::chrono;

int main() {
  auto a = torch::ones({1000, 1000}, at::TensorOptions().requires_grad(true));
  auto b = torch::ones({1000, 1000}, at::TensorOptions().requires_grad(true));

  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  auto c = at::mm(a, b);
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
  std::cout << time_span.count() << std::endl;

  auto d = c.sum();
  std::cout << "d = " << d << std::endl;

  return 0;
}
