#include <torch/script.h>
#include <valgrind/callgrind.h>

int main() {
  auto a = torch::ones({128}, at::TensorOptions().requires_grad(false));
  auto b = torch::ones({128}, at::TensorOptions().requires_grad(false));
  auto c = a;
  CALLGRIND_START_INSTRUMENTATION;
  for (int i = 0; i < 100000; ++i) {
    c = at::add(c, b);
  }
  CALLGRIND_STOP_INSTRUMENTATION;
  return 0;
}
