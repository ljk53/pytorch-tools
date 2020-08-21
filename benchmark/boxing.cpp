#include <iostream>
#include <torch/script.h>
#include "common.h"

void add_s1_nograd_outplace_novar(int count) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;
  auto a = torch::ones({1});
  auto b = torch::ones({1});
  auto c = torch::empty({1});

  benchmark("add_s1_nograd_outplace_novar", count, [&]() {
    at::add_out(c, a, b);
  });
}

template<class... Args>
static torch::jit::Stack boxArgs(Args... args) {
  torch::jit::Stack stack;
  stack.reserve(sizeof...(Args));
  torch::jit::push(stack, std::forward<Args>(args)...);
  return stack;
}

void add_s1_nograd_outplace_novar_boxing_1x1x100(int count) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;
  auto a = torch::ones({1});
  auto b = torch::ones({1});
  auto c = torch::empty({1});

  benchmark("add_s1_nograd_outplace_novar_boxing_1x1x100", count, [&]() {
    for (int i = 0; i < 100; i++) {
      boxArgs(a);
    }
    at::add_out(c, a, b);
  });
}

void add_s1_nograd_outplace_novar_boxing_2x1x10(int count) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;
  auto a = torch::ones({1});
  auto b = torch::ones({1});
  auto c = torch::empty({1});

  benchmark("add_s1_nograd_outplace_novar_boxing_2x1x10", count, [&]() {
    for (int i = 0; i < 10; i++) {
      boxArgs(a, b);
    }
    at::add_out(c, a, b);
  });
}

void add_s1_nograd_outplace_novar_boxing_4x1(int count) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;
  auto a = torch::ones({1});
  auto b = torch::ones({1});
  auto c = torch::empty({1});

  benchmark("add_s1_nograd_outplace_novar_boxing_4x1", count, [&]() {
    boxArgs(a, b, c, 1.0);
    at::add_out(c, a, b);
  });
}

void add_s1_nograd_outplace_novar_boxing_4x2(int count) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;
  auto a = torch::ones({1});
  auto b = torch::ones({1});
  auto c = torch::empty({1});

  benchmark("add_s1_nograd_outplace_novar_boxing_4x2", count, [&]() {
    boxArgs(a, b, c, 1.0,
            a, b, c, 1.0);
    at::add_out(c, a, b);
  });
}

void add_s1_nograd_outplace_novar_boxing_4x4(int count) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;
  auto a = torch::ones({1});
  auto b = torch::ones({1});
  auto c = torch::empty({1});

  benchmark("add_s1_nograd_outplace_novar_boxing_4x4", count, [&]() {
    boxArgs(a, b, c, 1.0,
            a, b, c, 1.0,
            a, b, c, 1.0,
            a, b, c, 1.0);
    at::add_out(c, a, b);
  });
}

void add_s1_nograd_outplace_novar_boxing_4x8(int count) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;
  auto a = torch::ones({1});
  auto b = torch::ones({1});
  auto c = torch::empty({1});

  benchmark("add_s1_nograd_outplace_novar_boxing_4x8", count, [&]() {
    boxArgs(a, b, c, 1.0,
            a, b, c, 1.0,
            a, b, c, 1.0,
            a, b, c, 1.0,
            a, b, c, 1.0,
            a, b, c, 1.0,
            a, b, c, 1.0,
            a, b, c, 1.0);
    at::add_out(c, a, b);
  });
}

#define DEFINE_ARGS(_) \
  auto a##_ = torch::ones({1}); \
  auto b##_ = torch::ones({1}); \
  auto c##_ = torch::empty({1});

#define REF_ARGS(_) \
  a##_, b##_, c##_, 1.0

void add_s1_nograd_outplace_novar_boxing_8(int count) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;
  DEFINE_ARGS()
  DEFINE_ARGS(2)

  benchmark("add_s1_nograd_outplace_novar_boxing_8", count, [&]() {
    boxArgs(
        REF_ARGS(),
        REF_ARGS(2));
    at::add_out(c, a, b);
  });
}

void add_s1_nograd_outplace_novar_boxing_16(int count) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;
  DEFINE_ARGS()
  DEFINE_ARGS(2)
  DEFINE_ARGS(3)
  DEFINE_ARGS(4)

  benchmark("add_s1_nograd_outplace_novar_boxing_16", count, [&]() {
    boxArgs(
        REF_ARGS(),
        REF_ARGS(2),
        REF_ARGS(3),
        REF_ARGS(4));
    at::add_out(c, a, b);
  });
}

void add_s1_nograd_outplace_novar_boxing_32(int count) {
  torch::NoGradGuard nograd_guard;
  torch::AutoNonVariableTypeMode non_variable_type_guard;
  DEFINE_ARGS()
  DEFINE_ARGS(2)
  DEFINE_ARGS(3)
  DEFINE_ARGS(4)
  DEFINE_ARGS(5)
  DEFINE_ARGS(6)
  DEFINE_ARGS(7)
  DEFINE_ARGS(8)

  benchmark("add_s1_nograd_outplace_novar_boxing_32", count, [&]() {
    boxArgs(
        REF_ARGS(),
        REF_ARGS(2),
        REF_ARGS(3),
        REF_ARGS(4),
        REF_ARGS(5),
        REF_ARGS(6),
        REF_ARGS(7),
        REF_ARGS(8));
    at::add_out(c, a, b);
  });
}

int main() {
  add_s1_nograd_outplace_novar(10000000);
  add_s1_nograd_outplace_novar_boxing_1x1x100(1000000);
  add_s1_nograd_outplace_novar_boxing_2x1x10(10000000);
  add_s1_nograd_outplace_novar_boxing_4x1(10000000);
  add_s1_nograd_outplace_novar_boxing_4x2(10000000);
  add_s1_nograd_outplace_novar_boxing_4x4(10000000);
  add_s1_nograd_outplace_novar_boxing_4x8(10000000);
  add_s1_nograd_outplace_novar_boxing_8(10000000);
  add_s1_nograd_outplace_novar_boxing_16(10000000);
  add_s1_nograd_outplace_novar_boxing_32(10000000);

  return 0;
}
