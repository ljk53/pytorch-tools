cmake_minimum_required(VERSION 3.1)

project(pytorch_tests)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)
find_package(Threads REQUIRED)

add_executable(callgrind callgrind.cpp)
target_include_directories(callgrind PUBLIC ${TORCH_INCLUDE_DIRS})

target_link_libraries(callgrind
  -Wl,--gc-sections
  -Wl,--whole-archive
  ${TORCH_LIBRARIES}
  -Wl,--no-whole-archive
  Threads::Threads
)
