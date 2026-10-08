cmake_minimum_required(VERSION 3.20)
project(cpuprobe CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
add_compile_options(-Wall -Wextra -Wpedantic)

option(CPUPROBE_SANITIZE "Enable ASan/UBSan" OFF)
if(CPUPROBE_SANITIZE)
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

add_library(cpuprobe_lib src/latency.cpp src/cpu_info.cpp src/util.cpp)
target_include_directories(cpuprobe_lib PUBLIC include)

add_executable(cpuprobe src/main.cpp)
target_link_libraries(cpuprobe PRIVATE cpuprobe_lib)

# tests
include(FetchContent)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(cpuprobe_tests tests/test_latency.cpp tests/test_util.cpp)
target_link_libraries(cpuprobe_tests PRIVATE cpuprobe_lib GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(cpuprobe_tests)
