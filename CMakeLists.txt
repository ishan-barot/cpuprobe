cmake_minimum_required(VERSION 3.20)
project(cpuprobe CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # clang-tidy reads this
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
# c++ only warning flags, nvcc chokes on them when it compiles .cu files
add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-Wall> $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                    $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>)

option(CPUPROBE_SANITIZE "Enable ASan/UBSan" OFF)
option(CPUPROBE_ENABLE_CUDA "Build the NVIDIA GPU backend (needs the CUDA toolkit)" OFF)
if(CPUPROBE_SANITIZE)
  add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fsanitize=address,undefined>
                      $<$<COMPILE_LANGUAGE:CXX>:-fno-omit-frame-pointer>)
  add_link_options(-fsanitize=address,undefined)
endif()

# stamp results files with the commit they came from
execute_process(COMMAND git rev-parse --short HEAD
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE CPUPROBE_GIT_COMMIT OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
if(NOT CPUPROBE_GIT_COMMIT)
  set(CPUPROBE_GIT_COMMIT unknown)
endif()

find_package(Threads REQUIRED)

add_library(cpuprobe_lib
  src/latency.cpp src/cpu_info.cpp src/util.cpp src/perf_counters.cpp src/kernels.cpp
  src/stats.cpp src/regression.cpp src/trace_json.cpp)
target_include_directories(cpuprobe_lib PUBLIC include)
target_link_libraries(cpuprobe_lib PUBLIC Threads::Threads)

add_executable(cpuprobe src/main.cpp)
target_link_libraries(cpuprobe PRIVATE cpuprobe_lib)

if(CPUPROBE_ENABLE_CUDA)
  # 89 = ada (rtx 40 series). pass a different cuda architectures value for other gpus, 75 for a t4
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 20)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  add_library(cpuprobe_gpu
    src/gpu/device.cu src/gpu/transpose.cu src/gpu/sgemm.cu src/gpu/latency.cu
    src/gpu/cupti_tracer.cpp src/gpu/gpu_cli.cpp)
  target_link_libraries(cpuprobe_gpu PUBLIC cpuprobe_lib
    PRIVATE CUDA::cudart CUDA::cublas CUDA::cupti)
  target_compile_definitions(cpuprobe_gpu PRIVATE CPUPROBE_GIT_COMMIT="${CPUPROBE_GIT_COMMIT}")
  target_compile_options(cpuprobe_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)

  target_compile_definitions(cpuprobe PRIVATE CPUPROBE_HAS_CUDA=1)
  target_link_libraries(cpuprobe PRIVATE cpuprobe_gpu)
endif()

# tests
include(FetchContent)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_executable(cpuprobe_tests
  tests/test_latency.cpp tests/test_util.cpp tests/test_perf_counters.cpp tests/test_kernels.cpp
  tests/test_regression.cpp tests/test_trace_json.cpp)
target_link_libraries(cpuprobe_tests PRIVATE cpuprobe_lib GTest::gtest_main)
include(GoogleTest)
gtest_discover_tests(cpuprobe_tests)

if(CPUPROBE_ENABLE_CUDA)
  # separate binary so compute-sanitizer can run just the gpu tests
  add_executable(cpuprobe_gpu_tests tests/gpu/test_gpu.cpp)
  target_link_libraries(cpuprobe_gpu_tests PRIVATE cpuprobe_gpu GTest::gtest_main)
  gtest_discover_tests(cpuprobe_gpu_tests DISCOVERY_MODE PRE_TEST)
endif()
