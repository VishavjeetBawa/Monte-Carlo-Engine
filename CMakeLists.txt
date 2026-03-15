cmake_minimum_required(VERSION 3.22)

project(urop LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(COMMON_PERF_FLAGS "-g -fno-omit-frame-pointer -march=native")

set(CMAKE_CXX_FLAGS_RELEASE "-O3 ${COMMON_PERF_FLAGS}")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 ${COMMON_PERF_FLAGS}")

add_compile_options(-ffast-math -ftree-vectorize)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_ARCHITECTURES native)

# ======================
# Core library
# ======================

add_library(core STATIC
    core/src/Payoff.cpp
    core/src/RunStats.cpp
    core/src/Timer.cpp
)

target_include_directories(core PUBLIC core/include)

# ======================
# CPU Monte Carlo engine
# ======================

add_library(cpu_mc STATIC
    cpu/src/GAsianPricer.cpp
    cpu/src/MCE.cpp
    cpu/src/RNG.cpp
)

target_include_directories(cpu_mc PUBLIC cpu/include)

target_link_libraries(cpu_mc PUBLIC core)

# ======================
# CUDA Monte Carlo engine
# ======================

add_library(cuda_mc STATIC
    cuda/src/CudaQOMCE.cu
    cuda/src/kernels.cu
    cuda/src/sobol_device.cu
    cuda/src/invnorm_device.cu
    cuda/src/brownian_bridge.cu
)

set_target_properties(cuda_mc PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

target_include_directories(cuda_mc PUBLIC cuda/include)

target_link_libraries(cuda_mc PUBLIC core cpu_mc)

# ======================
# QT GUI
# ======================

add_subdirectory(qt)

# ======================
# Tests
# ======================

add_executable(test_OMCE tests/test_OMCE.cpp)
add_executable(test_AVMCE tests/test_AVMCE.cpp)
add_executable(test_CMCE tests/test_CMCE.cpp)
add_executable(test_QMCE tests/test_QMCE.cpp)
add_executable(test_coqmce tests/test_coqmce.cpp)
add_executable(test_gpuqmce tests/test_gpuqmce.cpp)

target_link_libraries(test_OMCE PRIVATE cpu_mc)
target_link_libraries(test_AVMCE PRIVATE cpu_mc)
target_link_libraries(test_CMCE PRIVATE cpu_mc)
target_link_libraries(test_QMCE PRIVATE cpu_mc)
target_link_libraries(test_coqmce PRIVATE cpu_mc)
target_link_libraries(test_gpuqmce PRIVATE cuda_mc)
