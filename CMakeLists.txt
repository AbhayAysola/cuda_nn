cmake_minimum_required(VERSION 3.18)
project(CUDA_NEURAL_NETWORK LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# specify CUDA architectures (change 'native' to a specific SM version if needed, e.g. '80')
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(cuda_nn
  src/main.cpp
  src/utils/idx_parser.cpp
  src/utils/model_parser.cpp
  src/utils/model.cu
)

target_include_directories(cuda_nn PUBLIC "./src/utils/include/")

# Enable fast math for performance and optimization level O3
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-O3>")


