cmake_minimum_required(VERSION 3.18)

# Ensure this is set before project() enables CUDA language.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES native CACHE STRING "CUDA architectures" FORCE)
endif()

project(cvdv LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)

# Prefer NVIDIA HPC SDK math_libs matching the active CUDA toolkit version.
string(REGEX MATCH "^[0-9]+\\.[0-9]+" CUDATOOLKIT_VERSION_MM "${CUDAToolkit_VERSION}")
file(GLOB NVHPC_MATCHING_MATH_LIB_DIRS "/opt/nvidia/hpc_sdk/Linux_x86_64/*/math_libs/${CUDATOOLKIT_VERSION_MM}")

# Find cuTENSOR across common toolkit and distro install locations.
find_path(CUTENSOR_INCLUDE_DIR
    NAMES cutensor.h
    HINTS
        ${CUDAToolkit_INCLUDE_DIRS}
        ${CUDAToolkit_ROOT}
        ${NVHPC_MATCHING_MATH_LIB_DIRS}
        ENV CUDA_PATH
        ENV CUDA_HOME
    PATH_SUFFIXES
        include
        include/libcutensor
        libcutensor/13
        libcutensor/12
    PATHS
        /usr/include
        /usr/local/cuda/include
)

find_library(CUTENSOR_LIBRARY
    NAMES cutensor
    HINTS
        ${CUDAToolkit_LIBRARY_DIR}
        ${CUDAToolkit_LIBRARY_ROOT}
        ${CUDAToolkit_ROOT}
        ${NVHPC_MATCHING_MATH_LIB_DIRS}
        ENV CUDA_PATH
        ENV CUDA_HOME
    PATH_SUFFIXES
        lib64
        lib
        lib64/stubs
        libcutensor/13
        libcutensor/12
    PATHS
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64
)
if(NOT CUTENSOR_INCLUDE_DIR OR NOT CUTENSOR_LIBRARY)
    message(FATAL_ERROR "cuTENSOR not found. Install a matching cuTENSOR dev package or set CUTENSOR_INCLUDE_DIR/CUTENSOR_LIBRARY.")
endif()

add_library(cvdv SHARED
    src/cuda/context.cu
    src/cuda/gates.cu
    src/cuda/bosonic.cu
    src/cuda/measure.cu
)

target_include_directories(cvdv PRIVATE
    src/cuda/
    ${CUTENSOR_INCLUDE_DIR}
)
target_link_libraries(cvdv PRIVATE
    CUDA::cudart
    CUDA::cufft
    ${CUTENSOR_LIBRARY}
)

set_target_properties(cvdv PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
