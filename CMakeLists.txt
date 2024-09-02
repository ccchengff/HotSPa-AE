cmake_minimum_required(VERSION 3.24)
add_compile_options(-w)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(DCMAKE_BUILD_TYPE Debug)

project(Hetu LANGUAGES CXX CUDA)

include(cmake/config_refactor.cmake)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/Modules)

set(USE_FLASH_ATTN ON)
if (${USE_FLASH_ATTN})
  set(CMAKE_CXX_STANDARD 17)
else()
  set(CMAKE_CXX_STANDARD 14)
endif()
message("CPP_STANTARD:${CMAKE_CXX_STANDARD}")
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_CXX_FLAGS "-O3 -Wall -pthread")
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)

# TODO: add configurable options
set(HETU_COMPILE_OMP ON)
set(HETU_COMPILE_CUDA ON)
set(HETU_COMPILE_CUDNN ON)
set(HETU_COMPILE_NCCL ON)
set(HETU_COMPILE_NVML ON)
set(HETU_COMPILE_DNNL ON)

if(${HETU_COMPILE_OMP})
  find_package(MPI 3.1 REQUIRED)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

if(${HETU_COMPILE_CUDA})
  # set(CMAKE_CUDA_COMPILER ${CUDAToolkit_ROOT}/bin/nvcc)
  find_package(CUDAToolkit REQUIRED)
  enable_language(CUDA)
  if(${CMAKE_CUDA_COMPILER_VERSION} VERSION_LESS "10.1")
    message(FATAL_ERROR "Required CUDA version >= 10.1, while current CUDA version is ${CMAKE_CUDA_COMPILER_VERSION}")
  endif()
  include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

  if (${CMAKE_CUDA_COMPILER_VERSION} VERSION_GREATER_EQUAL "11")
    set(HETU_CUDA_ARCH 80 86)
  else()
    set(HETU_CUDA_ARCH 50 52 60 61 70 75)
  endif()

  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
endif()

add_subdirectory(${CMAKE_SOURCE_DIR}/hetu)
add_subdirectory(${CMAKE_SOURCE_DIR}/python_refactor)