######################
### Set paths ######
######################

# CUDA version >= 10.1
set(CUDAToolkit_ROOT /usr/local/cuda)

# CUDNN >= 7.5
# - CUDNN_ROOT: root directory of cudnn
set(CUDNN_ROOT)

# NCCL version >= 2.8
set(NCCL_ROOT)

# MPI >= 3.1
set(MPI_ROOT)

# PyBind11 2.6.2
# - pybind11_DIR: cmake directory of pybind11, 
#                 can be obtained by `python3 -m pybind11 --cmakedir` 
#                 if pybind11 has been installed via pip
# if not found, we'll download and compile it in time
set(pybind11_DIR)

# ZMQ 4.3.2
# - ZMQ_ROOT: root directory of zeromq
# - ZMQ_BUILD: build directory of zeromq
# if not found, we'll download and compile it in time
set(ZMQ_ROOT)
set(ZMQ_BUILD)

# DNNL (oneDNN) 3.0
# - DNNL_ROOT: root directory of zeromq
# - DNNL_BUILD: build directory of zeromq
# if not found, we'll download and compile it in time
set(DNNL_ROOT)
set(DNNL_BUILD)

# if you have a protoc in conda bin, ignore `/path/to/anaconda3/bin`
set(CMAKE_IGNORE_PATH)

set(FLASH_ROOT)
