include(ExternalProject)

set(NCCL_TAR ${CMAKE_SOURCE_DIR}/third_party/nccl/nccl_2.20.3-1+cuda11.0_x86_64.txz)
set(NCCL_TAR_FILE ${CMAKE_CURRENT_BINARY_DIR}/third_party/nccl_2.20.3-1+cuda11.0_x86_64)
set(NCCL_FILE ${CMAKE_CURRENT_BINARY_DIR}/third_party/nccl)

file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_party)
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E tar xzf ${NCCL_TAR} 
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_party
  )
  execute_process(
    COMMAND mv ${NCCL_TAR_FILE} ${NCCL_FILE}
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/third_party
  )

set(NCCL_ROOT "${CMAKE_CURRENT_BINARY_DIR}/third_party/nccl")
set(NCCL_INCLUDE_DIR ${NCCL_ROOT}/include)
set(NCCL_LIB_DIR ${NCCL_ROOT}/lib)

