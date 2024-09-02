# - Try to find CUTLASS (oneDNN)
# Once done this will define
# CUTLASS_FOUND - System has CUTLASS
# CUTLASS_INCLUDE_DIR - The CUTLASS include directories
# CUTLASS_BUILD_INCLUDE_DIR - CUTLASS include directories in build
# CUTLASS_LIBRARY - The libraries needed to use CUTLASS
# CUTLASS_DEFINITIONS - Compiler switches required for using CUTLASS
set(CUTLASS_ROOT "${CMAKE_SOURCE_DIR}/csrc/cutlass")
set(CUTLASS_BUILD "${CMAKE_SOURCE_DIR}/csrc/cutlass/build")
message("CUTLASS_ROOT:${CUTLASS_ROOT}/include")
find_path ( CUTLASS_INCLUDE_DIR cutlass/cutlass.h HINTS ${CUTLASS_ROOT}/include )
find_path ( CUTLASS_BUILD_INCLUDE_DIR cutlass/version.h HINTS ${CUTLASS_BUILD}/include )
find_library ( CUTLASS_LIBRARY NAMES cutlass HINTS ${CUTLASS_BUILD}/tools/library )
message("CUTLASS_INCLUDE_DIR:${CUTLASS_INCLUDE_DIR}")
message("CUTLASS_BUILD_INCLUDE_DIR:${CUTLASS_BUILD_INCLUDE_DIR}")
message("CUTLASS_LIBRARY:${CUTLASS_LIBRARY}")

include ( FindPackageHandleStandardArgs )
find_package_handle_standard_args ( CUTLASS DEFAULT_MSG CUTLASS_LIBRARY CUTLASS_INCLUDE_DIR CUTLASS_BUILD_INCLUDE_DIR )
