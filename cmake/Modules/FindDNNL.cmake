# - Try to find DNNL (oneDNN)
# Once done this will define
# DNNL_FOUND - System has DNNL
# DNNL_INCLUDE_DIR - The DNNL include directories
# DNNL_BUILD_INCLUDE_DIR - DNNL include directories in build
# DNNL_LIBRARY - The libraries needed to use DNNL
# DNNL_DEFINITIONS - Compiler switches required for using DNNL

find_path ( DNNL_INCLUDE_DIR dnnl.h HINTS ${DNNL_ROOT}/include )
find_path ( DNNL_BUILD_INCLUDE_DIR dnnl_config.h HINTS ${DNNL_BUILD}/include NO_DEFAULT_PATH )
find_library ( DNNL_LIBRARY NAMES dnnl HINTS ${DNNL_BUILD}/lib NO_DEFAULT_PATH )

include ( FindPackageHandleStandardArgs )
find_package_handle_standard_args ( DNNL DEFAULT_MSG DNNL_LIBRARY DNNL_INCLUDE_DIR DNNL_BUILD_INCLUDE_DIR )
