# Find NVML library

find_path(NVML_INCLUDE_DIR nvml.h
    HINTS
        $ENV{NVML_INCLUDE_DIR}
        /usr/local/cuda/include
        /usr/include
)

find_library(NVML_LIBRARY
    NAMES nvidia-ml libnvidia-ml libnvidia-ml.so.1
    HINTS
        $ENV{NVML_LIBRARY_DIR}
        /usr/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/nvidia/lib64
)

if (NVML_INCLUDE_DIR AND NVML_LIBRARY)
    set(NVML_FOUND TRUE)
endif()

if (NVML_FOUND)
    if (NOT NVML_FIND_QUIETLY)
        message(STATUS "Found NVML INCLUDE DIR: ${NVML_INCLUDE_DIR}")
        message(STATUS "Found NVML LIBRARY: ${NVML_LIBRARY}")
    endif()
else()
    if (NVML_FIND_REQUIRED)
        message(FATAL_ERROR "NVML library not found")
    endif()
endif()

mark_as_advanced(
    NVML_INCLUDE_DIR
    NVML_LIBRARY
)

# Set the output variables
set(NVML_INCLUDE_DIRS ${NVML_INCLUDE_DIR} CACHE PATH "Path to NVML includes")
set(NVML_LIBRARIES ${NVML_LIBRARY} CACHE FILEPATH "Path to NVML libraries")
