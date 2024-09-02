# - Try to find FLASH 
# Once done this will define
# FLASH_FOUND - System has FLASH
# FLASH_INCLUDE_DIR - The FLASH include directories
# FLASH_BUILD_INCLUDE_DIR - FLASH include directories in build
# FLASH_LIBRARY - The libraries needed to use FLASH
# FLASH_DEFINITIONS - Compiler switches required for using FLASH

find_path (FLASH_INCLUDE_DIR flash.h HINTS ${FLASH_ROOT}/include)
find_library (FLASH_LIBRARY NAMES FLASH HINTS ${FLASH_ROOT}/lib NO_DEFAULT_PATH)

include ( FindPackageHandleStandardArgs )
find_package_handle_standard_args (FLASH DEFAULT_MSG FLASH_LIBRARY FLASH_INCLUDE_DIR)