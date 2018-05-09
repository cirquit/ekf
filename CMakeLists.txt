cmake_minimum_required (VERSION 3.5.1)

set(PROJECT_NAME KaFi)
set(LIB_NAME  kafi)
set(EXEC_NAME kafi_exec)
set(TEST_NAME kafi_test)

project (${PROJECT_NAME})

# define library version 
# Note that the library version is set via FORCE. This prevents users changing the value in the CMakeCache.txt
set(KAFI_VERSION_MAJOR 1 CACHE STRING "major version" FORCE)
set(KAFI_VERSION_MINOR 0 CACHE STRING "minor version" FORCE)
set(KAFI_VERSION ${KAFI_VERSION_MAJOR}.${KAFI_VERSION_MINOR} CACHE STRING "version" FORCE)

#set(bin_dest "bin")
set(include_dest "include/kafi-${KAFI_VERSION}")
set(main_lib_dest "lib/kafi-${KAFI_VERSION}")

install(FILES kafi-config.cmake DESTINATION ${main_lib_dest})
# Sets the correct version number which find_package() will look for
configure_file(kafi-config-version.cmake ${CMAKE_CURRENT_BINARY_DIR}/kafi-config-version.cmake @ONLY)
install(FILES kafi-config.cmake ${CMAKE_CURRENT_BINARY_DIR}/kafi-config-version.cmake DESTINATION ${main_lib_dest})
install(EXPORT ${LIB_NAME} DESTINATION "${main_lib_dest}")

## this flag is used to set the compiler flags and should be used in the code
## with #if DEBUG_MODE <code> #else <code> #endif OR the debug print macro
OPTION(DEBUG_MODE "Enables the DEBUG messages and -O0 flags" 1)

# set compiler flags
set(CMAKE_CXX_COMPILER=g++)

## Compile as C++14
add_compile_options(-std=c++14)

# Compile differently if DEBUG_MODE is active
if(DEBUG_MODE)
    message("DEBUG_MODE is ON - no optimizations enabled")
    add_definitions(-DDEBUG_MODE=1)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -pthread -Wall -Wextra         -Wfatal-errors -pedantic -g -fPIC -Wno-reorder")
else()
    message("DEBUG_MODE is OFF - O2 optimizations enabled")
    add_definitions(-DDEBUG_MODE=0)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -pthread -Wall -Wextra -Werror -Wfatal-errors -pedantic -fPIC -Wno-reorder")
endif()

# add the actual code
add_subdirectory (library)
add_subdirectory (src)
add_subdirectory (tests)
