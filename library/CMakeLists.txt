
set(SOURCES kafi.h jacobian_function.h util.h autogen-${UNIQUE_DEBUG_ID}-macros.h)

find_package(LAPACK REQUIRED)
link_directories(${LAPACK_LIBRARIES})
include_directories(${LAPACK_INCLUDE_DIRS})

# create library
add_library(${CPP_LIB_NAME} ${SOURCES})
target_include_directories(${CPP_LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${KAFI_SOURCE_DIR}/library> # for headers when building
    $<INSTALL_INTERFACE:${include_dest}>          # for client in install mode
)

# linking library
target_link_libraries(${CPP_LIB_NAME} ${LAPACK_LIBRARIES})
set_target_properties(${CPP_LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)

# used for 'make install', too
install(FILES ${SOURCES} DESTINATION "${include_dest}")
install(TARGETS ${CPP_LIB_NAME} EXPORT ${CPP_LIB_NAME} DESTINATION "${main_lib_dest}")
