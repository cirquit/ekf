
set(SOURCES kafi.h jacobian_function.h util.h)

find_package(LAPACK REQUIRED)
link_directories(${LAPACK_LIBRARIES})
include_directories(${LAPACK_INCLUDE_DIRS})



add_library(${LIB_NAME} ${SOURCES})
target_include_directories(${LIB_NAME} PUBLIC $<BUILD_INTERFACE:${KAFI_SOURCE_DIR}/library> # for headers when building
                       						  $<INSTALL_INTERFACE:${include_dest}> # for client in install mode
                       						  )
target_link_libraries(${LIB_NAME} ${LAPACK_LIBRARIES})
set_target_properties(${LIB_NAME} PROPERTIES LINKER_LANGUAGE CXX)

#install(TARGETS ${LIB_NAME} DESTINATION "${main_lib_dest}")
install(FILES ${SOURCES} DESTINATION "${include_dest}")
install(TARGETS ${LIB_NAME} EXPORT ${LIB_NAME} DESTINATION "${main_lib_dest}")
#install(FILES ${CMAKE_CURRENT_BINARY_DIR}/config_impl.hpp DESTINATION "${include_dest}")
