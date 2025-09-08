# this is to configure the config.h file
configure_file(config.h.in ${CMAKE_CURRENT_BINARY_DIR}/config.h)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/config.h
        DESTINATION include/madd)
