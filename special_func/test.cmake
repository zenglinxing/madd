add_executable(test_special-func-Legendre special_func/test/Legendre.c)
target_include_directories(test_special-func-Legendre PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${thread_base_INCLUDE_DIRS})
target_link_directories(test_special-func-Legendre PUBLIC ${thread_base_LIBRARY_DIRS})
target_link_libraries(test_special-func-Legendre PUBLIC madd)
add_test(NAME SpecialFunc-Legendre
         COMMAND test_special-func-Legendre)