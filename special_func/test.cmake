add_executable(test_special-func-Legendre special_func/test/Legendre.c)
target_include_directories(test_special-func-Legendre PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_special-func-Legendre PUBLIC madd)
add_test(NAME SpecialFunc-Legendre
         COMMAND test_special-func-Legendre)