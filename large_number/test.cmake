add_executable(test_uint128 large_number/test/uint128.c)
target_include_directories(test_uint128 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_uint128 PUBLIC madd)
add_test(NAME LargeNumber_uint128
         COMMAND test_uint128)