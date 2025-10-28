add_executable(test_large-number-uint128 large_number/test/uint128.c)
target_include_directories(test_large-number-uint128 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${thread_base_INCLUDE_DIRS})
target_link_directories(test_large-number-uint128 PUBLIC ${thread_base_LIBRARY_DIRS})
target_link_libraries(test_large-number-uint128 PUBLIC madd)
add_test(NAME LargeNumber_uint128
         COMMAND test_large-number-uint128)