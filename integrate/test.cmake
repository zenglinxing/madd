add_executable(test_integrate-Simpson integrate/test/Simpson.c)
target_include_directories(test_integrate-Simpson PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_integrate-Simpson PUBLIC madd)
add_test(NAME Integrate_Simpson
         COMMAND test_integrate-Simpson)