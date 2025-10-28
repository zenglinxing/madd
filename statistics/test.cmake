add_executable(test_statistics-kahan-summation statistics/test/Kahan-summation.c)
target_include_directories(test_statistics-kahan-summation PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${thread_base_INCLUDE_DIRS})
target_link_directories(test_statistics-kahan-summation PUBLIC ${thread_base_LIBRARY_DIRS})
target_link_libraries(test_statistics-kahan-summation PUBLIC madd)
add_test(NAME Statistics-Kahansummation
         COMMAND test_statistics-kahan-summation)