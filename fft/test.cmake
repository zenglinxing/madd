add_executable(test_fft fft/test/fft.c)
target_include_directories(test_fft PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_fft PUBLIC madd)
add_test(NAME FFT
         COMMAND test_fft)