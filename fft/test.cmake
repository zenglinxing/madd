add_executable(test_fft fft/test/fft.c)
target_include_directories(test_fft PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_fft PUBLIC madd)
add_test(NAME FFT
         COMMAND test_fft)

add_executable(test_fft-dct2 fft/test/dct2.c)
target_include_directories(test_fft-dct2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_fft-dct2 PUBLIC madd)
add_test(NAME FFT-DCT2
         COMMAND test_fft-dct2)