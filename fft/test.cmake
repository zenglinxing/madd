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

if (ENABLE_CUDA)
    add_executable(test_fft-cuda fft/test/fft-cuda.c)
    target_include_directories(test_fft-cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_fft-cuda PUBLIC madd)
    add_test(NAME FFT-CUDA-n-default
            COMMAND test_fft-cuda)
    add_test(NAME FFT-CUDA-n9
            COMMAND test_fft-cuda 9)
endif ()