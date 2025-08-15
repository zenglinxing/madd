add_executable(test_basic-print-constant basic/test/print-constant.c)
target_include_directories(test_basic-print-constant PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_basic-print-constant PUBLIC madd)
add_test(NAME Basic-PrintConstant
         COMMAND test_basic-print-constant)

add_executable(test_basic-binary-1-number basic/test/binary-1-number.c)
target_include_directories(test_basic-binary-1-number PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_basic-binary-1-number PUBLIC madd)
add_test(NAME Basic-Binary1Number
         COMMAND test_basic-binary-1-number)

add_executable(test_basic-cnum basic/test/cnum.c)
target_include_directories(test_basic-cnum PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_basic-cnum PUBLIC madd)
add_test(NAME Basic-Cnum
         COMMAND test_basic-cnum)

add_executable(test_basic-log2-1 basic/test/log2-1.c)
target_include_directories(test_basic-log2-1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_basic-log2-1 PUBLIC madd)
add_test(NAME Basic-Log2FloorCeil
         COMMAND test_basic-log2-1)

add_executable(test_basic-log2-2 basic/test/log2-2.c)
target_include_directories(test_basic-log2-2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_basic-log2-2 PUBLIC madd)
add_test(NAME Basic-Log2Full
         COMMAND test_basic-log2-2)

add_executable(test_basic-norm basic/test/norm.c)
target_include_directories(test_basic-norm PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_basic-norm PUBLIC madd)
add_test(NAME Basic-Norm
         COMMAND test_basic-norm)

add_executable(test_basic-bit-reverse basic/test/bit-reverse.c)
target_include_directories(test_basic-bit-reverse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_basic-bit-reverse PUBLIC madd)
add_test(NAME Basic-BitReverse
         COMMAND test_basic-bit-reverse)

add_executable(test_basic-byte-reverse basic/test/byte-reverse.c)
target_include_directories(test_basic-byte-reverse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_basic-byte-reverse PUBLIC madd)
add_test(NAME Basic-ByteReverse
         COMMAND test_basic-byte-reverse)

add_executable(test_basic-file-endian basic/test/file-endian.c)
target_include_directories(test_basic-file-endian PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_basic-file-endian PUBLIC madd)
add_test(NAME Basic-FileEndian
         COMMAND test_basic-file-endian)

add_executable(test_basic-error-info basic/test/error-info.c)
target_include_directories(test_basic-error-info PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_basic-error-info PUBLIC madd)
add_test(NAME Basic-MaddErrorInfo
         COMMAND test_basic-error-info)

if (ENABLE_CUDA)
    add_executable(test_basic-cuda-base basic/test/cuda-base.c)
    target_include_directories(test_basic-cuda-base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_basic-cuda-base PUBLIC madd cudart_static)
    add_test(NAME Basic-Madd-cudaBase
            COMMAND test_basic-cuda-base)
endif()

add_executable(test_basic-hash-float basic/test/hash-float.c)
target_include_directories(test_basic-hash-float PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_basic-hash-float PUBLIC madd)
add_test(NAME Basic-HashFloat
         COMMAND test_basic-hash-float)