add_executable(test_linalg-matrix-multiply linalg/test/matrix-multiply.c)
target_include_directories(test_linalg-matrix-multiply PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_linalg-matrix-multiply PUBLIC madd)
add_test(NAME Linalg-MatrixMultiply
         COMMAND test_linalg-matrix-multiply)