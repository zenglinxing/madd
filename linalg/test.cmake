add_executable(test_linalg-matrix-multiply linalg/test/matrix-multiply.c)
target_include_directories(test_linalg-matrix-multiply PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_linalg-matrix-multiply PUBLIC madd)
add_test(NAME Linalg-MatrixMultiply
         COMMAND test_linalg-matrix-multiply)

add_executable(test_linalg-linear-equations linalg/test/linear-equations.c)
target_include_directories(test_linalg-linear-equations PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_linalg-linear-equations PUBLIC madd)
add_test(NAME Linalg-LinearEquations
         COMMAND test_linalg-linear-equations)

if (ENABLE_CUDA)
    add_executable(test_linalg-linear-equations-cuda linalg/test/linear-equations-cuda.c)
    target_include_directories(test_linalg-linear-equations-cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_linalg-linear-equations-cuda PUBLIC madd)
    add_test(NAME Linalg-LinearEquations-CUDA
             COMMAND test_linalg-linear-equations-cuda)
endif ()