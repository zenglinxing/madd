add_executable(test_linalg-eigen linalg/test/eigen.c)
target_include_directories(test_linalg-eigen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_linalg-eigen PUBLIC madd ${OpenBLAS_LIBRARY})
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    target_link_libraries(test_linalg-eigen PUBLIC gfortran)
endif()
add_test(NAME Linalg-Eigen
         COMMAND test_linalg-eigen)

add_executable(test_linalg-linear-equations linalg/test/linear-equations.c)
target_include_directories(test_linalg-linear-equations PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_linalg-linear-equations PUBLIC madd ${OpenBLAS_LIBRARY})
add_test(NAME Linalg-LinearEquations
         COMMAND test_linalg-linear-equations)

if (ENABLE_CUDA)
    add_executable(test_linalg-linear-equations-cuda linalg/test/linear-equations-cuda.c)
    target_include_directories(test_linalg-linear-equations-cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_linalg-linear-equations-cuda PUBLIC madd ${OpenBLAS_LIBRARY})
    add_test(NAME Linalg-LinearEquations-CUDA
             COMMAND test_linalg-linear-equations-cuda)
endif ()

add_executable(test_linalg-matrix-multiply linalg/test/matrix-multiply.c)
target_include_directories(test_linalg-matrix-multiply PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_linalg-matrix-multiply PUBLIC madd ${OpenBLAS_LIBRARY})
add_test(NAME Linalg-MatrixMultiply
         COMMAND test_linalg-matrix-multiply)