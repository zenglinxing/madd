add_executable(test_linalg-eigen linalg/test/eigen.c)
target_include_directories(test_linalg-eigen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_linalg-eigen PUBLIC madd ${OpenBLAS_LIBRARY})
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" OR ${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(test_linalg-eigen PUBLIC gfortran)
endif()
add_test(NAME Linalg-Eigen
         COMMAND test_linalg-eigen)

add_executable(test_linalg-generalized-eigen linalg/test/eigen-generalized.c)
target_include_directories(test_linalg-generalized-eigen PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_linalg-generalized-eigen PUBLIC madd ${OpenBLAS_LIBRARY})
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" OR ${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(test_linalg-generalized-eigen PUBLIC gfortran)
endif()
add_test(NAME Linalg-GeneralizedEigen
         COMMAND test_linalg-generalized-eigen)

if (ENABLE_CUDA)
    if (${CUDA_VERSION} VERSION_GREATER_EQUAL 12.6)
        add_executable(test_linalg-eigen-cuda linalg/test/eigen-cuda.c)
        target_include_directories(test_linalg-eigen-cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(test_linalg-eigen-cuda PUBLIC madd ${OpenBLAS_LIBRARY} cudart_static)
        if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" OR ${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
            target_link_libraries(test_linalg-eigen-cuda PUBLIC gfortran)
        endif()
        add_test(NAME Linalg-Eigen-CUDA
                 COMMAND test_linalg-eigen-cuda)
    endif ()
endif ()

add_executable(test_linalg-linear-equations linalg/test/linear-equations.c)
target_include_directories(test_linalg-linear-equations PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_linalg-linear-equations PUBLIC madd ${OpenBLAS_LIBRARY})
add_test(NAME Linalg-LinearEquations
         COMMAND test_linalg-linear-equations)

if (ENABLE_CUDA)
    add_executable(test_linalg-linear-equations-cuda linalg/test/linear-equations-cuda.c)
    target_include_directories(test_linalg-linear-equations-cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_linalg-linear-equations-cuda PUBLIC madd ${OpenBLAS_LIBRARY} cudart_static)
    add_test(NAME Linalg-LinearEquations-CUDA
             COMMAND test_linalg-linear-equations-cuda)

    # if CUDA >= 11.1, Linear_Equations_cuda64 is available
    if (${CUDA_VERSION} VERSION_GREATER_EQUAL 11.1)
        add_executable(test_linalg-linear-equations-cuda64 linalg/test/linear-equations-cuda64.c)
        target_include_directories(test_linalg-linear-equations-cuda64 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(test_linalg-linear-equations-cuda64 PUBLIC madd ${OpenBLAS_LIBRARY} cudart_static)
        add_test(NAME Linalg-LinearEquations-CUDA64
                COMMAND test_linalg-linear-equations-cuda64)
    endif ()
endif ()

add_executable(test_linalg-matrix-multiply linalg/test/matrix-multiply.c)
target_include_directories(test_linalg-matrix-multiply PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_linalg-matrix-multiply PUBLIC madd ${OpenBLAS_LIBRARY})
add_test(NAME Linalg-MatrixMultiply
         COMMAND test_linalg-matrix-multiply)

add_executable(test_linalg-determinant linalg/test/determinant.c)
target_include_directories(test_linalg-determinant PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_linalg-determinant PUBLIC madd ${OpenBLAS_LIBRARY})
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" OR ${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
    target_link_libraries(test_linalg-determinant PUBLIC gfortran)
endif()
add_test(NAME Linalg-Determinant
         COMMAND test_linalg-determinant)

if (ENABLE_CUDA)
    add_executable(test_linalg-determinant-cuda linalg/test/determinant-cuda.c)
    target_include_directories(test_linalg-determinant-cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_linalg-determinant-cuda PUBLIC madd ${OpenBLAS_LIBRARY})
    if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" OR ${CMAKE_HOST_SYSTEM_NAME} MATCHES "Linux")
        target_link_libraries(test_linalg-determinant-cuda PUBLIC gfortran)
    endif()
    add_test(NAME Linalg-Determinant-CUDA
            COMMAND test_linalg-determinant-cuda)
endif ()