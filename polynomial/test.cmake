add_executable(test_polynomial-poly1d polynomial/test/poly1d.c)
target_include_directories(test_polynomial-poly1d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_polynomial-poly1d PUBLIC madd)
add_test(NAME Polynomial-Poly1d
         COMMAND test_polynomial-poly1d)

add_executable(test_polynomial-poly1d-deriv1 polynomial/test/poly1d-deriv1.c)
target_include_directories(test_polynomial-poly1d-deriv1 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_polynomial-poly1d-deriv1 PUBLIC madd)
add_test(NAME Polynomial-Poly1d-Derivative1
         COMMAND test_polynomial-poly1d-deriv1)

add_executable(test_polynomial-poly1d-deriv2 polynomial/test/poly1d-deriv2.c)
target_include_directories(test_polynomial-poly1d-deriv2 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_polynomial-poly1d-deriv2 PUBLIC madd)
add_test(NAME Polynomial-Poly1d-Derivative2
         COMMAND test_polynomial-poly1d-deriv2)

add_executable(test_polynomial-poly1d-integrate polynomial/test/poly1d-integrate.c)
target_include_directories(test_polynomial-poly1d-integrate PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_polynomial-poly1d-integrate PUBLIC madd)
add_test(NAME Polynomial-Poly1d-Integrate
         COMMAND test_polynomial-poly1d-integrate)

add_executable(test_polynomial-poly1d-nintegrate polynomial/test/poly1d-nintegrate.c)
target_include_directories(test_polynomial-poly1d-nintegrate PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_polynomial-poly1d-nintegrate PUBLIC madd)
add_test(NAME Polynomial-Poly1d-NIntegrate
         COMMAND test_polynomial-poly1d-nintegrate)

# developing
#add_executable(test_polynomial-poly1d-deriv-Norder polynomial/test/poly1d-deriv-Norder.c)
#target_include_directories(test_polynomial-poly1d-deriv-Norder PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#target_link_libraries(test_polynomial-poly1d-deriv-Norder PUBLIC madd)
#add_test(NAME Polynomial-Poly1d-Derive-Norder
#         COMMAND test_polynomial-poly1d-deriv-Norder)

