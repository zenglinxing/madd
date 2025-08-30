add_executable(test_polynomial-poly1d polynomial/test/poly1d.c)
target_include_directories(test_polynomial-poly1d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_polynomial-poly1d PUBLIC madd)
add_test(NAME Polynomial-Poly1d
         COMMAND test_polynomial-poly1d)

# developing
#add_executable(test_polynomial-poly1d-deriv-Norder polynomial/test/poly1d-deriv-Norder.c)
#target_include_directories(test_polynomial-poly1d-deriv-Norder PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
#target_link_libraries(test_polynomial-poly1d-deriv-Norder PUBLIC madd)
#add_test(NAME Polynomial-Poly1d-Derive-Norder
#         COMMAND test_polynomial-poly1d-deriv-Norder)

