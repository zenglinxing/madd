add_executable(test_integrate-Trapeze integrate/test/trapeze.c)
target_include_directories(test_integrate-Trapeze PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_integrate-Trapeze PUBLIC madd)
add_test(NAME Integrate_Trapeze
         COMMAND test_integrate-Trapeze)

add_executable(test_integrate-Simpson integrate/test/Simpson.c)
target_include_directories(test_integrate-Simpson PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_integrate-Simpson PUBLIC madd)
add_test(NAME Integrate_Simpson
         COMMAND test_integrate-Simpson)

add_executable(test_integrate-Gauss-Legendre integrate/test/Gauss-Legendre.c)
target_include_directories(test_integrate-Gauss-Legendre PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_integrate-Gauss-Legendre PUBLIC madd)
add_test(NAME Integrate_GaussLegendre
         COMMAND test_integrate-Gauss-Legendre)

add_executable(test_integrate-Gauss-Laguerre integrate/test/Gauss-Laguerre.c)
target_include_directories(test_integrate-Gauss-Laguerre PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_integrate-Gauss-Laguerre PUBLIC madd)
add_test(NAME Integrate_GaussLaguerre
         COMMAND test_integrate-Gauss-Laguerre)