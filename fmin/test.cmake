# Nealder-Mead
add_executable(test_fmin-NM fmin/test/fmin-NM.c)
target_include_directories(test_fmin-NM PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_fmin-NM PUBLIC madd)
add_test(NAME Fmin-NealderMead
         COMMAND test_fmin-NM)

# Particle Swarm Optimization
add_executable(test_fmin-PSO fmin/test/fmin-PSO.c)
target_include_directories(test_fmin-PSO PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_fmin-PSO PUBLIC madd)
add_test(NAME Fmin-ParticleSwarmOptimization
         COMMAND test_fmin-PSO)

# Newton Iteration Methods
add_executable(test_fmin-Newton-Iteration fmin/test/fmin-Newton-Iteration.c)
target_include_directories(test_fmin-Newton-Iteration PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_fmin-Newton-Iteration PUBLIC madd ${OpenBLAS_LIBRARY})
add_test(NAME Fmin-NewtonIterationMethods
         COMMAND test_fmin-Newton-Iteration)

# Jacobi Iteration Methods
add_executable(test_fmin-Jacobi-Iteration-Sparse fmin/test/fmin-Jacobi-Iteration-Sparse.c)
target_include_directories(test_fmin-Jacobi-Iteration-Sparse PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_fmin-Jacobi-Iteration-Sparse PUBLIC madd)
add_test(NAME Fmin-JacobiIterationMethods-Sparse
         COMMAND test_fmin-Jacobi-Iteration-Sparse)