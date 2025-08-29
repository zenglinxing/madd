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