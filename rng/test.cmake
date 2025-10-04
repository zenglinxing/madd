add_executable(test_RNG-MT rng/test/RNG-MT.c)
target_include_directories(test_RNG-MT PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_RNG-MT PUBLIC madd)
add_test(NAME RNG-MT
         COMMAND test_RNG-MT)

add_executable(test_RNG-Clib rng/test/RNG-clib.c)
target_include_directories(test_RNG-Clib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_RNG-Clib PUBLIC madd)
add_test(NAME RNG-Clib
         COMMAND test_RNG-Clib)

# Xorshift
add_executable(test_RNG-Xorshift64 rng/test/RNG-Xorshift64.c)
target_include_directories(test_RNG-Xorshift64 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_RNG-Xorshift64 PUBLIC madd)
add_test(NAME RNG-Xorshift64
         COMMAND test_RNG-Xorshift64)

add_executable(test_RNG-Xorshift64s rng/test/RNG-Xorshift64s.c)
target_include_directories(test_RNG-Xorshift64s PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_RNG-Xorshift64s PUBLIC madd)
add_test(NAME RNG-Xorshift64*
         COMMAND test_RNG-Xorshift64s)

add_executable(test_RNG-Xorshift1024s rng/test/RNG-Xorshift1024s.c)
target_include_directories(test_RNG-Xorshift1024s PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_RNG-Xorshift1024s PUBLIC madd)
add_test(NAME RNG-Xorshift1024*
         COMMAND test_RNG-Xorshift1024s)

# Xoshiro
add_executable(test_RNG-Xoshiro256ss rng/test/RNG-Xoshiro256ss.c)
target_include_directories(test_RNG-Xoshiro256ss PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_RNG-Xoshiro256ss PUBLIC madd)
add_test(NAME RNG-Xoshiro256**
         COMMAND test_RNG-Xoshiro256ss)


add_executable(test_RNG-Xoshiro256p rng/test/RNG-Xoshiro256p.c)
target_include_directories(test_RNG-Xoshiro256p PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_RNG-Xoshiro256p PUBLIC madd)
add_test(NAME RNG-Xoshiro256+
         COMMAND test_RNG-Xoshiro256p)

# Xorwow
add_executable(test_RNG-Xorwow rng/test/RNG-Xorwow.c)
target_include_directories(test_RNG-Xorwow PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_RNG-Xorwow PUBLIC madd)
add_test(NAME RNG-Xorwow
         COMMAND test_RNG-Xorwow)

if (${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64" OR ${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "AMD64")
    add_executable(test_RNG-x86 rng/test/RNG-x86.c)
    target_include_directories(test_RNG-x86 PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(test_RNG-x86 PUBLIC madd)
    add_test(NAME RNG-x86
            COMMAND test_RNG-x86)
endif ()

# RNG Param
add_executable(test_RNG-Param rng/test/RNG-Param.c)
target_include_directories(test_RNG-Param PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(test_RNG-Param PUBLIC madd)
add_test(NAME RNG-Param
         COMMAND test_RNG-Param)