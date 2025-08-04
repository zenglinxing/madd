# coding: utf-8
#[[
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./CMakeLists.txt
]]
cmake_minimum_required(VERSION 3.10)

SET(CMAKE_C_USE_RESPONSE_FILE_FOR_OBJECTS 1)
SET(CMAKE_CXX_USE_RESPONSE_FILE_FOR_OBJECTS 1)
SET(CMAKE_C_USE_RESPONSE_FILE_FOR_INCLUDES 1)
SET(CMAKE_CXX_USE_RESPONSE_FILE_FOR_INCLUDES 1)

set(CMAKE_C_RESPONSE_FILE_LINK_FLAG "@")
set(CMAKE_CXX_RESPONSE_FILE_LINK_FLAG "@")

if (NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17 CACHE INTERNAL "set Cmake C++ standard.")
else()
    set(CMAKE_CXX_STANDARD ${CMAKE_CXX_STANDARD} CACHE INTERNAL "set Cmake C++ standard.")
endif ()
message(STATUS "C++ standard: c++${CMAKE_CXX_STANDARD}")

project(madd
        VERSION 0.0.0
        LANGUAGES C CXX
        DESCRIPTION "Math Addition: A mathematics extension for C programming."
        )

#message(STATUS "C compiler ID: ${CMAKE_C_COMPILER_ID}")

# multithread
option(ENABLE_MULTITHREAD "Enable multithread." ON)
if (ENABLE_MULTITHREAD)
    add_definitions(-DMADD_ENABLE_MULTITHREAD)
    set(MADD_DEF_ENABLE_MULTITHREAD "#define MADD_ENABLE_MULTITHREAD")
endif()

# CUDA
option(ENABLE_CUDA "Enable support for CUDA. NVIDIA compiler is required" OFF)
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architecture. You should check it on https://developer.nvidia.com/cuda-gpus")
if (ENABLE_CUDA)
    enable_language(CUDA)
endif()

# quadmath
option(ENABLE_QUADPRECISION "Enable quadmath (128-bit float number) in GNU compiler" OFF)
if (ENABLE_QUADPRECISION)
    add_definitions(-DENABLE_QUADPRECISION)
    set(MADD_DEF_ENABLE_QUADPRECISION "#define ENABLE_QUADPRECISION")
endif()

# latex document
option(LATEX_DOCUMENT "Build LaTeX document guide. Requires LaTeX compiler." ON)

# test
option(ENABLE_TEST "Enable test to check the functions work well." ON)

# =================================================================================================
# configure the config.h
# =================================================================================================
include(config.cmake)

# =================================================================================================
# source list
# =================================================================================================
add_subdirectory(basic)
add_subdirectory(data_struct)
add_subdirectory(rng)
add_subdirectory(large_number)
add_subdirectory(sort)
set(MADD_SRC_LIST_THREAD_BASE)
if (ENABLE_MULTITHREAD)
    add_subdirectory(thread_base)
else()
    unset(MADD_SRC_LIST_THREAD_BASE CACHE)
endif()
message(STATUS "thread base src: ${MADD_SRC_LIST_THREAD_BASE}")

if (LATEX_DOCUMENT)
    add_subdirectory(doc)
endif ()

# =================================================================================================
# library
# =================================================================================================
add_definitions(-DMADD_BUILD_STAGE)
#add_compile_options(-fsanitize=address)
add_library(madd STATIC
            ${MADD_SRC_LIST_BASIC} # basic/
            ${MADD_SRC_LIST_DATA_STRUCT} # data_struct/
            ${MADD_SRC_LIST_RNG} # rng/
            ${MADD_SRC_LIST_LARGE_NUMBER} # large_number/
            ${MADD_SRC_LIST_SORT} # sort/
            ${MADD_SRC_LIST_THREAD_BASE} # thread_base
)
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "GNU" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "AppleClang")
    target_compile_options(madd PUBLIC "-march=native")
    target_link_libraries(madd PUBLIC m)
endif()
if ("${CMAKE_C_COMPILER_ID}" STREQUAL "MSVC")
    target_compile_options(madd PUBLIC "-O2")
else()
    target_compile_options(madd PUBLIC "-O3")
endif()
if (ENABLE_CUDA)
    target_link_libraries(madd PRIVATE cudart_static cublas)
    target_compile_definitions(madd PRIVATE ENABLE_CUDA)
    set_target_properties(madd
                          PROPERTIES
                          CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(madd
                          PROPERTIES
                          CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()
if (ENABLE_QUADPRECISION)
    target_link_libraries(madd PRIVATE quadmath)
endif()
if (ENABLE_MULTITHREAD)
    if (${MADD_THREAD_API} STREQUAL "pthread")
        target_link_libraries(madd PRIVATE pthread)
    endif()
endif()

# =================================================================================================
# test
# =================================================================================================
if (ENABLE_TEST)
    enable_testing()
    include(basic/test.cmake)
    include(data_struct/test.cmake)
    include(rng/test.cmake)
    include(large_number/test.cmake)
    include(sort/test.cmake)
    if (ENABLE_MULTITHREAD)
        include(thread_base/test.cmake)
    endif ()
endif()

# =================================================================================================
# install
# =================================================================================================
# Generate Config.cmake
set(INCLUDE_DIRS include)
set(LIBRARIES time_unified)
set(LIB_DIR lib)

include(CMakePackageConfigHelpers)

write_basic_package_version_file(
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake
    VERSION ${CMAKE_PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion)
configure_package_config_file(
    ${PROJECT_SOURCE_DIR}/${PROJECT_NAME}Config.cmake.in
    ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    INSTALL_DESTINATION lib/cmake/madd
    PATH_VARS INCLUDE_DIRS LIBRARIES LIB_DIR
    INSTALL_PREFIX ${CMAKE_INSTALL_PREFIX}
)

install(
    FILES ${PROJECT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake ${PROJECT_BINARY_DIR}/${PROJECT_NAME}Config.cmake
    DESTINATION lib/cmake/madd
)
# header files
file(GLOB_RECURSE HEADER_FILES 
     RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     "*.h" "*.cuh")
file(GLOB_RECURSE HEADER_REMOVE_FILES
     RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     "build/*" "build*/*")
list(REMOVE_ITEM HEADER_FILES ${HEADER_REMOVE_FILES})
# 安装每个头文件并保持相对路径
foreach(header ${HEADER_FILES})
    get_filename_component(header_dir "${header}" DIRECTORY)
    install(
        FILES "${CMAKE_CURRENT_SOURCE_DIR}/${header}"
        DESTINATION "${CMAKE_INSTALL_PREFIX}/include/madd/${header_dir}"
    )
endforeach()
# lib
install(
    TARGETS madd
    ARCHIVE DESTINATION lib
)