cmake_minimum_required(VERSION 3.10)

project(cpp-thread-test)

set(MADD_THREAD_API "C++" CACHE STRING "Thread API.")
set_property(CACHE MADD_THREAD_API PROPERTY STRINGS
             "C++"
             "pthread"
             "Windows"
             )

message(STATUS "THREAD API: ${MADD_THREAD_API}")

#aux_source_directory(lib SRC_LIST_THREAD_BASE)
if (${MADD_THREAD_API} STREQUAL "C++")
    set(SRC_LIST_LIB_THREAD thread_base/cpp_thread.cpp)
elseif (${MADD_THREAD_API} STREQUAL "pthread")
    set(SRC_LIST_LIB_THREAD thread_base/pthread.c)
elseif (${MADD_THREAD_API} STREQUAL "Windows")
    set(SRC_LIST_LIB_THREAD thread_base/win_thread.c)
endif ()

set(MADD_SRC_LIST_THREAD_BASE ${SRC_LIST_LIB_THREAD} CACHE INTERNAL "MADD_SRC_LIST_THREAD_BASE")