# coding: utf-8
#[[
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/CMakeLists.txt
]]
aux_source_directory(. MADD_SRC_LIST_BASIC_TEMP)
foreach(SRC ${MADD_SRC_LIST_BASIC_TEMP})
    list(APPEND MADD_SRC_LIST_BASIC basic/${SRC})
endforeach()
set(MADD_SRC_LIST_BASIC ${MADD_SRC_LIST_BASIC} CACHE INTERNAL "MADD_SRC_LIST_BASIC")
#message(STATUS "MADD_SRC_LIST_BASIC: ${MADD_SRC_LIST_BASIC}")