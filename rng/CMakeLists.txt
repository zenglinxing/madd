# coding: utf-8
#[[
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./rng/CMakeLists.txt
]]
aux_source_directory(. MADD_SRC_LIST_RNG_TEMP)
foreach(SRC ${MADD_SRC_LIST_RNG_TEMP})
    list(APPEND MADD_SRC_LIST_RNG rng/${SRC})
endforeach()
set(MADD_SRC_LIST_RNG ${MADD_SRC_LIST_RNG} CACHE INTERNAL "MADD_SRC_LIST_RNG")