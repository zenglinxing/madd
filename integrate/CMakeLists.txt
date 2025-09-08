# coding: utf-8
#[[
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./integrate/CMakeLists.txt
]]
aux_source_directory(. MADD_SRC_LIST_INTEGRATE_TEMP)
foreach(SRC ${MADD_SRC_LIST_INTEGRATE_TEMP})
    list(APPEND MADD_SRC_LIST_INTEGRATE integrate/${SRC})
endforeach()
set(MADD_SRC_LIST_INTEGRATE ${MADD_SRC_LIST_INTEGRATE} CACHE INTERNAL "MADD_SRC_LIST_INTEGRATE")