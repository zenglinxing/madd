# coding: utf-8
#[[
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/CMakeLists.txt
]]
aux_source_directory(. MADD_SRC_LIST_LINALG_TEMP)
foreach(SRC ${MADD_SRC_LIST_LINALG_TEMP})
    list(APPEND MADD_SRC_LIST_LINALG linalg/${SRC})
endforeach()
set(MADD_SRC_LIST_LINALG ${MADD_SRC_LIST_LINALG} CACHE INTERNAL "MADD_SRC_LIST_LINALG")