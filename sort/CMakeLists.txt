# coding: utf-8
#[[
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./sort/CMakeLists.txt
]]
aux_source_directory(. MADD_SRC_LIST_SORT_TEMP)
foreach(SRC ${MADD_SRC_LIST_SORT_TEMP})
    list(APPEND MADD_SRC_LIST_SORT sort/${SRC})
endforeach()
set(MADD_SRC_LIST_SORT ${MADD_SRC_LIST_SORT} CACHE INTERNAL "MADD_SRC_LIST_SORT")