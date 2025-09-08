# coding: utf-8
#[[
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./statistics/CMakeLists.txt
]]
aux_source_directory(. MADD_SRC_LIST_STATISTICS_TEMP)
foreach(SRC ${MADD_SRC_LIST_STATISTICS_TEMP})
    list(APPEND MADD_SRC_LIST_STATISTICS statistics/${SRC})
endforeach()
set(MADD_SRC_LIST_STATISTICS ${MADD_SRC_LIST_STATISTICS} CACHE INTERNAL "MADD_SRC_LIST_STATISTICS")