# coding: utf-8
#[[
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./CMakeLists.txt
]]
aux_source_directory(. MADD_SRC_LIST_DATA_STRUCT_TEMP)
foreach(SRC ${MADD_SRC_LIST_DATA_STRUCT_TEMP})
    list(APPEND MADD_SRC_LIST_DATA_STRUCT data_struct/${SRC})
endforeach()
set(MADD_SRC_LIST_DATA_STRUCT ${MADD_SRC_LIST_DATA_STRUCT} CACHE INTERNAL "MADD_SRC_LIST_DATA_STRUCT")