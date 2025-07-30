# coding: utf-8
#[[
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./large_number/CMakeLists.txt
]]
aux_source_directory(. MADD_SRC_LIST_LARGE_NUMBER_TEMP)
foreach(SRC ${MADD_SRC_LIST_LARGE_NUMBER_TEMP})
    list(APPEND MADD_SRC_LIST_LARGE_NUMBER large_number/${SRC})
endforeach()
set(MADD_SRC_LIST_LARGE_NUMBER ${MADD_SRC_LIST_LARGE_NUMBER} CACHE INTERNAL "MADD_SRC_LIST_LARGE_NUMBER")