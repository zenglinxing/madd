# coding: utf-8
#[[
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./polynomial/CMakeLists.txt
]]
aux_source_directory(. MADD_SRC_LIST_POLYNOMIAL_TEMP)
foreach(SRC ${MADD_SRC_LIST_POLYNOMIAL_TEMP})
    list(APPEND MADD_SRC_LIST_POLYNOMIAL polynomial/${SRC})
endforeach()
set(MADD_SRC_LIST_POLYNOMIAL ${MADD_SRC_LIST_POLYNOMIAL} CACHE INTERNAL "MADD_SRC_LIST_POLYNOMIAL")