# coding: utf-8
#[[
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./special_func/CMakeLists.txt
]]
aux_source_directory(. MADD_SRC_LIST_SPECIAL_FUNC_TEMP)
foreach(SRC ${MADD_SRC_LIST_SPECIAL_FUNC_TEMP})
    list(APPEND MADD_SRC_LIST_SPECIAL_FUNC special_func/${SRC})
endforeach()
set(MADD_SRC_LIST_SPECIAL_FUNC ${MADD_SRC_LIST_SPECIAL_FUNC} CACHE INTERNAL "MADD_SRC_LIST_SPECIAL_FUNC")