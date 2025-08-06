# coding: utf-8
#[[
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fmin/CMakeLists.txt
]]
aux_source_directory(. MADD_SRC_LIST_FMIN_TEMP)
foreach(SRC ${MADD_SRC_LIST_FMIN_TEMP})
    list(APPEND MADD_SRC_LIST_FMIN fmin/${SRC})
endforeach()
set(MADD_SRC_LIST_FMIN ${MADD_SRC_LIST_FMIN} CACHE INTERNAL "MADD_SRC_LIST_FMIN")