# coding: utf-8
#[[
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./interpolation/CMakeLists.txt
]]
aux_source_directory(. MADD_SRC_LIST_INTERPOLATION_TEMP)
foreach(SRC ${MADD_SRC_LIST_INTERPOLATION_TEMP})
    list(APPEND MADD_SRC_LIST_INTERPOLATION interpolation/${SRC})
endforeach()
set(MADD_SRC_LIST_INTERPOLATION ${MADD_SRC_LIST_INTERPOLATION} CACHE INTERNAL "MADD_SRC_LIST_INTERPOLATION")