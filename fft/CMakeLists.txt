# coding: utf-8
#[[
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fft/CMakeLists.txt
]]
aux_source_directory(. MADD_SRC_LIST_FFT_TEMP)
foreach(SRC ${MADD_SRC_LIST_FFT_TEMP})
    list(APPEND MADD_SRC_LIST_FFT fft/${SRC})
endforeach()
set(MADD_SRC_LIST_FFT ${MADD_SRC_LIST_FFT} CACHE INTERNAL "MADD_SRC_LIST_FFT")