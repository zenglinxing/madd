/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/cuda_base.c
*/
#ifndef ENABLE_CUDA

#include<wchar.h>
#include"basic.h"
#include"cuda_base.cuh"

int Madd_N_cuda_GPU(void)
{
    Madd_Error_Add(MADD_WARNING, L"Madd_N_cuda_GPU: you hadn't compile Madd library with CUDA enabled. Try to re-compile Madd with ENABLE_CUDA option ON.");
    return 0;
}

#endif