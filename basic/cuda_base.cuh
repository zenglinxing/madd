/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/cuda_base.cuh
*/
#ifndef _CUDA_BASE_CUH
#define _CUDA_BASE_CUH

typedef struct{
    int n_device;
    size_t *mem_free, *mem_total;
    struct cudaDeviceProp *devices;
} Madd_cuda_Device_Properties;

int Madd_N_cuda_GPU(void);
Madd_cuda_Device_Properties Madd_Get_cuda_Device_Property(void);

#endif