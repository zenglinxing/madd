/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/cuda_base.cuh
*/
#ifndef MADD_CUDA_BASE_CUH
#define MADD_CUDA_BASE_CUH

#if defined(__CUDACC_VER__) && !defined(__CUDACC_VER_MAJOR__)
#define __CUDACC_VER_MAJOR__ (__CUDACC_VER__/10000)
#define __CUDACC_VER_MINOR__ ((__CUDACC_VER__/100)%100)
#define __CUDACC_VER_BUILD__ (__CUDACC_VER__%100)
#endif

typedef struct{
    int n_device;
    struct cudaDeviceProp *devices;
} Madd_cuda_Device_Properties;

int Madd_N_cuda_GPU(void);
Madd_cuda_Device_Properties Madd_cuda_Get_Device_Property(void);
void Madd_cuda_Get_Device_Mem(int i_dev, size_t *free_mem, size_t *total_mem);

void Madd_cudaMalloc_error(int ret, const char *func_name, size_t size_alloc, const char *var_name);
void Madd_cudaSetStream_error(int ret, const char *func_name);

#endif /* MADD_CUDA_BASE_H */