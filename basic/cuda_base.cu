/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/cuda_base.cu
*/
extern "C"{

#include<wchar.h>
#include<stdint.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include"basic.h"
#include"cuda_base.cuh"

int Madd_N_cuda_GPU(void)
{
    int count;
    cudaError_t res = cudaGetDeviceCount(&count);
    if (res != cudaSuccess){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        const char *cuda_info = cudaGetErrorString(res);
        swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"Madd_N_cuda_GPU: cuda func cudaGetDeviceCount reports an error: %hs.", cuda_info);
        Madd_Error_Add(MADD_ERROR, error_info);
    }
    return count;
}

Madd_cuda_Device_Properties Madd_cuda_Get_Device_Property(void)
{
    cudaError_t res_count, res_property;
    Madd_cuda_Device_Properties dp;
    dp.devices = NULL;
    res_count = cudaGetDeviceCount(&dp.n_device);
    if (res_count != cudaSuccess){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        const char *cuda_info = cudaGetErrorString(res_count);
        swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"Madd_Get_cuda_Device_Property: cuda func cudaGetDeviceCount reports an error: %hs.", cuda_info);
        Madd_Error_Add(MADD_ERROR, error_info);
    }
    dp.devices = (struct cudaDeviceProp*)malloc(dp.n_device*sizeof(struct cudaDeviceProp));

    int i_dev;
    for (i_dev=0; i_dev<dp.n_device; i_dev++){
        res_property = cudaGetDeviceProperties(dp.devices+i_dev, i_dev);
        if (res_property != cudaSuccess){
            wchar_t error_info[MADD_ERROR_INFO_LEN];
            swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"Madd_Get_cuda_Device_Property: cuda func cudaGetDeviceProperties reports an error: %hs.", cudaGetErrorString(res_property));
            Madd_Error_Add(MADD_ERROR, error_info);
        }
    }
    return dp;
}

void Madd_cuda_Get_Device_Mem(int i_dev, size_t *free_mem, size_t *total_mem)
{
    int i_current_dev;
    cudaGetDevice(&i_current_dev);
    cudaSetDevice(i_dev);
    cudaError_t res = cudaMemGetInfo(free_mem, total_mem);
    if (res != cudaSuccess){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"Madd_cuda_Device_Mem: cuda func cudaMemGetInfo reports an error: %s.", cudaGetErrorString(res));
        Madd_Error_Add(MADD_ERROR, error_info);
    }
    cudaSetDevice(i_current_dev);
}

void Madd_cuda_Device_Property_Destroy(Madd_cuda_Device_Properties dp)
{
    free(dp.devices);
}

}