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

Madd_cuda_Device_Properties Madd_Get_cuda_Device_Property(void)
{
    cudaError_t res_count, res_property;
    Madd_cuda_Device_Properties dp;
    dp.devices = NULL;
    dp.mem_free = dp.mem_total = NULL;
    res_count = cudaGetDeviceCount(&dp.n_device);
    if (res_count != cudaSuccess){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        const char *cuda_info = cudaGetErrorString(res_count);
        swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"Madd_Get_cuda_Device_Property: cuda func cudaGetDeviceCount reports an error: %hs.", cuda_info);
        Madd_Error_Add(MADD_ERROR, error_info);
    }
    dp.devices = (struct cudaDeviceProp*)malloc(dp.n_device*sizeof(struct cudaDeviceProp));
    dp.mem_free = (size_t*)malloc(dp.n_device*sizeof(size_t));
    dp.mem_total = (size_t*)malloc(dp.n_device*sizeof(size_t));

    int i_dev, i_current_dev;
    cudaGetDevice(&i_current_dev);
    for (i_dev=0; i_dev<dp.n_device; i_dev++){
        res_property = cudaGetDeviceProperties(dp.devices+i_dev, i_dev);
        if (res_property != cudaSuccess){
            wchar_t error_info[MADD_ERROR_INFO_LEN];
            swprintf(error_info, MADD_ERROR_INFO_LEN-1, L"Madd_Get_cuda_Device_Property: cuda func cudaGetDeviceProperties reports an error: %hs.", cudaGetErrorString(res_property));
            Madd_Error_Add(MADD_ERROR, error_info);
        }
        cudaSetDevice(i_dev);
        cudaMemGetInfo(dp.mem_free+i_dev, dp.mem_total+i_dev);
    }
    cudaSetDevice(i_current_dev);
    return dp;
}

void Madd_cuda_Device_Property_Destroy(Madd_cuda_Device_Properties dp)
{
    free(dp.devices);
    free(dp.mem_free);
    free(dp.mem_total);
}

}