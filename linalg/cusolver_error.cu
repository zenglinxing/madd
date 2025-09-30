/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/cublas_error.cu
*/
#include<wchar.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<cusolverDn.h>
extern "C"{
#include"../basic/basic.h"
}

extern "C"{

void Madd_cusolverDnCreate_error(int ret, const char *func_name)
{
    if (ret == CUSOLVER_STATUS_SUCCESS) return;
    wchar_t error_info[MADD_ERROR_INFO_LEN];
    switch (ret){
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: The CUDA Runtime initialization failed.", func_name);
            break;
        case CUSOLVER_STATUS_ALLOC_FAILED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: The resources could not be allocated.", func_name);
            break;
        case CUSOLVER_STATUS_ARCH_MISMATCH:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: The device only supports compute capability 5.0 and above.", func_name);
            break;
        default:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnCreate returns an error %x that Madd doesn't know.", func_name, ret);
    }
    Madd_Error_Add(MADD_ERROR, error_info);
}

void Madd_cusolverDnSetStream_error(int ret, const char *func_name)
{
    if (ret == CUSOLVER_STATUS_SUCCESS) return;
    wchar_t error_info[MADD_ERROR_INFO_LEN];
    switch (ret){
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: The library was not initialized.", func_name);
            break;
        default:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnCreate returns an error %x that Madd doesn't know.", func_name, ret);
    }
    Madd_Error_Add(MADD_ERROR, error_info);
}

}