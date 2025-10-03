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
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnCreate (CUSOLVER_STATUS_NOT_INITIALIZED) The CUDA Runtime initialization failed.", func_name);
            break;
        case CUSOLVER_STATUS_ALLOC_FAILED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnCreate (CUSOLVER_STATUS_ALLOC_FAILED) The resources could not be allocated.", func_name);
            break;
        case CUSOLVER_STATUS_ARCH_MISMATCH:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnCreate (CUSOLVER_STATUS_ARCH_MISMATCH) The device only supports compute capability 5.0 and above.", func_name);
            break;
        default:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnCreate returns an error 0x%x that Madd doesn't know.", func_name, ret);
    }
    Madd_Error_Add(MADD_ERROR, error_info);
}

void Madd_cusolverDnSetStream_error(int ret, const char *func_name)
{
    if (ret == CUSOLVER_STATUS_SUCCESS) return;
    wchar_t error_info[MADD_ERROR_INFO_LEN];
    switch (ret){
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnSetStream (CUSOLVER_STATUS_NOT_INITIALIZED) The library was not initialized.", func_name);
            break;
        default:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnCreate returns an error 0x%x that Madd doesn't know.", func_name, ret);
    }
    Madd_Error_Add(MADD_ERROR, error_info);
}

void Madd_cusolverDnCreateParams_error(int ret, const char *func_name)
{
    if (ret == CUSOLVER_STATUS_SUCCESS) return;
    wchar_t error_info[MADD_ERROR_INFO_LEN];
    switch (ret){
        case CUSOLVER_STATUS_ALLOC_FAILED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnCreateParams (CUSOLVER_STATUS_ALLOC_FAILED) The resources could not be allocated.", func_name);
            break;
        default:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnCreate returns an error 0x%x that Madd doesn't know.", func_name, ret);
    }
    Madd_Error_Add(MADD_ERROR, error_info);
}

void Madd_cusolverDnSetAdvOptions_error(int ret, const char *func_name)
{
    if (ret == CUSOLVER_STATUS_SUCCESS) return;
    wchar_t error_info[MADD_ERROR_INFO_LEN];
    switch (ret){
        case CUSOLVER_STATUS_INVALID_VALUE:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnSetAdvOptions (CUSOLVER_STATUS_INVALID_VALUE) Wrong combination of function and algo.", func_name);
            break;
        default:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnSetAdvOptions returns an error 0x%x that Madd doesn't know.", func_name, ret);
    }
    Madd_Error_Add(MADD_ERROR, error_info);
}

}