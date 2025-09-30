/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/cublas_error.cu
*/
#include<wchar.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>
extern "C"{
#include"../basic/basic.h"
}

extern "C"{

void Madd_cublasCreate_error(int ret, const char *func_name)
{
    if (ret == CUBLAS_STATUS_SUCCESS) return;
    wchar_t error_info[MADD_ERROR_INFO_LEN];
    switch (ret){
        case CUBLAS_STATUS_NOT_INITIALIZED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: The CUDA™ Runtime initialization failed.", func_name);
            break;
        case CUBLAS_STATUS_ALLOC_FAILED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: The resources could not be allocated.", func_name);
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: handle is NULL.", func_name);
            break;
        default:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cublasCreate returns an error 0x%x that Madd doesn't know.", func_name, ret);
    }
    Madd_Error_Add(MADD_ERROR, error_info);
}

void Madd_cublasSetStream_error(int ret, const char *func_name)
{
    if (ret == CUBLAS_STATUS_SUCCESS) return;
    wchar_t error_info[MADD_ERROR_INFO_LEN];
    switch (ret){
        case CUBLAS_STATUS_NOT_INITIALIZED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: the library was not initialized.", func_name);
            break;
        default:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cublasSetStream returns an error 0x%x that Madd doesn't know.", func_name, ret);
    }
    Madd_Error_Add(MADD_ERROR, error_info);
}

}