/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fft/fft-error.cu
*/
#include<wchar.h>
#include<stdlib.h>
#include<cuda_runtime.h>
#include<cufft.h>
extern "C"{
#include"../basic/basic.h"
}

extern "C"{

void Madd_cufftPlan1d_error(int ret_plan, char *func_name)
{
    wchar_t error_info[MADD_ERROR_INFO_LEN];
    switch (ret_plan){
        case CUFFT_INVALID_PLAN:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cufftPlan1d (CUFFT_INVALID_PLAN): Handle is not valid when the plan is locked.", func_name);
            break;
        case CUFFT_ALLOC_FAILED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cufftPlan1d (CUFFT_ALLOC_FAILED): The allocation of GPU resources for the plan failed.", func_name);
            break;
        case CUFFT_INVALID_VALUE:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cufftPlan1d (CUFFT_INVALID_VALUE): One or more invalid parameters were passed to the API.", func_name);
            break;
        case CUFFT_INTERNAL_ERROR:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cufftPlan1d (CUFFT_INTERNAL_ERROR): An internal driver error was detected.", func_name);
            break;
        case CUFFT_SETUP_FAILED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cufftPlan1d (CUFFT_SETUP_FAILED): The cuFFT library failed to initialize.", func_name);
            break;
        case CUFFT_INVALID_SIZE:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cufftPlan1d (CUFFT_INVALID_SIZE): The nx or batch parameter is not a supported size.", func_name);
            break;
        //case CUFFT_MISSING_DEPENDENCY:
        //    swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cufftPlan1d (CUFFT_MISSING_DEPENDENCY): The cuFFT library was unable to find a dependency either because it is missing or the version found is incompatible.", func_name);
        //    break;
        //case CUFFT_NVRTC_FAILURE:
        //    swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cufftPlan1d (CUFFT_NVRTC_FAILURE): NVRTC encountered an error during planning.", func_name);
        //    break;
        //case CUFFT_NVJITLINK_FAILURE:
        //    swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cufftPlan1d (CUFFT_NVJITLINK_FAILURE): nvJitLink encountered an error during planning.", func_name);
        //    break;
        default:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cufftPlan1d returns an error %x that Madd doesn't know.", func_name, ret_plan);
    }
    Madd_Error_Add(MADD_ERROR, error_info);
}

void Madd_cufftExec_error(int ret, const char *func_name, const char *func_exec_name)
{
    wchar_t error_info[MADD_ERROR_INFO_LEN];
    switch (ret){
        case CUFFT_INVALID_PLAN:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUFFT_INVALID_PLAN): The plan parameter is not a valid handle.", func_name, func_exec_name);
            break;
        case CUFFT_INVALID_VALUE:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUFFT_INVALID_VALUE): At least one of the parameters idata, odata, and direction is not valid.", func_name, func_exec_name);
            break;
        case CUFFT_INTERNAL_ERROR:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUFFT_INTERNAL_ERROR): An internal driver error was detected.", func_name, func_exec_name);
            break;
        case CUFFT_EXEC_FAILED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUFFT_EXEC_FAILED): cuFFT failed to execute the transform on the GPU.", func_name, func_exec_name);
            break;
        case CUFFT_SETUP_FAILED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUFFT_SETUP_FAILED): The cuFFT library failed to initialize.", func_name, func_exec_name);
            break;
        default:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs returns an error %x that Madd doesn't know.", func_name, func_exec_name, ret);
    }
    Madd_Error_Add(MADD_ERROR, error_info);
}

}