/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fft/fft.cu
*/
#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>
#include<cuda_runtime.h>
#include<cufft.h>
extern "C"{
#include"fft.h"
#include"fft.cuh"
#include"../basic/basic.h"
}

#define FFT_CUDA__ALGORITHM(Cnum, cufft_type, \
                            cufftDoubleComplex, \
                            cufftExecZ2Z, cufftExecZ2Z_name, \
                            Cnum_Div) \
{ \
    if (n == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n is 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (arr == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: arr is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (fft_direction != MADD_FFT_FORWARD && fft_direction != MADD_FFT_INVERSE){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: fft_direction should be either MADD_FFT_FORWARD or MADD_FFT_INVERSE. You set %d.", __func__, fft_direction); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    int cufft_direction = (fft_direction == MADD_FFT_FORWARD) ? CUFFT_FORWARD : CUFFT_INVERSE; \
 \
    /* cuda plan */ \
    cufftHandle handle; \
    cufftResult ret_plan = cufftPlan1d(&handle, n, cufft_type, 1); \
    if (ret_plan != CUFFT_SUCCESS){ \
        Madd_cufftPlan1d_error(ret_plan, __func__); \
        return false; \
    } \
    cudaStream_t stream; \
    cudaError_t ret_stream_create = cudaStreamCreate(&stream); \
    if (ret_stream_create != cudaSuccess){ \
        Madd_cudaSetStream_error(ret_stream_create, __func__); \
        return false; \
    } \
    cufftSetStream(handle, stream); \
 \
    /* copy data */ \
    size_t size_arr = n*sizeof(cufftDoubleComplex); \
    cufftDoubleComplex *d_arr; \
    cudaError_t ret_malloc = cudaMalloc(&d_arr, size_arr); \
    if (ret_malloc != cudaSuccess){ \
        Madd_cudaMalloc_error(ret_malloc, __func__, size_arr, "d_arr"); \
        return false; \
    } \
    cudaMemcpy(d_arr, arr, size_arr, cudaMemcpyHostToDevice); \
 \
    /* execute fft */ \
    cufftResult ret_exec = cufftExecZ2Z(handle, d_arr, d_arr, cufft_direction); \
    if (ret_exec != CUFFT_SUCCESS){ \
        Madd_cufftExec_error(ret_exec, __func__, "cufftExecZ2Z"); \
        cudaFree(d_arr); \
        return false; \
    } \
    cudaStreamSynchronize(stream); \
    cufftDestroy(handle); \
 \
    cudaMemcpy(arr, d_arr, size_arr, cudaMemcpyDeviceToHost); \
    cudaFree(d_arr); \
 \
    if (fft_direction == MADD_FFT_INVERSE){ \
        Cnum div; \
        div.real = n; \
        div.imag = 0; \
        for (uint64_t i=0; i<n; i++){ \
            arr[i] = Cnum_Div(arr[i], div); \
        } \
    } \
 \
    return true; \
} \

extern "C"{

bool Fast_Fourier_Transform_cuda(int n, Cnum *arr, int fft_direction)
FFT_CUDA__ALGORITHM(Cnum, CUFFT_Z2Z,
                    cufftDoubleComplex,
                    cufftExecZ2Z, "cufftExecZ2Z",
                    Cnum_Div)

bool Fast_Fourier_Transform_cuda_c32(int n, Cnum32 *arr, int fft_direction)
FFT_CUDA__ALGORITHM(Cnum32, CUFFT_C2C,
                    cufftComplex,
                    cufftExecC2C, "cufftExecC2C",
                    Cnum_Div_c32)

}