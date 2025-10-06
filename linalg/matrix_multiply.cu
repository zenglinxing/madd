/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/matrix_multiply.cu
*/
#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>
#include<cuda_runtime.h>
#include<cublas_v2.h>
extern "C"{
#include"linalg.h"
#include"../basic/basic.h"
}

static inline void gemm_error(cublasStatus_t ret, const char *func_name, const char *func_gemm_name)
{
    if (ret == CUBLAS_STATUS_SUCCESS) return;
    wchar_t error_info[MADD_ERROR_INFO_LEN];
    switch (ret){
        case CUBLAS_STATUS_NOT_INITIALIZED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUBLAS_STATUS_NOT_INITIALIZED) The library was not initialized.", func_name, func_gemm_name);
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUBLAS_STATUS_INVALID_VALUE) If m < 0 or n < 0 or l < 0, or if transa and transb are not one of CUBLAS_OP_N, CUBLAS_OP_C, CUBLAS_OP_T, or if lda < max(1, m) when transa == CUBLAS_OP_N and lda < max(1, l) otherwise, or if ldb < max(1, l) when transb == CUBLAS_OP_N and ldb < max(1, n) otherwise, or if ldc < max(1, m), or if alpha or beta are NULL, or if res is NULL when beta is not zero.", func_name, func_gemm_name);
            break;
        case CUBLAS_STATUS_ARCH_MISMATCH:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUBLAS_STATUS_ARCH_MISMATCH) In the case of cublasHgemm() the device does not support math in half precision.", func_name, func_gemm_name);
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUBLAS_STATUS_EXECUTION_FAILED) The function failed to launch on the GPU.", func_name, func_gemm_name);
            break;
        default:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs returns an error 0x%x that Madd doesn't know.", func_name, func_gemm_name, ret);
    }
    Madd_Error_Add(MADD_ERROR, error_info);
}

#define MATRIX_MULTIPLY_CUDA__ALGORITHM(num_type, cublasDgemm, Matrix_Transpose, func_gemm_name) \
{ \
    if (m == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: m is 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (n == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n is 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (l == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: l is 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (a == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix a is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (b == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix b is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (res == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix res is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    /* memory */ \
    uint64_t mm = m, nn = n, ll = l; \
    size_t size_a = mm*ll*sizeof(num_type), size_b = ll*nn*sizeof(num_type), size_res = mm*nn*sizeof(num_type); \
    num_type *d_a, *d_b, *d_res; \
    cudaError_t cuda_malloc_a; \
    cuda_malloc_a = cudaMalloc(&d_a, size_a + size_b + size_res); \
    if (cuda_malloc_a != cudaSuccess){ \
        Madd_cudaMalloc_error(cuda_malloc_a, __func__, size_a + size_b + size_res, "d_a & d_b & d_res"); \
        return false; \
    } \
    d_b = d_a + mm*ll; \
    d_res = d_b + ll*nn; \
    cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice); \
    cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice); \
 \
    /* matrix multiply */ \
    cudaStream_t stream; \
    cudaError_t ret_stream = cudaStreamCreate(&stream); \
    if (ret_stream != cudaSuccess){ \
        cudaFree(d_a); \
        Madd_cudaSetStream_error(ret_stream, __func__); \
        return false; \
    } \
    cublasHandle_t handle; \
    cublasStatus_t ret_handle = cublasCreate(&handle); \
    if (ret_handle != CUBLAS_STATUS_SUCCESS){ \
        cudaFree(d_a); \
        cudaStreamDestroy(stream); \
        Madd_cublasCreate_error(ret_handle, __func__); \
        return false; \
    } \
    cublasSetStream(handle, stream); \
 \
    num_type alpha = 1, beta = 0; \
    cublasStatus_t ret_gemm = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, \
                m, n, l, \
                &alpha, \
                d_a, l, \
                d_b, n, \
                &beta, \
                d_res, m); \
    cudaStreamSynchronize(stream); \
    if (ret_gemm != CUBLAS_STATUS_SUCCESS){ \
        gemm_error(ret_gemm, __func__, func_gemm_name); \
        return false; \
    } \
 \
    cudaMemcpy(res, d_res, size_res, cudaMemcpyDeviceToHost); \
    cublasDestroy(handle); \
    cudaStreamDestroy(stream); \
    cudaFree(d_a); \
 \
    Matrix_Transpose(n, m, res); \
    return true; \
} \

#define MATRIX_MULTIPLY_CUDA_CNUM__ALGORITHM(num_type, cublasDgemm, Matrix_Transpose, func_gemm_name) \
{ \
    if (m == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: m is 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (n == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n is 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (l == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: l is 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (a == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix a is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (b == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix b is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (res == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix res is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    /* memory */ \
    uint64_t mm = m, nn = n, ll = l; \
    size_t size_a = mm*ll*sizeof(num_type), size_b = ll*nn*sizeof(num_type), size_res = mm*nn*sizeof(num_type); \
    num_type *d_a, *d_b, *d_res; \
    cudaError_t cuda_malloc_a; \
    cuda_malloc_a = cudaMalloc(&d_a, size_a + size_b + size_res); \
    if (cuda_malloc_a != cudaSuccess){ \
        Madd_cudaMalloc_error(cuda_malloc_a, __func__, size_a + size_b + size_res, "d_a & d_b & d_res"); \
        return false; \
    } \
    d_b = d_a + mm*ll; \
    d_res = d_b + ll*nn; \
    cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice); \
    cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice); \
 \
    /* matrix multiply */ \
    cudaStream_t stream; \
    cudaError_t ret_stream = cudaStreamCreate(&stream); \
    if (ret_stream != cudaSuccess){ \
        cudaFree(d_a); \
        Madd_cudaSetStream_error(ret_stream, __func__); \
        return false; \
    } \
    cublasHandle_t handle; \
    cublasStatus_t ret_handle = cublasCreate(&handle); \
    if (ret_handle != CUBLAS_STATUS_SUCCESS){ \
        cudaFree(d_a); \
        cudaStreamDestroy(stream); \
        Madd_cublasCreate_error(ret_handle, __func__); \
        return false; \
    } \
    cublasSetStream(handle, stream); \
 \
    num_type alpha, beta; \
    alpha.x = 1; \
    alpha.y = beta.x = beta.y = 0; \
    cublasStatus_t ret_gemm = cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, \
                m, n, l, \
                &alpha, \
                d_a, l, \
                d_b, n, \
                &beta, \
                d_res, m); \
    cudaStreamSynchronize(stream); \
    if (ret_gemm != CUBLAS_STATUS_SUCCESS){ \
        gemm_error(ret_gemm, __func__, func_gemm_name); \
        return false; \
    } \
 \
    cudaMemcpy(res, d_res, size_res, cudaMemcpyDeviceToHost); \
    cublasDestroy(handle); \
    cudaStreamDestroy(stream); \
    cudaFree(d_a); \
 \
    Matrix_Transpose(n, m, res); \
    return true; \
} \

extern "C"{

bool Matrix_Multiply_cuda(int m, int n, int l,
                          double *a, double *b, double *res)
MATRIX_MULTIPLY_CUDA__ALGORITHM(double, cublasDgemm, Matrix_Transpose, "cublasDgemm")

bool Matrix_Multiply_cuda_f32(int m, int n, int l,
                              float *a, float *b, float *res)
MATRIX_MULTIPLY_CUDA__ALGORITHM(float, cublasSgemm, Matrix_Transpose_f32, "cublasSgemm")

bool Matrix_Multiply_cuda_c64(int m, int n, int l,
                              Cnum *a, Cnum *b, Cnum *res)
MATRIX_MULTIPLY_CUDA_CNUM__ALGORITHM(cuDoubleComplex, cublasZgemm, Matrix_Transpose_c64, "cublasZgemm")

bool Matrix_Multiply_cuda_c32(int m, int n, int l,
                              Cnum32 *a, Cnum32 *b, Cnum32 *res)
MATRIX_MULTIPLY_CUDA_CNUM__ALGORITHM(cuComplex, cublasCgemm, Matrix_Transpose_c32, "cublasCgemm")

}