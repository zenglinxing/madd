/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/matrix_multiply_64.cu
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

#if __CUDACC_VER_MAJOR__ >=12

static inline void gemm_error(cublasStatus_t ret, const char *func_name, const char *func_gemm_name)
{
    if (ret == CUBLAS_STATUS_SUCCESS) return;
    wchar_t error_info[MADD_ERROR_INFO_LEN];
    switch (ret){
        case CUBLAS_STATUS_NOT_INITIALIZED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUBLAS_STATUS_NOT_INITIALIZED) The library was not initialized.", func_name, func_gemm_name);
            break;
        case CUBLAS_STATUS_INVALID_VALUE:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUBLAS_STATUS_INVALID_VALUE) the parameters m,n,l<0.", func_name, func_gemm_name);
            break;
        case CUBLAS_STATUS_EXECUTION_FAILED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUBLAS_STATUS_EXECUTION_FAILED) the function failed to launch on the GPU.", func_name, func_gemm_name);
            break;
        default:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs returns an error 0x%x that Madd doesn't know.", func_name, func_gemm_name, ret);
    }
    Madd_Error_Add(MADD_ERROR, error_info);
}

#define MATRIX_MULTIPLY_CUDA64__ALGORITHM(num_type, cublasDgemm, Matrix_Transpose, func_gemm_name) \
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

#define MATRIX_MULTIPLY_CUDA64_CNUM__ALGORITHM(num_type, cublasDgemm, Matrix_Transpose, func_gemm_name) \
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

bool Matrix_Multiply_cuda64(int64_t m, int64_t n, int64_t l,
                            double *a, double *b, double *res)
MATRIX_MULTIPLY_CUDA64__ALGORITHM(double, cublasDgemm_64, Matrix_Transpose, "cublasDgemm_64")

bool Matrix_Multiply_cuda64_f32(int64_t m, int64_t n, int64_t l,
                                float *a, float *b, float *res)
MATRIX_MULTIPLY_CUDA64__ALGORITHM(float, cublasSgemm_64, Matrix_Transpose_f32, "cublasSgemm_64")

bool Matrix_Multiply_cuda64_c64(int64_t m, int64_t n, int64_t l,
                                Cnum *a, Cnum *b, Cnum *res)
MATRIX_MULTIPLY_CUDA64_CNUM__ALGORITHM(cuDoubleComplex, cublasZgemm_64, Matrix_Transpose_c64, "cublasZgemm_64")

bool Matrix_Multiply_cuda64_c32(int64_t m, int64_t n, int64_t l,
                                Cnum32 *a, Cnum32 *b, Cnum32 *res)
MATRIX_MULTIPLY_CUDA64_CNUM__ALGORITHM(cuComplex, cublasCgemm_64, Matrix_Transpose_c32, "cublasCgemm_64")

}

#endif /* __CUDACC_VER_MAJOR__ >= 12 */