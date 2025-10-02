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

#define MATRIX_MULTIPLY_CUDA64__ALGORITHM(num_type, cublasDgemm, Matrix_Transpose) \
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
    cudaError_t cuda_malloc_a, cuda_malloc_b, cuda_malloc_res; \
    cuda_malloc_a = cudaMalloc(&d_a, size_a); \
    if (cuda_malloc_a == cudaErrorMemoryAllocation){ \
        return false; \
    } \
    cuda_malloc_b = cudaMalloc(&d_b, size_b); \
    if (cuda_malloc_b == cudaErrorMemoryAllocation){ \
        cudaFree(d_a); \
        return false; \
    } \
    cuda_malloc_res = cudaMalloc(&d_res, size_res); \
    if (cuda_malloc_res == cudaErrorMemoryAllocation){ \
        cudaFree(d_a); \
        cudaFree(d_b); \
        return false; \
    } \
    cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice); \
    cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice); \
 \
    /* matrix multiply */ \
    cudaStream_t stream; \
    cublasHandle_t handle; \
    cublasCreate(&handle); \
    cudaStreamCreate(&stream); \
    cublasSetStream(handle, stream); \
 \
    num_type alpha = 1, beta = 0; \
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, \
                m, n, l, \
                &alpha, \
                d_a, l, \
                d_b, n, \
                &beta, \
                d_res, m); \
    cudaStreamSynchronize(stream); \
 \
    cudaMemcpy(res, d_res, size_res, cudaMemcpyDeviceToHost); \
    cublasDestroy(handle); \
    cudaFree(d_a); \
    cudaFree(d_b); \
    cudaFree(d_res); \
 \
    Matrix_Transpose(n, m, res); \
    return true; \
} \

#define MATRIX_MULTIPLY_CUDA64_CNUM__ALGORITHM(num_type, cublasDgemm, Matrix_Transpose) \
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
    cudaError_t cuda_malloc_a, cuda_malloc_b, cuda_malloc_res; \
    cuda_malloc_a = cudaMalloc(&d_a, size_a); \
    if (cuda_malloc_a == cudaErrorMemoryAllocation){ \
        return false; \
    } \
    cuda_malloc_b = cudaMalloc(&d_b, size_b); \
    if (cuda_malloc_b == cudaErrorMemoryAllocation){ \
        cudaFree(d_a); \
        return false; \
    } \
    cuda_malloc_res = cudaMalloc(&d_res, size_res); \
    if (cuda_malloc_res == cudaErrorMemoryAllocation){ \
        cudaFree(d_a); \
        cudaFree(d_b); \
        return false; \
    } \
    cudaMemcpy(d_a, a, size_a, cudaMemcpyHostToDevice); \
    cudaMemcpy(d_b, b, size_b, cudaMemcpyHostToDevice); \
 \
    /* matrix multiply */ \
    cudaStream_t stream; \
    cublasHandle_t handle; \
    cublasCreate(&handle); \
    cudaStreamCreate(&stream); \
    cublasSetStream(handle, stream); \
 \
    num_type alpha, beta; \
    alpha.x = 1; \
    alpha.y = beta.x = beta.y = 0; \
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, \
                m, n, l, \
                &alpha, \
                d_a, l, \
                d_b, n, \
                &beta, \
                d_res, m); \
    cudaStreamSynchronize(stream); \
 \
    cudaMemcpy(res, d_res, size_res, cudaMemcpyDeviceToHost); \
    cublasDestroy(handle); \
    cudaFree(d_a); \
    cudaFree(d_b); \
    cudaFree(d_res); \
 \
    Matrix_Transpose(n, m, res); \
    return true; \
} \

extern "C"{

bool Matrix_Multiply_cuda64(int64_t m, int64_t n, int64_t l,
                          double *a, double *b, double *res)
MATRIX_MULTIPLY_CUDA64__ALGORITHM(double, cublasDgemm_64, Matrix_Transpose)

bool Matrix_Multiply_cuda64_f32(int64_t m, int64_t n, int64_t l,
                              float *a, float *b, float *res)
MATRIX_MULTIPLY_CUDA64__ALGORITHM(float, cublasSgemm_64, Matrix_Transpose_f32)

bool Matrix_Multiply_cuda64_c64(int64_t m, int64_t n, int64_t l,
                              Cnum *a, Cnum *b, Cnum *res)
MATRIX_MULTIPLY_CUDA64_CNUM__ALGORITHM(cuDoubleComplex, cublasZgemm_64, Matrix_Transpose_c64)

bool Matrix_Multiply_cuda64_c32(int64_t m, int64_t n, int64_t l,
                              Cnum32 *a, Cnum32 *b, Cnum32 *res)
MATRIX_MULTIPLY_CUDA64_CNUM__ALGORITHM(cuComplex, cublasCgemm_64, Matrix_Transpose_c32)

}

#endif /* __CUDACC_VER_MAJOR__ >= 12 */