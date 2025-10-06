/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/determinant.c
*/
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>
#include<cuda_runtime.h>
#include<cusolverDn.h>
extern "C"{
    #include"linalg.h"
    #include"../basic/basic.h"
}

static inline void cuda_func_error(cusolverStatus_t ret, const char *func_name, const char *func_cuda_name)
{
    if (ret == CUSOLVER_STATUS_SUCCESS) return;
    wchar_t error_info[MADD_ERROR_INFO_LEN];
    switch (ret){
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUSOLVER_STATUS_NOT_INITIALIZED) The library was not initialized.", func_name, func_cuda_name);
            break;
        case CUSOLVER_STATUS_INVALID_VALUE:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUSOLVER_STATUS_INVALID_VALUE) Invalid parameters were passed (m,n<0 or lda<max(1,m)).", func_name, func_cuda_name);
            break;
        case CUSOLVER_STATUS_ARCH_MISMATCH:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUSOLVER_STATUS_ARCH_MISMATCH) The device only supports compute capability 5.0 and above.", func_name, func_cuda_name);
            break;
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUSOLVER_STATUS_NOT_INITIALIZED) An internal operation failed.", func_name, func_cuda_name);
            break;
        default:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs returns an error 0x%x that Madd doesn't know.", func_name, func_cuda_name, ret);
    }
    Madd_Error_Add(MADD_ERROR, error_info);
}

#define DET_CUDA_GPU__ALGORITHM(num_type) \
{ \
    signed char sign = 1; \
    int n1 = n + 1, i; \
    num_type *p = matrix; \
    *res = 1; \
    for (i=0; i<n; i++, p+=n1){ \
        if (ipiv[i] != i+1) sign *= -1; \
        *res *= *p; \
        printf("cuda %d-th: %f\n", i, *p); \
    } \
    *res *= sign; \
} \

static __global__ void Determinant_GPU(int n, double *matrix, int *ipiv, double *res)
DET_CUDA_GPU__ALGORITHM(double)

static __global__ void Determinant_GPU_f32(int n, float *matrix, int *ipiv, float *res)
DET_CUDA_GPU__ALGORITHM(float)

#define DET_CNUM_CUDA_GPU__ALGORITHM(num_type, Cnum_Mul, Cnum_Mul_Real) \
{ \
    signed char sign = 1; \
    int n1 = n + 1, i; \
    num_type *p = matrix; \
    res->real = 1; \
    res->imag = 0; \
    for (i=0; i<n; i++, p+=n1){ \
        if (ipiv[i] != i+1) sign *= -1; \
        *res = Cnum_Mul(*res, *p); \
    } \
    *res = Cnum_Mul_Real(*res, sign); \
} \

static __global__ void Determinant_GPU_c64(int n, Cnum *matrix, int *ipiv, Cnum *res)
DET_CNUM_CUDA_GPU__ALGORITHM(Cnum, Cnum_Mul, Cnum_Mul_Real)

static __global__ void Determinant_GPU_c32(int n, Cnum32 *matrix, int *ipiv, Cnum32 *res)
DET_CNUM_CUDA_GPU__ALGORITHM(Cnum32, Cnum_Mul_c32, Cnum_Mul_Real_c32)

extern "C"{

#define DET_CUDA__ALGORITHM(num_type, cuda_num_type, \
                            cusolverDnDgetrf_bufferSize, cusolverDnDgetrf, \
                            Determinant_GPU, \
                            func_getrf_buffer_name, func_getrf_name) \
{ \
    if (res == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: res is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (matrix == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (n == 0){ \
        *res = 0; \
        return true; \
    } \
 \
    uint64_t nn = (uint64_t)n*n; \
    size_t size_nn = nn*sizeof(num_type), size_ipiv = (uint64_t)n*sizeof(int); \
    num_type *d_matrix, *d_res; \
    int *d_ipiv, *d_info, info; \
    cudaError_t error_matrix = cudaMalloc(&d_matrix, size_nn+size_ipiv+sizeof(int)+sizeof(num_type)); \
    if (error_matrix != cudaSuccess){ \
        Madd_cudaMalloc_error(error_matrix, __func__, size_nn+size_ipiv+sizeof(int)+sizeof(num_type), "d_matrix & d_ipiv & d_info & d_res"); \
        return false; \
    } \
    d_res = (num_type*)(d_matrix+nn); \
    d_ipiv = (int*)(d_res + 1); \
    d_info = (int*)(d_ipiv + n); \
    cudaMemcpy(d_matrix, matrix, size_nn, cudaMemcpyHostToDevice); \
 \
    cudaStream_t stream; \
    cudaError_t ret_stream = cudaStreamCreate(&stream); \
    if (ret_stream != cudaSuccess){ \
        cudaFree(d_matrix); \
        Madd_cudaSetStream_error(ret_stream, __func__); \
        return false; \
    } \
    cusolverDnHandle_t handle; \
    cusolverStatus_t status_create = cusolverDnCreate(&handle); \
    if (status_create != CUSOLVER_STATUS_SUCCESS){ \
        cudaFree(d_matrix); \
        cudaStreamDestroy(stream); \
        Madd_cusolverDnCreate_error(status_create, __func__); \
        return false; \
    } \
    cusolverDnSetStream(handle, stream); \
 \
    int lwork; \
    cusolverStatus_t ret_buffer = cusolverDnDgetrf_bufferSize( \
        handle, n, n, \
        (num_type*)d_matrix, n, \
        &lwork \
    ); \
    cudaStreamSynchronize(stream); \
    if (ret_buffer != CUSOLVER_STATUS_SUCCESS){ \
        cudaFree(d_matrix); \
        cudaStreamDestroy(stream); \
        cusolverDnDestroy(handle); \
        cuda_func_error(ret_buffer, __func__, func_getrf_buffer_name); \
    } \
    cuda_num_type *d_workspace; \
    size_t size_workspace = (uint64_t)lwork*sizeof(cuda_num_type); \
    cudaError_t ret_workspace = cudaMalloc(&d_workspace, size_workspace); \
    if (ret_workspace != cudaSuccess){ \
        cudaFree(d_matrix); \
        cudaStreamDestroy(stream); \
        cusolverDnDestroy(handle); \
        Madd_cudaMalloc_error(ret_workspace, __func__, size_workspace, "d_workspace"); \
        return false; \
    } \
 \
    cusolverStatus_t ret_getrf = cusolverDnDgetrf( \
        handle, n, n, \
        (cuda_num_type*)d_matrix, n, \
        d_workspace, \
        d_ipiv, \
        d_info \
    ); \
    cudaStreamSynchronize(stream); \
    if (ret_getrf != CUSOLVER_STATUS_SUCCESS){ \
        cudaFree(d_matrix); \
        cudaStreamDestroy(stream); \
        cusolverDnDestroy(handle); \
        cudaFree(d_workspace); \
        cuda_func_error(ret_getrf, __func__, func_getrf_name); \
        return false; \
    } \
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost); \
    if (info < 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs: the %d-th parameter is wrong (not counting handle)", __func__, func_getrf_name, -info); \
        cudaFree(d_matrix); \
        cudaStreamDestroy(stream); \
        cusolverDnDestroy(handle); \
        cudaFree(d_workspace); \
        return false; \
    } \
    if (info > 0){ \
        /* this mean matrix[info, info] = 0 */ \
        *res = 0; \
        cudaFree(d_matrix); \
        cudaStreamDestroy(stream); \
        cusolverDnDestroy(handle); \
        cudaFree(d_workspace); \
        return true; \
    } \
 \
    Determinant_GPU<<<1, 1, 0, stream>>>(n, d_matrix, d_ipiv, d_res); \
    cudaStreamSynchronize(stream); \
 \
    cudaMemcpy(res, d_res, sizeof(num_type), cudaMemcpyDeviceToHost); \
 \
    cudaFree(d_matrix); \
    cudaStreamDestroy(stream); \
    cusolverDnDestroy(handle); \
    cudaFree(d_workspace); \
    return true; \
} \

bool Determinant_cuda(int n, double *matrix, double *res)
DET_CUDA__ALGORITHM(double, double,
                    cusolverDnDgetrf_bufferSize, cusolverDnDgetrf,
                    Determinant_GPU,
                    "cusolverDnDgetrf_bufferSize", "cusolverDnDgetrf")

bool Determinant_cuda_f32(int n, float *matrix, float *res)
DET_CUDA__ALGORITHM(float, float,
                    cusolverDnSgetrf_bufferSize, cusolverDnSgetrf,
                    Determinant_GPU_f32,
                    "cusolverDnSgetrf_bufferSize", "cusolverDnSgetrf")

#define DET_CNUM_CUDA__ALGORITHM(num_type, cuda_num_type, \
                                 cusolverDnZgetrf_bufferSize, cusolverDnZgetrf, \
                                 Determinant_GPU, \
                                 func_getrf_buffer_name, func_getrf_name) \
{ \
    if (res == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: res is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (matrix == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (n == 0){ \
        res->real = res->imag = 0; \
        return true; \
    } \
 \
    uint64_t nn = (uint64_t)n*n; \
    size_t size_nn = nn*sizeof(num_type), size_ipiv = (uint64_t)n*sizeof(int); \
    num_type *d_matrix, *d_res; \
    int *d_ipiv, *d_info, info; \
    cudaError_t error_matrix = cudaMalloc(&d_matrix, size_nn+size_ipiv+sizeof(int)+sizeof(num_type)); \
    if (error_matrix != cudaSuccess){ \
        Madd_cudaMalloc_error(error_matrix, __func__, size_nn+size_ipiv+sizeof(int)+sizeof(num_type), "d_matrix & d_ipiv & d_info & d_res"); \
        return false; \
    } \
    d_res = (num_type*)(d_matrix+nn); \
    d_ipiv = (int*)(d_res + 1); \
    d_info = (int*)(d_ipiv + n); \
    cudaMemcpy(d_matrix, matrix, size_nn, cudaMemcpyHostToDevice); \
 \
    cudaStream_t stream; \
    cudaError_t ret_stream = cudaStreamCreate(&stream); \
    if (ret_stream != cudaSuccess){ \
        cudaFree(d_matrix); \
        Madd_cudaSetStream_error(ret_stream, __func__); \
        return false; \
    } \
    cusolverDnHandle_t handle; \
    cusolverStatus_t status_create = cusolverDnCreate(&handle); \
    if (status_create != CUSOLVER_STATUS_SUCCESS){ \
        cudaFree(d_matrix); \
        cudaStreamDestroy(stream); \
        Madd_cusolverDnCreate_error(status_create, __func__); \
        return false; \
    } \
    cusolverDnSetStream(handle, stream); \
 \
    int lwork; \
    cusolverStatus_t ret_buffer = cusolverDnZgetrf_bufferSize( \
        handle, n, n, \
        (cuda_num_type*)d_matrix, n, \
        &lwork \
    ); \
    cudaStreamSynchronize(stream); \
    if (ret_buffer != CUSOLVER_STATUS_SUCCESS){ \
        cudaFree(d_matrix); \
        cudaStreamDestroy(stream); \
        cusolverDnDestroy(handle); \
        cuda_func_error(ret_buffer, __func__, func_getrf_buffer_name); \
    } \
    cuda_num_type *d_workspace; \
    size_t size_workspace = (uint64_t)lwork*sizeof(cuda_num_type); \
    cudaError_t ret_workspace = cudaMalloc(&d_workspace, size_workspace); \
    if (ret_workspace != cudaSuccess){ \
        cudaFree(d_matrix); \
        cudaStreamDestroy(stream); \
        cusolverDnDestroy(handle); \
        Madd_cudaMalloc_error(ret_workspace, __func__, size_workspace, "d_workspace"); \
        return false; \
    } \
 \
    cusolverStatus_t ret_getrf = cusolverDnZgetrf( \
        handle, n, n, \
        (cuda_num_type*)d_matrix, n, \
        d_workspace, \
        d_ipiv, \
        d_info \
    ); \
    cudaStreamSynchronize(stream); \
    if (ret_getrf != CUSOLVER_STATUS_SUCCESS){ \
        cudaFree(d_matrix); \
        cudaStreamDestroy(stream); \
        cusolverDnDestroy(handle); \
        cudaFree(d_workspace); \
        cuda_func_error(ret_getrf, __func__, func_getrf_name); \
        return false; \
    } \
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost); \
    if (info < 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs: the %d-th parameter is wrong (not counting handle)", __func__, func_getrf_name, -info); \
        cudaFree(d_matrix); \
        cudaStreamDestroy(stream); \
        cusolverDnDestroy(handle); \
        cudaFree(d_workspace); \
        return false; \
    } \
    if (info > 0){ \
        /* this mean matrix[info, info] = 0 */ \
        res->real = res->imag = 0; \
        cudaFree(d_matrix); \
        cudaStreamDestroy(stream); \
        cusolverDnDestroy(handle); \
        cudaFree(d_workspace); \
        return true; \
    } \
 \
    Determinant_GPU<<<1, 1, 0, stream>>>(n, d_matrix, d_ipiv, d_res); \
    cudaStreamSynchronize(stream); \
 \
    cudaMemcpy(res, d_res, sizeof(num_type), cudaMemcpyDeviceToHost); \
 \
    cudaFree(d_matrix); \
    cudaStreamDestroy(stream); \
    cusolverDnDestroy(handle); \
    cudaFree(d_workspace); \
    return true; \
} \

bool Determinant_cuda_c64(int n, Cnum *matrix, Cnum *res)
DET_CNUM_CUDA__ALGORITHM(Cnum, cuDoubleComplex,
                         cusolverDnZgetrf_bufferSize, cusolverDnZgetrf,
                         Determinant_GPU_c64,
                         "cusolverDnZgetrf_bufferSize", "cusolverDnZgetrf")

bool Determinant_cuda_c32(int n, Cnum32 *matrix, Cnum32 *res)
DET_CNUM_CUDA__ALGORITHM(Cnum32, cuComplex,
                         cusolverDnCgetrf_bufferSize, cusolverDnCgetrf,
                         Determinant_GPU_c32,
                         "cusolverDnCgetrf_bufferSize", "cusolverDnCgetrf")

}