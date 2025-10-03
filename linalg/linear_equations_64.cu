/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/linear_equations_64.cu
check
https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnxgetrf
https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnxgetrs
*/
#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>
#include<cuda_runtime.h>
#include<cusolverDn.h>
extern "C"{
    #include"../basic/basic.h"
    #include"linalg.h"
}

// cUDA version should be >= 11.1
#if __CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 1)

static inline void Madd_cusolverDnXgetrf_error(cusolverStatus_t ret, const char *func_name)
{
    if (ret == CUSOLVER_STATUS_SUCCESS) return;
    wchar_t error_info[MADD_ERROR_INFO_LEN];
    switch (ret){
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnXgetrf_bufferSize (CUSOLVER_STATUS_NOT_INITIALIZED) The library was not initialized.", func_name);
            break;
        case CUSOLVER_STATUS_INVALID_VALUE:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnXgetrf_bufferSize (CUSOLVER_STATUS_INVALID_VALUE) Invalid parameters were passed (m,n<0 or lda<max(1,m)).", func_name);
            break;
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnXgetrf_bufferSize (CUSOLVER_STATUS_INTERNAL_ERROR) An internal operation failed.", func_name);
            break;
        default:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnXgetrf_bufferSize returns an error 0x%x that Madd doesn't know.", func_name, ret);
    }
    Madd_Error_Add(MADD_ERROR, error_info);
}

static inline void Madd_cusolverDnXgetrs_error(cusolverStatus_t ret, const char *func_name)
{
    if (ret == CUSOLVER_STATUS_SUCCESS) return;
    wchar_t error_info[MADD_ERROR_INFO_LEN];
    switch (ret){
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnXgetrf_bufferSize (CUSOLVER_STATUS_NOT_INITIALIZED) The library was not initialized.", func_name);
            break;
        case CUSOLVER_STATUS_INVALID_VALUE:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnXgetrf_bufferSize (CUSOLVER_STATUS_INVALID_VALUE) Invalid parameters were passed (n<0 or lda<max(1,n) or ldb<max(1,n)).", func_name);
            break;
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnXgetrf_bufferSize (CUSOLVER_STATUS_INTERNAL_ERROR) An internal operation failed.", func_name);
            break;
        default:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnXgetrf_bufferSize returns an error 0x%x that Madd doesn't know.", func_name, ret);
    }
    Madd_Error_Add(MADD_ERROR, error_info);
}

#define LINEAR_EQUATIONS_CUDA64__ALGORITHM(num_type, CUDA_R_64F, Matrix_Transpose) \
{ \
    if (n == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n is 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (matrix == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (n_vector == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n_vector is 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (vector == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: vector is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    size_t size_nn = (uint64_t)n*n*sizeof(num_type), size_nvec = (uint64_t)n*n_vector*sizeof(num_type); \
    num_type *d_matrix, *d_vector; \
    cudaError_t error_matrix = cudaMalloc(&d_matrix, size_nn + size_nvec); \
    if (error_matrix != cudaSuccess){ \
        Madd_cudaMalloc_error(error_matrix, __func__); \
        return false; \
    } \
    d_vector = d_matrix + (uint64_t)n * n; \
 \
    Matrix_Transpose(n, n, matrix); \
    cudaMemcpy(d_matrix, matrix, size_nn, cudaMemcpyHostToDevice); \
    Matrix_Transpose(n, n, matrix); \
    Matrix_Transpose(n, n_vector, vector); \
    cudaMemcpy(d_vector, vector, size_nvec, cudaMemcpyHostToDevice); \
    /*Matrix_Transpose(n_vector, n, vector);*/ \
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
        Madd_cusolverDnCreate_error(status_create, __func__); \
        return false; \
    } \
    cusolverDnSetStream(handle, stream); \
 \
    cusolverDnParams_t params = NULL; \
    cusolverStatus_t ret_create_params = cusolverDnCreateParams(&params); \
    if (ret_create_params != CUSOLVER_STATUS_SUCCESS){ \
        cudaFree(d_matrix); \
        cusolverDnDestroy(handle); \
        Madd_cusolverDnCreateParams_error(ret_create_params, __func__); \
        return false; \
    } \
    cusolverStatus_t ret_set_params = cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0); \
    if (ret_set_params != CUSOLVER_STATUS_SUCCESS){ \
        cudaFree(d_matrix); \
        cusolverDnDestroy(handle); \
        cusolverDnDestroyParams(params); \
        Madd_cusolverDnSetAdvOptions_error(ret_set_params, __func__); \
        return false; \
    } \
 \
    /* get buffer sizes */ \
    size_t workspaceInBytesOnDevice, workspaceInBytesOnHost; \
    cusolverStatus_t ret_trf_buffer = cusolverDnXgetrf_bufferSize( \
        handle, params, n, n, CUDA_R_64F, \
        d_matrix, n, CUDA_R_64F, \
        &workspaceInBytesOnDevice, &workspaceInBytesOnHost \
    ); \
    cudaStreamSynchronize(stream); \
    if (ret_trf_buffer != CUSOLVER_STATUS_SUCCESS){ \
        cudaFree(d_matrix); \
        cusolverDnDestroy(handle); \
        cusolverDnDestroyParams(params); \
        Madd_cusolverDnXgetrf_error(ret_trf_buffer, __func__); \
        return false; \
    } \
    void *bufferOnDevice = NULL, *bufferOnHost = NULL; \
    cudaError_t error_dev_buffer = cudaMalloc(&bufferOnDevice, workspaceInBytesOnDevice); \
    if (error_dev_buffer != cudaSuccess){ \
        cudaFree(d_matrix); \
        cusolverDnDestroy(handle); \
        cusolverDnDestroyParams(params); \
        Madd_cudaMalloc_error(error_dev_buffer, __func__); \
        return false; \
    } \
    bufferOnHost = malloc(workspaceInBytesOnHost); \
    if (workspaceInBytesOnHost && bufferOnHost){ \
        cudaFree(d_matrix); \
        cusolverDnDestroy(handle); \
        cusolverDnDestroyParams(params); \
        cudaFree(bufferOnDevice); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for bufferOnHost.", __func__, workspaceInBytesOnHost); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    /* LU */ \
    int64_t *d_ipiv; \
    int info, *d_info; \
    cudaError_t error_ipiv = cudaMalloc(&d_ipiv, (uint64_t)n*sizeof(int64_t) + sizeof(int)); \
    if (error_ipiv != cudaSuccess){ \
        cudaFree(d_matrix); \
        cusolverDnDestroy(handle); \
        cusolverDnDestroyParams(params); \
        cudaFree(bufferOnDevice); \
        free(bufferOnHost); \
        Madd_cudaMalloc_error(error_ipiv, __func__); \
        return false; \
    } \
    d_info = (int*)(d_ipiv + n); \
    cusolverStatus_t ret_getrf = cusolverDnXgetrf( \
        handle, params, n, n, CUDA_R_64F, \
        d_matrix, n, d_ipiv, CUDA_R_64F, \
        bufferOnDevice, workspaceInBytesOnDevice, \
        bufferOnHost, workspaceInBytesOnHost, \
        d_info \
    ); \
    cudaStreamSynchronize(stream); \
    if (ret_getrf != CUSOLVER_STATUS_SUCCESS){ \
        cudaFree(d_matrix); \
        cusolverDnDestroy(handle); \
        cusolverDnDestroyParams(params); \
        cudaFree(bufferOnDevice); \
        free(bufferOnHost); \
        cudaFree(d_ipiv); \
        Madd_cusolverDnXgetrf_error(ret_getrf, __func__); \
        return false; \
    } \
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost); \
 \
    /* linear equation */ \
    cusolverStatus_t ret_getrs = cusolverDnXgetrs( \
        handle, params, CUBLAS_OP_N, n, n_vector, CUDA_R_64F, \
        d_matrix, n, d_ipiv, CUDA_R_64F, \
        d_vector, n, d_info \
    ); \
    cudaStreamSynchronize(stream); \
    if (ret_getrs != CUSOLVER_STATUS_SUCCESS){ \
        cudaFree(d_matrix); \
        cusolverDnDestroy(handle); \
        cusolverDnDestroyParams(params); \
        cudaFree(bufferOnDevice); \
        free(bufferOnHost); \
        cudaFree(d_ipiv); \
        Madd_cusolverDnXgetrs_error(ret_getrs, __func__); \
        return false; \
    } \
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost); \
    if (info != 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        if (info < 0){ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: the %d-th parameter is wrong (not counting handle).", __func__, -info); \
        }else{ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: up to now, Madd developer never saw the NVIDIA doc explains why info (0x%x) from cusolverDnXgetrs is greater than 0.", __func__, info); \
        } \
        Madd_Error_Add(MADD_ERROR, error_info); \
    } \
 \
    cudaMemcpy(vector, d_vector, size_nvec, cudaMemcpyDeviceToHost); \
    Matrix_Transpose(n_vector, n, vector); \
 \
    cusolverDnDestroyParams(params); \
    cudaFree(d_matrix); \
    cudaFree(bufferOnDevice); \
    cudaFree(d_ipiv); \
    free(bufferOnHost); \
    cusolverDnDestroy(handle); \
    return true; \
} \

extern "C"{

bool Linear_Equations_cuda64(int64_t n, double *matrix, int64_t n_vector, double *vector)
LINEAR_EQUATIONS_CUDA64__ALGORITHM(double, CUDA_R_64F, Matrix_Transpose)

bool Linear_Equations_cuda64_f32(int64_t n, float *matrix, int64_t n_vector, float *vector)
LINEAR_EQUATIONS_CUDA64__ALGORITHM(float, CUDA_R_32F, Matrix_Transpose_f32)

bool Linear_Equations_cuda64_c64(int64_t n, Cnum *matrix, int64_t n_vector, Cnum *vector)
LINEAR_EQUATIONS_CUDA64__ALGORITHM(Cnum, CUDA_C_64F, Matrix_Transpose_c64)

bool Linear_Equations_cuda64_c32(int64_t n, Cnum32 *matrix, int64_t n_vector, Cnum32 *vector)
LINEAR_EQUATIONS_CUDA64__ALGORITHM(Cnum32, CUDA_C_32F, Matrix_Transpose_c32)

}

#endif /* __CUDACC_VER_MAJOR__ > 11 || (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ >= 1) */