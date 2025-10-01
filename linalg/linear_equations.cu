/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/linear_equations.cu
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

static void solver_error(cusolverStatus_t ret, const char *func_name, const char *func_exec_name)
{
    if (ret == CUSOLVER_STATUS_SUCCESS) return;
    wchar_t error_info[MADD_ERROR_INFO_LEN];
    switch (ret){
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUSOLVER_STATUS_NOT_INITIALIZED) The cusolver library was not initialized.", func_name, func_exec_name);
            break;
        case CUSOLVER_STATUS_INVALID_VALUE:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUSOLVER_STATUS_INVALID_VALUE) Invalid parameters were passed, for example: n<0; lda<max(1,n); ldb<max(1,n); ldx<max(1,n)", func_name, func_exec_name);
            break;
        case CUSOLVER_STATUS_ARCH_MISMATCH:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUSOLVER_STATUS_ARCH_MISMATCH) The IRS solver supports compute capability 7.0 and above. The lowest precision options CUSOLVER_[CR]_16BF and CUSOLVER_[CR]_TF32 are only available on compute capability 8.0 and above.", func_name, func_exec_name);
            break;
#if __CUDACC_VER_MAJOR__ >= 11
        case CUSOLVER_STATUS_INVALID_WORKSPACE:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUSOLVER_STATUS_INVALID_WORKSPACE) lwork_bytes is smaller than the required workspace. Could happen if the users called cusolverDnIRSXgesv_bufferSize() function, then changed some of the configurations setting such as the lowest precision.", func_name, func_exec_name);
            break;
#endif /* __CUDACC_VER_MAJOR__ >= 11 */
        case CUSOLVER_STATUS_IRS_OUT_OF_RANGE:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUSOLVER_STATUS_IRS_OUT_OF_RANGE) Numerical error related to niters <0, see niters description for more details.", func_name, func_exec_name);
            break;
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs (CUSOLVER_STATUS_INTERNAL_ERROR) An internal error occurred, check the dinfo and the niters arguments for more details.", func_name, func_exec_name);
            break;
        default:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: %hs returns an error 0x%x that Madd doesn't know.", func_name, func_exec_name, ret);
    }
    Madd_Error_Add(MADD_ERROR, error_info);
}

#define LINEAR_EQUATION_CUDA__ALGORITHM(num_type, cusolver_num_type, Matrix_Transpose, \
                                        cusolverDnDDgesv_bufferSize, cusolverDnDDgesv, \
                                        cusolverDnDDgesv_func_name) \
{ \
    if (n == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n is 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (eq == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: eq is NULL.", __func__); \
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
    Matrix_Transpose(n, n, eq); \
    Matrix_Transpose(n, n_vector, vector); \
 \
    size_t size_arr = (uint64_t)n*n*sizeof(double), size_vector = (uint64_t)n*n_vector*sizeof(double), size_ipiv = (uint64_t)n*sizeof(int), size_info = sizeof(int); \
    void *d_temp_space; \
    double *d_arr, *d_vector, *d_result; \
    int *d_ipiv, *d_info; \
    cudaError_t ret_tmp = cudaMalloc(&d_temp_space, size_arr + size_vector + size_ipiv + size_info + size_vector); \
    if (ret_tmp != cudaSuccess){ \
        Madd_cudaMalloc_error(ret_tmp, __func__); \
        return false; \
    } \
    d_arr = (double*)d_temp_space; \
    d_vector = (double*)(d_arr + (uint64_t) n * n); \
    d_ipiv = (int*)(d_vector + (uint64_t) n * n_vector); \
    d_info = (int*)(d_ipiv + n); \
    d_result = (double*)(d_info + 1); \
 \
    cudaMemcpy(d_arr, eq, size_arr, cudaMemcpyHostToDevice); \
    cudaMemcpy(d_vector, vector, size_vector, cudaMemcpyHostToDevice); \
    Matrix_Transpose(n, n, eq); \
 \
    cusolverDnHandle_t handle; \
    cusolverStatus_t status_create = cusolverDnCreate(&handle); \
    if (status_create != CUSOLVER_STATUS_SUCCESS){ \
        cudaFree(d_temp_space); \
        Madd_cusolverDnCreate_error(status_create, __func__); \
        return false; \
    } \
    cudaStream_t stream; \
    cudaError_t ret_stream = cudaStreamCreate(&stream); \
    if (ret_stream != cudaSuccess){ \
        cudaFree(d_temp_space); \
        Madd_cudaSetStream_error(ret_stream, __func__); \
        return false; \
    } \
    cusolverDnSetStream(handle, stream); \
 \
    /* check for working space bytes */ \
    size_t lwork_bytes; \
    cusolverStatus_t status_buffersize = cusolverDnDDgesv_bufferSize(handle, n, n_vector, \
                                                                     (cusolver_num_type*)d_arr, n, \
                                                                     d_ipiv, \
                                                                     (cusolver_num_type*)d_vector, n, \
                                                                     (cusolver_num_type*)d_result, n, \
                                                                     NULL, &lwork_bytes); \
    if (status_buffersize != CUSOLVER_STATUS_SUCCESS){ \
        cudaFree(d_temp_space); \
        cusolverDnDestroy(handle); \
        printf("unable to get buffer size\n"); \
        return false; \
    } \
    /* allocate working space */ \
    double *d_work; \
    cudaError_t ret_work = cudaMalloc(&d_work, lwork_bytes); \
    if (ret_work != cudaSuccess){ \
        cudaFree(d_temp_space); \
        cusolverDnDestroy(handle); \
        Madd_cudaMalloc_error(ret_work, __func__); \
        return false; \
    } \
 \
    int iter; \
    cusolverStatus_t status_solve = cusolverDnDDgesv(handle, n, n_vector, \
                                                     (cusolver_num_type*)d_arr, n, d_ipiv, \
                                                     (cusolver_num_type*)d_vector, n, \
                                                     (cusolver_num_type*)d_result, n, \
                                                     d_work, lwork_bytes, &iter, d_info); \
    cudaStreamSynchronize(stream); \
    if (status_solve != CUSOLVER_STATUS_SUCCESS){ \
        cudaFree(d_temp_space); \
        cudaFree(d_work); \
        cusolverDnDestroy(handle); \
        solver_error(status_solve, __func__, cusolverDnDDgesv_func_name); \
        return false; \
    } \
 \
    cudaMemcpy(vector, d_result, size_vector, cudaMemcpyDeviceToHost); \
    Matrix_Transpose(n_vector, n, vector); \
 \
    int info; \
    cudaMemcpy(&info, d_info, size_info, cudaMemcpyDeviceToHost); \
    if (info != 0){ \
        cudaFree(d_temp_space); \
        cudaFree(d_work); \
        cusolverDnDestroy(handle); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        if (info < 0){ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: the %d-th argument had an illegal value.", __func__, -info); \
        }else{ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: U(%d,%d) is exactly zero.  The factorization has been completed, but the factor U is exactly singular, so the solution could not be computed.", __func__, info, info); \
        } \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    cudaFree(d_temp_space); \
    cudaFree(d_work); \
    cusolverDnDestroy(handle); \
    return true; \
} \

extern "C"{

bool Linear_Equations_cuda(int n, double *eq, int n_vector, double *vector)
LINEAR_EQUATION_CUDA__ALGORITHM(double, double, Matrix_Transpose,
                                cusolverDnDDgesv_bufferSize, cusolverDnDDgesv,
                                "cusolverDnDDgesv")

bool Linear_Equations_cuda_f32(int n, float *eq, int n_vector, float *vector)
LINEAR_EQUATION_CUDA__ALGORITHM(float, float, Matrix_Transpose_f32,
                                cusolverDnSSgesv_bufferSize, cusolverDnSSgesv,
                                "cusolverDnSSgesv")

bool Linear_Equations_cuda_c64(int n, Cnum *eq, int n_vector, Cnum *vector)
LINEAR_EQUATION_CUDA__ALGORITHM(Cnum, cuDoubleComplex, Matrix_Transpose_c64,
                                cusolverDnZZgesv_bufferSize, cusolverDnZZgesv,
                                "cusolverDnZZgesv")

bool Linear_Equations_cuda_c32(int n, Cnum32 *eq, int n_vector, Cnum32 *vector)
LINEAR_EQUATION_CUDA__ALGORITHM(Cnum32, cuComplex, Matrix_Transpose_c32,
                                cusolverDnCCgesv_bufferSize, cusolverDnCCgesv,
                                "cusolverDnCCgesv")

}