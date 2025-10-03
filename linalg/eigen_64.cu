/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/eigen_64.cu
check
https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnxgeev
*/
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>
#include<cuda_runtime.h>
#include<cusolverDn.h>

extern "C"{
    #include"linalg.h"
    //#include"linalg.cuh"
    #include"../basic/basic.h"
}

/* Xgeev is introduced since CUDA 12.6 */
#if __CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 6)

static inline void buffer_func_error(cusolverStatus_t ret, const char *func_name)
{
    if (ret == CUSOLVER_STATUS_SUCCESS) return;
    wchar_t error_info[MADD_ERROR_INFO_LEN];
    switch (ret){
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnXgeev_bufferSize (CUSOLVER_STATUS_NOT_INITIALIZED) The library was not initialized.", func_name);
            break;
        case CUSOLVER_STATUS_INVALID_VALUE:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnXgeev_bufferSize (CUSOLVER_STATUS_INVALID_VALUE) Invalid parameters were passed (jobvl is not CUSOLVER_EIG_MODE_NOVECTOR or CUSOLVER_EIG_MODE_VECTOR, or jobvr is not CUSOLVER_EIG_MODE_NOVECTOR or CUSOLVER_EIG_MODE_VECTOR, n<0, or lda < max(1,n), or ldvl < n if jobvl is CUSOLVER_EIG_MODE_VECTOR, or ldvr < n if jobvr is CUSOLVER_EIG_MODE_VECTOR).", func_name);
            break;
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnXgeev_bufferSize (CUSOLVER_STATUS_NOT_INITIALIZED) An internal operation failed.", func_name);
            break;
        default:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnXgeev_bufferSize returns an error 0x%x that Madd doesn't know.", func_name, ret);
    }
    Madd_Error_Add(MADD_ERROR, error_info);
}

static inline void Xgeev_error(cusolverStatus_t ret, const char *func_name)
{
    if (ret == CUSOLVER_STATUS_SUCCESS) return;
    wchar_t error_info[MADD_ERROR_INFO_LEN];
    switch (ret){
        case CUSOLVER_STATUS_NOT_INITIALIZED:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnXgeev (CUSOLVER_STATUS_NOT_INITIALIZED) The library was not initialized.", func_name);
            break;
        case CUSOLVER_STATUS_INVALID_VALUE:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnXgeev (CUSOLVER_STATUS_INVALID_VALUE) Invalid parameters were passed (jobvl is not CUSOLVER_EIG_MODE_NOVECTOR or CUSOLVER_EIG_MODE_VECTOR, or jobvr is not CUSOLVER_EIG_MODE_NOVECTOR or CUSOLVER_EIG_MODE_VECTOR, n<0, or lda < max(1,n), or ldvl < n if jobvl is CUSOLVER_EIG_MODE_VECTOR, or ldvr < n if jobvr is CUSOLVER_EIG_MODE_VECTOR).", func_name);
            break;
        case CUSOLVER_STATUS_INTERNAL_ERROR:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnXgeev (CUSOLVER_STATUS_NOT_INITIALIZED) An internal operation failed.", func_name);
            break;
        default:
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnXgeev returns an error 0x%x that Madd doesn't know.", func_name, ret);
    }
    Madd_Error_Add(MADD_ERROR, error_info);
}

#define EIGEN_CUDA64__ALGORITHM(Cnum, real_type, Matrix_Transpose, CUDA_R_64F, CUDA_C_64F) \
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
    if (eigenvalue == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: eigenvalue is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (flag_left && eigenvector_left == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: flag_left is true, but eigenvector_left is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (flag_right && eigenvector_right == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: flag_right is true, but eigenvector_right is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    if (flag_left){ \
        wchar_t warning_info[MADD_ERROR_INFO_LEN]; \
        swprintf(warning_info, MADD_ERROR_INFO_LEN, L"%hs: according to NVIDIA doc https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnxgeev, CUDA at present may not support the computation of left eigenvectors. You may encounter CUDA error soon.", __func__); \
        Madd_Error_Add(MADD_WARNING, warning_info); \
    } \
 \
    uint64_t nn = (uint64_t)n * n; \
    size_t size_matrix = nn * sizeof(real_type), size_nn = nn * sizeof(Cnum), size_n = (uint64_t)n * sizeof(Cnum); \
    size_t size_alloc = size_matrix + size_n; \
    if (flag_left) size_alloc += size_nn; \
    if (flag_right) size_alloc += size_nn; \
    real_type *d_matrix; \
    Cnum *d_eigenvalue, *d_eigenvector_left=NULL, *d_eigenvector_right=NULL; \
    cudaError_t error_matrix = cudaMalloc(&d_matrix, size_alloc); \
    if (error_matrix != cudaSuccess){ \
        Madd_cudaMalloc_error(error_matrix, __func__, size_alloc, "d_matrix & d_eigenvalue, maybe d_eigenvector_left & d_eigenvector_right"); \
        return false; \
    } \
    d_eigenvalue = (Cnum*)(((real_type*)d_matrix + nn)); \
    /*printf("d_matrix: %p\n", d_matrix);*/ \
    /*printf("d_eigenvalue: %p\n", d_eigenvalue);*/ \
    if (flag_left){ \
        d_eigenvector_left = d_eigenvalue + n; \
        if (flag_right){ \
            d_eigenvector_right = d_eigenvector_left + nn; \
        } \
    }else if (flag_right){ \
        d_eigenvector_right = d_eigenvalue + n; \
    } \
    Matrix_Transpose(n, n, matrix); \
    cudaMemcpy(d_matrix, matrix, size_matrix, cudaMemcpyHostToDevice); \
    Matrix_Transpose(n, n, matrix); \
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
    int64_t lda = n, ldvl = n, ldvr = n; \
    if (ret_create_params != CUSOLVER_STATUS_SUCCESS){ \
        cudaFree(d_matrix); \
        cusolverDnDestroy(handle); \
        Madd_cusolverDnCreateParams_error(ret_create_params, __func__); \
        return false; \
    } \
 \
    /* buffer */ \
    cusolverEigMode_t  jobvl = (flag_left)  ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR; \
    cusolverEigMode_t  jobvr = (flag_right) ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR; \
    size_t workspaceInBytesOnDevice = 0, workspaceInBytesOnHost = 0; \
    /* this func gets buffer sizes */ \
    cusolverStatus_t ret_buffer = cusolverDnXgeev_bufferSize( \
        handle, params, jobvl, jobvr, \
        n, CUDA_R_64F, d_matrix, lda, \
        CUDA_C_64F, d_eigenvalue, \
        CUDA_R_64F, d_eigenvector_left, ldvl, \
        CUDA_R_64F, d_eigenvector_right, ldvr, \
        CUDA_R_64F, \
        &workspaceInBytesOnDevice, &workspaceInBytesOnHost \
    ); \
    cudaStreamSynchronize(stream); \
    if (ret_buffer != CUSOLVER_STATUS_SUCCESS){ \
        cudaFree(d_matrix); \
        cusolverDnDestroy(handle); \
        cusolverDnDestroyParams(params); \
        buffer_func_error(ret_buffer, __func__); \
        return false; \
    } \
    void *bufferOnDevice, *bufferOnHost; \
    int info = 0, *d_info; \
    cudaError_t error_buffer_device = cudaMalloc(&bufferOnDevice, workspaceInBytesOnDevice + sizeof(int)); \
    if (error_buffer_device != cudaSuccess){ \
        cudaFree(d_matrix); \
        cusolverDnDestroy(handle); \
        cusolverDnDestroyParams(params); \
        Madd_cudaMalloc_error(error_buffer_device, __func__, workspaceInBytesOnDevice + sizeof(int), "bufferOnDevice & d_info"); \
        return false; \
    } \
    bufferOnHost = malloc(workspaceInBytesOnHost); \
    if (bufferOnHost == NULL){ \
        cudaFree(d_matrix); \
        cudaFree(bufferOnDevice); \
        free(bufferOnHost); \
        cusolverDnDestroy(handle); \
        cusolverDnDestroyParams(params); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for bufferOnHost.", __func__, workspaceInBytesOnHost); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    d_info = (int*)((unsigned char*)bufferOnDevice + workspaceInBytesOnDevice); \
 \
    /* geev */ \
    cusolverStatus_t ret_geev = cusolverDnXgeev( \
        handle, params, jobvl, jobvr, \
        n, CUDA_R_64F, d_matrix, lda, \
        CUDA_C_64F, d_eigenvalue, \
        CUDA_R_64F, d_eigenvector_left, ldvl, \
        CUDA_R_64F, d_eigenvector_right, ldvr, \
        CUDA_R_64F, \
        bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, \
        d_info \
    ); \
    cudaStreamSynchronize(stream); \
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost); \
    /* here free some mem first */ \
    cudaFree(bufferOnDevice); \
    free(bufferOnHost); \
    cusolverDnDestroy(handle); \
    cusolverDnDestroyParams(params); \
    /* check if Xgeev succeeded */ \
    if (ret_geev != CUSOLVER_STATUS_SUCCESS){ \
        cudaFree(d_matrix); \
        Xgeev_error(ret_geev, __func__); \
        return false; \
    } \
    if (info != 0){ \
        cudaFree(d_matrix); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        if (info < 0){ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnXgeev (info < 0) the %d-th parameter is wrong (not counting handle).", __func__, -info); \
        }else{ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnXgeev (info > 0) the QR algorithm failed to compute all the eigenvalues and no eigenvectors have been computed; elements %d+1:n of W contain eigenvalues which have converged.", __func__, info); \
        } \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    /* temporarily save the results (vl & vr) from CUDA */ \
    size_t size_vl_vr = 0; \
    if (flag_left) size_vl_vr += size_nn; \
    if (flag_right) size_vl_vr += size_nn; \
    real_type *vl = NULL, *vr = NULL; \
    if (size_vl_vr){ \
        vl = (real_type*)malloc(size_vl_vr); \
        vr = (flag_left) ? vl + nn : vl; \
        if (vl == NULL){ \
            cudaFree(d_matrix); \
            wchar_t error_info[MADD_ERROR_INFO_LEN]; \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for vl & vr.", __func__, size_nn); \
            Madd_Error_Add(MADD_ERROR, error_info); \
            return false; \
        } \
    } \
    cudaMemcpy(eigenvalue, d_eigenvalue, size_n, cudaMemcpyDeviceToHost); \
    if (flag_left){ \
        cudaMemcpy(vl, d_eigenvector_left, size_nn, cudaMemcpyDeviceToHost); \
    } \
    if (flag_right){ \
        cudaMemcpy(vr, d_eigenvector_right, size_nn, cudaMemcpyDeviceToHost); \
    } \
 \
    if (flag_left || flag_right){ \
        uint64_t i, j; \
        for (i=0; i<n; i++){ \
            Cnum *le = eigenvector_left + i, *re = eigenvector_right + i; \
            real_type *lv = vl + i*n, *rv = vr + i*n; \
            if (eigenvalue[i].imag == 0){ \
                for (j=0; j<n; j++,lv++,rv++,le+=n,re+=n){ \
                    if (flag_left){ \
                        le->real = *lv; \
                        le->imag = 0; \
                    } \
                    if (flag_right){ \
                        re->real = *rv; \
                        re->imag = 0; \
                    } \
                } \
            }else{ \
                for (j=0; j<n; j++,lv++,rv++,le+=n,re+=n){ \
                    if (flag_left){ \
                        le->real = le[1].real = *lv; \
                        le->imag = lv[n]; \
                        le[1].imag = -lv[n]; \
                    } \
                    if (flag_right){ \
                        re->real = re[1].real = *rv; \
                        re->imag = rv[n]; \
                        re[1].imag = -rv[n]; \
                    } \
                } \
                i ++; \
                continue; \
            } \
        } \
    } \
 \
    /* here the second time to free */ \
    cudaFree(d_matrix); \
    free(vl); \
    return true; \
} \

extern "C"{

bool Eigen_cuda64(int64_t n, double *matrix,
                  Cnum *eigenvalue,
                  bool flag_left, Cnum *eigenvector_left,
                  bool flag_right, Cnum *eigenvector_right)
EIGEN_CUDA64__ALGORITHM(Cnum, double, Matrix_Transpose, CUDA_R_64F, CUDA_C_64F)

bool Eigen_cuda64_f32(int64_t n, float *matrix,
                      Cnum32 *eigenvalue,
                      bool flag_left, Cnum32 *eigenvector_left,
                      bool flag_right, Cnum32 *eigenvector_right)
EIGEN_CUDA64__ALGORITHM(Cnum32, float, Matrix_Transpose_f32, CUDA_R_32F, CUDA_C_32F)

#define EIGEN_CUDA64_CNUM__ALGORITHM(Cnum, Matrix_Transpose_c64, CUDA_C_64F) \
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
    if (eigenvalue == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: eigenvalue is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (flag_left && eigenvector_left == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: flag_left is true, but eigenvector_left is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (flag_right && eigenvector_right == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: flag_right is true, but eigenvector_right is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    if (flag_left){ \
        wchar_t warning_info[MADD_ERROR_INFO_LEN]; \
        swprintf(warning_info, MADD_ERROR_INFO_LEN, L"%hs: according to NVIDIA doc https://docs.nvidia.com/cuda/cusolver/index.html#cusolverdnxgeev, CUDA at present may not support the computation of left eigenvectors. You may encounter CUDA error soon.", __func__); \
        Madd_Error_Add(MADD_WARNING, warning_info); \
    } \
 \
    uint64_t nn = (uint64_t)n * n; \
    size_t size_nn = nn * sizeof(Cnum), size_n = (uint64_t)n * sizeof(Cnum); \
    size_t size_alloc = size_nn + size_n; \
    if (flag_left) size_alloc += size_nn; \
    if (flag_right) size_alloc += size_nn; \
    Cnum *d_matrix; \
    Cnum *d_eigenvalue, *d_eigenvector_left=NULL, *d_eigenvector_right=NULL; \
    cudaError_t error_matrix = cudaMalloc(&d_matrix, size_alloc); \
    if (error_matrix != cudaSuccess){ \
        Madd_cudaMalloc_error(error_matrix, __func__, size_alloc, "d_matrix & d_eigenvalue, maybe d_eigenvector_left & d_eigenvector_right"); \
        return false; \
    } \
    d_eigenvalue = (Cnum*)(d_matrix + nn); \
    if (flag_left){ \
        d_eigenvector_left = d_eigenvalue + n; \
        if (flag_right){ \
            d_eigenvector_right = d_eigenvector_left + nn; \
        } \
    }else if (flag_right){ \
        d_eigenvector_right = d_eigenvalue + n; \
    } \
    Matrix_Transpose_c64(n, n, matrix); \
    cudaMemcpy(d_matrix, matrix, size_nn, cudaMemcpyHostToDevice); \
    Matrix_Transpose_c64(n, n, matrix); \
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
    int64_t lda = n, ldvl = n, ldvr = n; \
    if (ret_create_params != CUSOLVER_STATUS_SUCCESS){ \
        cudaFree(d_matrix); \
        cusolverDnDestroy(handle); \
        Madd_cusolverDnCreateParams_error(ret_create_params, __func__); \
        return false; \
    } \
 \
    /* buffer */ \
    cusolverEigMode_t  jobvl = (flag_left)  ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR; \
    cusolverEigMode_t  jobvr = (flag_right) ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR; \
    size_t workspaceInBytesOnDevice = 0, workspaceInBytesOnHost = 0; \
    /* this func gets buffer sizes */ \
    cusolverStatus_t ret_buffer = cusolverDnXgeev_bufferSize( \
        handle, params, jobvl, jobvr, \
        n, CUDA_C_64F, d_matrix, lda, \
        CUDA_C_64F, d_eigenvalue, \
        CUDA_C_64F, d_eigenvector_left, ldvl, \
        CUDA_C_64F, d_eigenvector_right, ldvr, \
        CUDA_C_64F, \
        &workspaceInBytesOnDevice, &workspaceInBytesOnHost \
    ); \
    cudaStreamSynchronize(stream); \
    if (ret_buffer != CUSOLVER_STATUS_SUCCESS){ \
        cudaFree(d_matrix); \
        cusolverDnDestroy(handle); \
        cusolverDnDestroyParams(params); \
        buffer_func_error(ret_buffer, __func__); \
        return false; \
    } \
    void *bufferOnDevice, *bufferOnHost; \
    int info = 0, *d_info; \
    cudaError_t error_buffer_device = cudaMalloc(&bufferOnDevice, workspaceInBytesOnDevice + sizeof(int)); \
    if (error_buffer_device != cudaSuccess){ \
        cudaFree(d_matrix); \
        cusolverDnDestroy(handle); \
        cusolverDnDestroyParams(params); \
        Madd_cudaMalloc_error(error_buffer_device, __func__, workspaceInBytesOnDevice + sizeof(int), "bufferOnDevice & d_info"); \
        return false; \
    } \
    bufferOnHost = malloc(workspaceInBytesOnHost); \
    if (bufferOnHost == NULL){ \
        cudaFree(d_matrix); \
        cudaFree(bufferOnDevice); \
        cusolverDnDestroy(handle); \
        cusolverDnDestroyParams(params); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for bufferOnHost.", __func__, workspaceInBytesOnHost); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    d_info = (int*)((unsigned char*)bufferOnDevice + workspaceInBytesOnDevice); \
 \
    /* geev */ \
    cusolverStatus_t ret_geev = cusolverDnXgeev( \
        handle, params, jobvl, jobvr, \
        n, CUDA_C_64F, d_matrix, lda, \
        CUDA_C_64F, d_eigenvalue, \
        CUDA_C_64F, d_eigenvector_left, ldvl, \
        CUDA_C_64F, d_eigenvector_right, ldvr, \
        CUDA_C_64F, \
        bufferOnDevice, workspaceInBytesOnDevice, bufferOnHost, workspaceInBytesOnHost, \
        d_info \
    ); \
    cudaStreamSynchronize(stream); \
    cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost); \
    /* here free some mem first */ \
    cudaFree(bufferOnDevice); \
    free(bufferOnHost); \
    cusolverDnDestroy(handle); \
    cusolverDnDestroyParams(params); \
    /* check if Xgeev succeeded */ \
    if (ret_geev != CUSOLVER_STATUS_SUCCESS){ \
        cudaFree(d_matrix); \
        Xgeev_error(ret_geev, __func__); \
        return false; \
    } \
    if (info != 0){ \
        cudaFree(d_matrix); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        if (info < 0){ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnXgeev (info < 0) the %d-th parameter is wrong (not counting handle).", __func__, -info); \
        }else{ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: cusolverDnXgeev (info > 0) the QR algorithm failed to compute all the eigenvalues and no eigenvectors have been computed; elements %d+1:n of W contain eigenvalues which have converged.", __func__, info); \
        } \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    cudaMemcpy(eigenvalue, d_eigenvalue, size_n, cudaMemcpyDeviceToHost); \
    if (flag_left){ \
        cudaMemcpy(eigenvector_left, d_eigenvector_left, size_nn, cudaMemcpyDeviceToHost); \
        Matrix_Transpose_c64(n, n, eigenvector_left); \
    } \
    if (flag_right){ \
        cudaMemcpy(eigenvector_right, d_eigenvector_right, size_nn, cudaMemcpyDeviceToHost); \
        Matrix_Transpose_c64(n, n, eigenvector_right); \
    } \
 \
    /* here the second time to free */ \
    cudaFree(d_matrix); \
    return true; \
} \

bool Eigen_cuda64_c64(int64_t n, Cnum *matrix,
                      Cnum *eigenvalue,
                      bool flag_left, Cnum *eigenvector_left,
                      bool flag_right, Cnum *eigenvector_right)
EIGEN_CUDA64_CNUM__ALGORITHM(Cnum, Matrix_Transpose_c64, CUDA_C_64F)

bool Eigen_cuda64_c32(int64_t n, Cnum32 *matrix,
                      Cnum32 *eigenvalue,
                      bool flag_left, Cnum32 *eigenvector_left,
                      bool flag_right, Cnum32 *eigenvector_right)
EIGEN_CUDA64_CNUM__ALGORITHM(Cnum32, Matrix_Transpose_c32, CUDA_C_32F)

}

#endif /* __CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 6) */