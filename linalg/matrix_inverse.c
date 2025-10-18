/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/matrix_inverse.c
*/
#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<stdbool.h>
#define HAVE_LAPACK_CONFIG_H
#include <lapacke.h>
#include <cblas.h>
#include"linalg.h"
#include"../basic/basic.h"

#define MATRIX_INVERSE__ALGORITHM(blas_num_type, LAPACKE_dgetrf, LAPACKE_dgetri) \
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
 \
    lapack_int *ipiv = (lapack_int*)malloc((uint64_t)n * sizeof(lapack_int)); \
    if (!ipiv) { \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for ipiv.", __func__, (uint64_t)n * sizeof(lapack_int)); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    int info_getrf = LAPACKE_dgetrf( \
        LAPACK_ROW_MAJOR, n, n, \
        (blas_num_type*)matrix, n, ipiv); \
    if (info_getrf != 0) { \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        if (info_getrf < 0){ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: the %d-th argument had an illegal value", __func__, -info_getrf); \
        }else{ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: U(%d,%d) is exactly zero. The factorization has been completed, but the factor U is exactly singular, and division by zero will occur if it is used to solve a system of equations", __func__, info_getrf, info_getrf); \
        } \
        Madd_Error_Add(MADD_ERROR, error_info); \
        free(ipiv); \
        return false; \
    } \
 \
    int info_getri = LAPACKE_dgetri( \
        LAPACK_ROW_MAJOR, n, \
        (blas_num_type*)matrix, n, ipiv); \
    if (info_getri != 0) { \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        if (info_getri < 0){ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: the %d-th argument had an illegal value", __func__, -info_getri); \
        }else{ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: U(%d,%d) is exactly zero; the matrix is singular and its inverse could not be computed", __func__, info_getri, info_getri); \
        } \
        Madd_Error_Add(MADD_ERROR, error_info); \
        free(ipiv); \
        return false; \
    } \
 \
    free(ipiv); \
    return true; \
} \

bool Matrix_Inverse(int32_t n, double *matrix)
MATRIX_INVERSE__ALGORITHM(double, LAPACKE_dgetrf, LAPACKE_dgetri)

bool Matrix_Inverse_f32(int32_t n, float *matrix)
MATRIX_INVERSE__ALGORITHM(float, LAPACKE_sgetrf, LAPACKE_sgetri)

bool Matrix_Inverse_c64(int32_t n, Cnum *matrix)
MATRIX_INVERSE__ALGORITHM(lapack_complex_double, LAPACKE_zgetrf, LAPACKE_zgetri)

bool Matrix_Inverse_c32(int32_t n, Cnum32 *matrix)
MATRIX_INVERSE__ALGORITHM(lapack_complex_float, LAPACKE_cgetrf, LAPACKE_cgetri)