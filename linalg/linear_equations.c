/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/linear_equations.c
*/
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>
#define HAVE_LAPACK_CONFIG_H
#include<lapacke.h>

#include"linalg.h"
#include"../basic/basic.h"

#define LINEAR_EQUATIONS__ALGORITHM(num_type, lapack_num_type, LAPACKE_dgesv) \
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
    size_t size_ipiv = (uint64_t)n * sizeof(int), size_coefficient = (uint64_t)n*n*sizeof(num_type); \
    int *ipiv = (int*)malloc(size_ipiv); \
    if (ipiv == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for ipiv.", __func__, size_ipiv); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, n, n_vector, \
                             (lapack_num_type*)eq, n, ipiv, \
                             (lapack_num_type*)vector, n_vector); \
    free(ipiv); \
 \
    if (info < 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: the %d-th argument had an illegal value.", __func__, -info); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (info > 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: U(%d,%d) is exactly zero.  The factorization has been completed, but the factor U is exactly singular, so the solution could not be computed.", __func__, info, info); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    return true; \
} \

bool Linear_Equations(int n, double *eq, int n_vector, double *vector)
LINEAR_EQUATIONS__ALGORITHM(double, double, LAPACKE_dgesv)

bool Linear_Equations_f32(int n, float *eq, int n_vector, float *vector)
LINEAR_EQUATIONS__ALGORITHM(float, float, LAPACKE_sgesv)

bool Linear_Equations_c64(int n, Cnum *eq, int n_vector, Cnum *vector)
LINEAR_EQUATIONS__ALGORITHM(Cnum, lapack_complex_double, LAPACKE_zgesv)

bool Linear_Equations_c32(int n, Cnum32 *eq, int n_vector, Cnum32 *vector)
LINEAR_EQUATIONS__ALGORITHM(Cnum32, lapack_complex_float, LAPACKE_cgesv)