/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/linear_equations_tridiagonal.c
*/
#include<stdio.h>
#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>
#define HAVE_LAPACK_CONFIG_H
#include<lapacke.h>

#include"linalg.h"
#include"../basic/basic.h"

#define LINEAR_EQUATIONS_TRIDIAGONAL__ALGORITHM(num_type, lapack_num_type, LAPACKE_dgtsv) \
{ \
    if (n == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n is 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (lower == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: lower is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (diag == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: diag is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (upper == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: upper is NULL.", __func__); \
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
    int info = LAPACKE_dgtsv(LAPACK_ROW_MAJOR, n, n_vector, \
                             (lapack_num_type*)lower, \
                             (lapack_num_type*)diag, \
                             (lapack_num_type*)upper, \
                             (lapack_num_type*)vector, n_vector); \
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

bool Linear_Equations_Tridiagonal(int32_t n, double *lower, double *diag, double *upper,
                                  int32_t n_vector, double *vector)
LINEAR_EQUATIONS_TRIDIAGONAL__ALGORITHM(double, double, LAPACKE_dgtsv)

bool Linear_Equations_Tridiagonal_f32(int32_t n, float *lower, float *diag, float *upper,
                                      int32_t n_vector, float *vector)
LINEAR_EQUATIONS_TRIDIAGONAL__ALGORITHM(float, float, LAPACKE_sgtsv)

bool Linear_Equations_Tridiagonal_c64(int32_t n, Cnum *lower, Cnum *diag, Cnum *upper,
                                      int32_t n_vector, Cnum *vector)
LINEAR_EQUATIONS_TRIDIAGONAL__ALGORITHM(Cnum, lapack_complex_double, LAPACKE_zgtsv)

bool Linear_Equations_Tridiagonal_c32(int32_t n, Cnum32 *lower, Cnum32 *diag, Cnum32 *upper,
                                      int32_t n_vector, Cnum32 *vector)
LINEAR_EQUATIONS_TRIDIAGONAL__ALGORITHM(Cnum32, lapack_complex_float, LAPACKE_cgtsv)