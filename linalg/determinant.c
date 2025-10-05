/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/determinant.c
*/
#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>
#include<lapacke.h>
#include"linalg.h"
#include"../basic/basic.h"

#define DET__ALGORITHM(num_type, LAPACKE_dgetrf) \
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
    int *ipiv = (int*)malloc((uint64_t)n * sizeof(int)); \
    int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, n, n, matrix, n, ipiv); \
    if (info){ \
        free(ipiv); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        if (info < 0){ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: the %d-th argument had an illegal value.", __func__, -info); \
            Madd_Error_Add(MADD_ERROR, error_info); \
            return false; \
        } \
        else{ \
            /*swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: U(%d,%d) is exactly zero. The factorization has been completed, but the factor U is exactly singular, and division by zero will occur if it is used to solve a system of equations.", __func__, info, info);*/ \
            /*Madd_Error_Add(MADD_WARNING, error_info);*/ \
            *res = 0; \
            return true; \
        } \
    } \
 \
    signed char sign = 1; \
    num_type *p = matrix, result = 1; \
    for (int i=0, n_step=n+1; i<n; i++, p += n_step){ \
        if (ipiv[i] != i + 1){ \
            sign *= -1; \
        } \
        result *= *p; \
    } \
    *res = result * sign; \
 \
    free(ipiv); \
    return true; \
} \

bool Determinant(int n, double *matrix, double *res)
DET__ALGORITHM(double, LAPACKE_dgetrf)

bool Determinant_f32(int n, float *matrix, float *res)
DET__ALGORITHM(float, LAPACKE_sgetrf)

#define DET_CNUM__ALGORITHM(Cnum, LAPACKE_zgetrf, lapack_complex_double, Cnum_Mul, Cnum_Mul_Real) \
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
    int *ipiv = (int*)malloc((uint64_t)n * sizeof(int)); \
    int info = LAPACKE_zgetrf(LAPACK_ROW_MAJOR, n, n, (lapack_complex_double*)matrix, n, ipiv); \
    if (info){ \
        free(ipiv); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        if (info < 0){ \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: the %d-th argument had an illegal value.", __func__, -info); \
            Madd_Error_Add(MADD_ERROR, error_info); \
            return false; \
        } \
        else{ \
            /*swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: U(%d,%d) is exactly zero. The factorization has been completed, but the factor U is exactly singular, and division by zero will occur if it is used to solve a system of equations.", __func__, info, info);*/ \
            /*Madd_Error_Add(MADD_WARNING, error_info);*/ \
            res->real = res->imag = 0; \
            return true; \
        } \
    } \
 \
    signed char sign = 1; \
    Cnum *p = matrix, result = {.real=1, .imag=0}; \
    for (int i=0, n_step=n+1; i<n; i++, p += n_step){ \
        if (ipiv[i] != i + 1){ \
            sign *= -1; \
        } \
        result = Cnum_Mul(result, *p); \
    } \
    *res = Cnum_Mul_Real(result, sign); \
 \
    free(ipiv); \
    return true; \
} \

bool Determinant_c64(int n, Cnum *matrix, Cnum *res)
DET_CNUM__ALGORITHM(Cnum, LAPACKE_zgetrf, lapack_complex_double, Cnum_Mul, Cnum_Mul_Real)

bool Determinant_c32(int n, Cnum32 *matrix, Cnum32 *res)
DET_CNUM__ALGORITHM(Cnum32, LAPACKE_cgetrf, lapack_complex_float, Cnum_Mul_c32, Cnum_Mul_Real_c32)