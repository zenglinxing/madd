/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/matrix_multiply.c
*/
#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>
#include<cblas.h>
#include"linalg.h"
#include"../basic/basic.h"

#define MATRIX_MULTIPLY__ALGORITHM(num_type, cblas_dgemm, Madd_Set0) \
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
    num_type alpha = 1, beta = 0; \
    Madd_Set0((uint64_t)m*n, res); \
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                m, n, l, \
                alpha, \
                a, l, \
                b, n, \
                beta, \
                res, n); \
    return true; \
} \

#define MATRIX_MULTIPLY_CNUM__ALGORITHM(num_type, cblas_cgemm, Madd_Set0_c64) \
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
    num_type alpha = {.real=1, .imag=0}, beta = {.real=0, .imag=0}; \
    Madd_Set0_c64((uint64_t)m*n, res); \
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, \
                m, n, l, \
                &alpha, \
                a, l, \
                b, n, \
                &beta, \
                res, n); \
    return true; \
} \

bool Matrix_Multiply(int32_t m, int32_t n, int32_t l,
                     double *a, double *b, double *res)
MATRIX_MULTIPLY__ALGORITHM(double, cblas_dgemm, Madd_Set0)

bool Matrix_Multiply_f32(int32_t m, int32_t n, int32_t l,
                         float *a, float *b, float *res)
MATRIX_MULTIPLY__ALGORITHM(float, cblas_sgemm, Madd_Set0_f32)

bool Matrix_Multiply_c64(int32_t m, int32_t n, int32_t l,
                         Cnum *a, Cnum *b, Cnum *res)
MATRIX_MULTIPLY_CNUM__ALGORITHM(Cnum, cblas_zgemm, Madd_Set0_c64)

bool Matrix_Multiply_c32(int32_t m, int32_t n, int32_t l,
                         Cnum32 *a, Cnum32 *b, Cnum32 *res)
MATRIX_MULTIPLY_CNUM__ALGORITHM(Cnum32, cblas_cgemm, Madd_Set0_c32)