/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/matrix_multiply_vector.c
*/
#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>
#include<cblas.h>
#include"linalg.h"
#include"../basic/basic.h"

#define MATRIX_MULTIPLY_VECTOR__ALGORITHM(Madd_Set0, cblas_dgemv, CblasNoTrans) \
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
    if (matrix == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (vector == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: vector is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (result == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: result is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    Madd_Set0(n, result); \
    cblas_dgemv(CblasRowMajor, CblasNoTrans, m, n, \
                1, matrix, n, vector, 1, 0, result, 1); \
    return true; \
} \

bool Matrix_Multiply_Vector(int32_t m, int32_t n, const double *matrix, const double *vector, double *result)
MATRIX_MULTIPLY_VECTOR__ALGORITHM(Madd_Set0, cblas_dgemv, CblasNoTrans)

bool Matrix_Multiply_Vector_f32(int32_t m, int32_t n, const float *matrix, const float *vector, float *result)
MATRIX_MULTIPLY_VECTOR__ALGORITHM(Madd_Set0_f32, cblas_sgemv, CblasNoTrans)

bool Vector_Multiply_Matrix(int32_t m, int32_t n, const double *vector, const double *matrix, double *result)
MATRIX_MULTIPLY_VECTOR__ALGORITHM(Madd_Set0, cblas_dgemv, CblasTrans)

bool Vector_Multiply_Matrix_f32(int32_t m, int32_t n, const float *vector, const float *matrix, float *result)
MATRIX_MULTIPLY_VECTOR__ALGORITHM(Madd_Set0_f32, cblas_sgemv, CblasTrans)

#define MATRIX_MULTIPLY_VECTOR_CNUM__ALGORITHM(Cnum, Madd_Set0_c64, cblas_zgemv, CblasNoTrans) \
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
    if (matrix == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (vector == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: vector is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (result == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: result is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    Cnum alpha = {.real=1, .imag=0}, beta = {.real=0, .imag=0}; \
    Madd_Set0_c64(n, result); \
    cblas_zgemv(CblasRowMajor, CblasNoTrans, m, n, \
                (void*)&alpha, (void*)matrix, n, (void*)vector, 1, (void*)&beta, (void*)result, 1); \
    return true; \
} \

bool Matrix_Multiply_Vector_c64(int32_t m, int32_t n, const Cnum *matrix, const Cnum *vector, Cnum *result)
MATRIX_MULTIPLY_VECTOR_CNUM__ALGORITHM(Cnum, Madd_Set0_c64, cblas_zgemv, CblasNoTrans)

bool Matrix_Multiply_Vector_c32(int32_t m, int32_t n, const Cnum32 *matrix, const Cnum32 *vector, Cnum32 *result)
MATRIX_MULTIPLY_VECTOR_CNUM__ALGORITHM(Cnum32, Madd_Set0_c32, cblas_cgemv, CblasNoTrans)

bool Vector_Multiply_Matrix_c64(int32_t m, int32_t n, const Cnum *vector, const Cnum *matrix, Cnum *result)
MATRIX_MULTIPLY_VECTOR_CNUM__ALGORITHM(Cnum, Madd_Set0_c64, cblas_zgemv, CblasTrans)

bool Vector_Multiply_Matrix_c32(int32_t m, int32_t n, const Cnum32 *vector, const Cnum32 *matrix, Cnum32 *result)
MATRIX_MULTIPLY_VECTOR_CNUM__ALGORITHM(Cnum32, Madd_Set0_c32, cblas_cgemv, CblasTrans)