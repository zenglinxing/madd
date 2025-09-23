/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/matrix_transpose.c
*/
#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>
#include"linalg.h"
#include"../basic/basic.h"

#define MATRIX_TRANSPOSE__ALGORITHM(num_type, Matrix_Transpose_Inplace, Matrix_Transpose_Outplace) \
{ \
    if (matrix == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
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
 \
    size_t size_matrix = (uint64_t)m * n * sizeof(num_type); \
    num_type *new_matrix = (num_type*)malloc(size_matrix); \
    if (new_matrix == NULL){ \
        Matrix_Transpose_Inplace(m, n, matrix); \
    }else{ \
        Matrix_Transpose_Outplace(m, n, matrix, new_matrix); \
        memcpy(matrix, new_matrix, size_matrix); \
        free(new_matrix); \
    } \
    return true; \
} \

bool Matrix_Transpose(uint64_t m, uint64_t n, double *matrix)
MATRIX_TRANSPOSE__ALGORITHM(double, Matrix_Transpose_Inplace, Matrix_Transpose_Outplace)

bool Matrix_Transpose_f32(uint32_t m, uint32_t n, float *matrix)
MATRIX_TRANSPOSE__ALGORITHM(float, Matrix_Transpose_Inplace_f32, Matrix_Transpose_Outplace_f32)

bool Matrix_Transpose_fl(uint64_t m, uint64_t n, long double *matrix)
MATRIX_TRANSPOSE__ALGORITHM(long double, Matrix_Transpose_Inplace_fl, Matrix_Transpose_Outplace_fl)

#ifdef ENABLE_QUADPRECISION
bool Matrix_Transpose_f128(uint64_t m, uint64_t n, __float128 *matrix)
MATRIX_TRANSPOSE__ALGORITHM(__float128, Matrix_Transpose_Inplace_f128, Matrix_Transpose_Outplace_f128)
#endif /* ENABLE_QUADPRECISION */

bool Matrix_Transpose_c64(uint64_t m, uint64_t n, Cnum *matrix)
MATRIX_TRANSPOSE__ALGORITHM(Cnum, Matrix_Transpose_Inplace_c64, Matrix_Transpose_Outplace_c64)

bool Matrix_Transpose_c32(uint64_t m, uint64_t n, Cnum32 *matrix)
MATRIX_TRANSPOSE__ALGORITHM(Cnum32, Matrix_Transpose_Inplace_c32, Matrix_Transpose_Outplace_c32)

bool Matrix_Transpose_cl(uint64_t m, uint64_t n, Cnuml *matrix)
MATRIX_TRANSPOSE__ALGORITHM(Cnuml, Matrix_Transpose_Inplace_cl, Matrix_Transpose_Outplace_cl)

#ifdef ENABLE_QUADPRECISION
bool Matrix_Transpose_c128(uint64_t m, uint64_t n, Cnum128 *matrix)
MATRIX_TRANSPOSE__ALGORITHM(Cnum128, Matrix_Transpose_Inplace_c128, Matrix_Transpose_Outplace_c128)
#endif /* ENABLE_QUADPRECISION */