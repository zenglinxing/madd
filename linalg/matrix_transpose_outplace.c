/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/matrix_transpose_outplace.c
*/
#include<stdint.h>
#include<stdbool.h>
#include"linalg.h"
#include"../basic/basic.h"

#define MATRIX_TRANSPOSE_OUTPLACE__ALGORITHM(num_type) \
{ \
    if (matrix == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: matrix is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (newmat == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: newmat is NULL.", __func__); \
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
    uint64_t i, j; \
    num_type *p1 = matrix, *p2; \
    for (i=0; i<m; i++){ \
        p2 = newmat + i; \
        for (j=0; j<n; j++,p2+=m){ \
            *p2 = *p1; \
            p1 ++; \
        } \
    } \
    return true; \
} \

bool Matrix_Transpose_Outplace(uint64_t m, uint64_t n, double *matrix, double *newmat)
MATRIX_TRANSPOSE_OUTPLACE__ALGORITHM(double)

bool Matrix_Transpose_Outplace_f32(uint32_t m, uint32_t n, float *matrix, float *newmat)
MATRIX_TRANSPOSE_OUTPLACE__ALGORITHM(float)

bool Matrix_Transpose_Outplace_fl(uint64_t m, uint64_t n, long double *matrix, long double *newmat)
MATRIX_TRANSPOSE_OUTPLACE__ALGORITHM(long double)

#ifdef ENABLE_QUADPRECISION
bool Matrix_Transpose_Outplace_f128(uint64_t m, uint64_t n, __float128 *matrix, __float128 *newmat)
MATRIX_TRANSPOSE_OUTPLACE__ALGORITHM(__float128)
#endif /* ENABLE_QUADPRECISION */

bool Matrix_Transpose_Outplace_c64(uint64_t m, uint64_t n, Cnum *matrix, Cnum *newmat)
MATRIX_TRANSPOSE_OUTPLACE__ALGORITHM(Cnum)

bool Matrix_Transpose_Outplace_c32(uint64_t m, uint64_t n, Cnum32 *matrix, Cnum32 *newmat)
MATRIX_TRANSPOSE_OUTPLACE__ALGORITHM(Cnum32)

bool Matrix_Transpose_Outplace_cl(uint64_t m, uint64_t n, Cnuml *matrix, Cnuml *newmat)
MATRIX_TRANSPOSE_OUTPLACE__ALGORITHM(Cnuml)

#ifdef ENABLE_QUADPRECISION
bool Matrix_Transpose_Outplace_c128(uint64_t m, uint64_t n, Cnum128 *matrix, Cnum128 *newmat)
MATRIX_TRANSPOSE_OUTPLACE__ALGORITHM(Cnum128)
#endif /* ENABLE_QUADPRECISION */