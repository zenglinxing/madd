/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./linalg/matrix_transpose_inplace.c
*/
#include<stdint.h>
#include<stdbool.h>
#include"linalg.h"
#include"../basic/basic.h"

#define MATRIX_TRANSPOSE_INPLACE__ALGORITHM(num_type) \
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
    uint64_t total = (uint64_t)m * n; \
    for (uint64_t start = 0; start < total; start++){ \
        uint64_t next = start; \
        do { \
            /* 计算下一个位置（置换环映射） */ \
            next = (next % m) * n + (next / m); \
        } while (next < start); /* 确保仅处理未移动的环起点 */ \
 \
        if (next == start){ /* 找到新置换环起点 */ \
            num_type temp = matrix[start]; \
            uint64_t current = start; \
            do { \
                next = (current % m) * n + (current / m); \
                if (next != start) { \
                    matrix[current] = matrix[next]; \
                } else { \
                    matrix[current] = temp; \
                } \
                current = next; \
            } while (current != start); \
        } \
    } \
    return true; \
} \

bool Matrix_Transpose_Inplace(uint64_t m, uint64_t n, double *matrix)
MATRIX_TRANSPOSE_INPLACE__ALGORITHM(double)

bool Matrix_Transpose_Inplace_f32(uint32_t m, uint32_t n, float *matrix)
MATRIX_TRANSPOSE_INPLACE__ALGORITHM(float)

bool Matrix_Transpose_Inplace_fl(uint64_t m, uint64_t n, long double *matrix)
MATRIX_TRANSPOSE_INPLACE__ALGORITHM(long double)

#ifdef ENABLE_QUADPRECISION
bool Matrix_Transpose_Inplace_f128(uint64_t m, uint64_t n, __float128 *matrix)
MATRIX_TRANSPOSE_INPLACE__ALGORITHM(__float128)
#endif /* ENABLE_QUADPRECISION */

bool Matrix_Transpose_Inplace_c64(uint64_t m, uint64_t n, Cnum *matrix)
MATRIX_TRANSPOSE_INPLACE__ALGORITHM(Cnum)

bool Matrix_Transpose_Inplace_c32(uint64_t m, uint64_t n, Cnum32 *matrix)
MATRIX_TRANSPOSE_INPLACE__ALGORITHM(Cnum32)

bool Matrix_Transpose_Inplace_cl(uint64_t m, uint64_t n, Cnuml *matrix)
MATRIX_TRANSPOSE_INPLACE__ALGORITHM(Cnuml)

#ifdef ENABLE_QUADPRECISION
bool Matrix_Transpose_Inplace_c128(uint64_t m, uint64_t n, Cnum128 *matrix)
MATRIX_TRANSPOSE_INPLACE__ALGORITHM(Cnum128)
#endif /* ENABLE_QUADPRECISION */