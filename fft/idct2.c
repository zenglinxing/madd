/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fft/idct2.c
Inverse Discrete Cosine Transform (DCT-II)
*/
#include<stdlib.h>
#include<math.h>
#include<stdint.h>
#include<stdbool.h>

#include"fft.h"
#include"../basic/basic.h"
/*#include"cnum.h""*/

#define IDCT2__ALGOTIRHM(Inverse_Discrete_Cosine_Transform_2_Naive) \
{ \
    if (n == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n = 0.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return; \
    } \
    if (arr == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: array pointer (arr) is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return; \
    } \
    if (n > UINT64_MAX / sizeof(Cnum)){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n is too large, causing integer overflow.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return; \
    } \
 \
    Inverse_Discrete_Cosine_Transform_2_Naive(n, arr); \
} \

void Inverse_Discrete_Cosine_Transform_2(uint64_t n, double *arr)
IDCT2__ALGOTIRHM(Inverse_Discrete_Cosine_Transform_2_Naive)

void Inverse_Discrete_Cosine_Transform_2_f32(uint32_t n, float *arr)
IDCT2__ALGOTIRHM(Inverse_Discrete_Cosine_Transform_2_Naive_f32)

void Inverse_Discrete_Cosine_Transform_2_fl(uint64_t n, long double *arr)
IDCT2__ALGOTIRHM(Inverse_Discrete_Cosine_Transform_2_Naive_fl)

#ifdef ENABLE_QUADPRECISION
void Inverse_Discrete_Cosine_Transform_2_f128(uint64_t n, __float128 *arr)
IDCT2__ALGOTIRHM(Inverse_Discrete_Cosine_Transform_2_Naive_f128)
#endif /* ENABLE_QUADPRECISION */