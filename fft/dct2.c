/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fft/dct2.c
Discrete Cosine Transform (DCT-II)

Implementation of DCT-II using FFT. The algorithm extends the input signal to 2N points,
performs FFT, and then applies a phase shift and scaling to get the DCT-II coefficients.

Standard DCT-II formula:
  X_k = c_k * sqrt(2/N) * sum_{n=0}^{N-1} x_n * cos(pi * k * (2n+1) / (2N))
where c_0 = 1/sqrt(2) and c_k = 1 for k > 0.

In this code, we use FFT to compute it efficiently.
*/
#include<stdlib.h>
#include<math.h>
#include<stdint.h>
#include<stdbool.h>

#include"fft.h"
#include"../basic/basic.h"

#define DCT2__ALGORITHM(Discrete_Cosine_Transform_2_Base2, Discrete_Cosine_Transform_2_Naive) \
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
    uint64_t log2_n_floor, log2_n_ceil, n2 = (uint64_t)n << 1; \
    Log2_Full(n2, &log2_n_floor, &log2_n_ceil); \
    if (log2_n_floor == log2_n_ceil){ \
        Discrete_Cosine_Transform_2_Base2(n, arr); \
    } \
    else{ \
        Discrete_Cosine_Transform_2_Naive(n, arr); \
    } \
} \

void Discrete_Cosine_Transform_2(uint64_t n, double *arr)
DCT2__ALGORITHM(Discrete_Cosine_Transform_2_Base2, Discrete_Cosine_Transform_2_Naive)

void Discrete_Cosine_Transform_2_f32(uint32_t n, float *arr)
DCT2__ALGORITHM(Discrete_Cosine_Transform_2_Base2_f32, Discrete_Cosine_Transform_2_Naive_f32)

void Discrete_Cosine_Transform_2_fl(uint64_t n, long double *arr)
DCT2__ALGORITHM(Discrete_Cosine_Transform_2_Base2_fl, Discrete_Cosine_Transform_2_Naive_fl)

#ifdef ENABLE_QUADPRECISION
void Discrete_Cosine_Transform_2_f128(uint64_t n, __float128 *arr)
DCT2__ALGORITHM(Discrete_Cosine_Transform_2_Base2_f128, Discrete_Cosine_Transform_2_Naive_f128)
#endif /* ENABLE_QUADPRECISION */