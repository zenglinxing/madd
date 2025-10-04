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

#define IDCT2__ALGOTIRHM(Cnum, real_type, cos, sin, Fast_Fourier_Transform, sqrt) \
{ \
    if (n == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n = 0.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return false; \
    } \
    if (arr == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: array pointer (arr) is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (n > UINT64_MAX / sizeof(Cnum)){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n is too large, causing integer overflow.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (n == 1) return true; \
 \
    uint64_t m = n << 1; \
    Cnum *x = (Cnum*)malloc(m*sizeof(Cnum)); \
    if (x == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for x.", __func__, n*sizeof(Cnum)); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    real_type angle_base = _CONSTANT_PI / ((real_type)2 * n); \
    x[0].real = arr[0] * sqrt(2); \
    x[0].imag = x[n].real = x[n].imag = 0; \
    for (uint64_t i=1; i<n; i++){ \
        real_type angle = i *angle_base; \
        x[i].real = x[m - i].real = arr[i] * cos(angle); \
        x[i].imag = arr[i] * sin(angle); \
        x[m - i].imag = - x[i].imag; \
    } \
    bool flag_fft = Fast_Fourier_Transform(m, x, MADD_FFT_INVERSE); \
    if (!flag_fft){ \
        free(x); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: see info from %hs.", __func__, "Fast_Fourier_Transform"); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    real_type scale = sqrt((real_type)2 * n); \
    for (uint64_t i=0; i<n; i++){ \
        arr[i] = scale * x[i].real; \
    } \
    free(x); \
    return true; \
} \

bool Inverse_Discrete_Cosine_Transform_2(uint64_t n, double *arr)
IDCT2__ALGOTIRHM(Cnum, double, cos, sin, Fast_Fourier_Transform, sqrt)

bool Inverse_Discrete_Cosine_Transform_2_f32(uint32_t n, float *arr)
IDCT2__ALGOTIRHM(Cnum32, float, cosf, sinf, Fast_Fourier_Transform_c32, sqrtf)

bool Inverse_Discrete_Cosine_Transform_2_fl(uint64_t n, long double *arr)
IDCT2__ALGOTIRHM(Cnuml, long double, cosl, sinl, Fast_Fourier_Transform_cl, sqrtl)

#ifdef ENABLE_QUADPRECISION
bool Inverse_Discrete_Cosine_Transform_2_f128(uint64_t n, __float128 *arr)
IDCT2__ALGOTIRHM(Cnum128, __float128, cosq, sinq, Fast_Fourier_Transform_c128, sqrtq)
#endif /* ENABLE_QUADPRECISION */