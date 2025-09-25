/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fft/dct2-base2.c
*/
#include<stdlib.h>
#include<math.h>
#include<stdint.h>
#include<stdbool.h>

#include"fft.h"
#include"../basic/basic.h"
/*#include"cnum.h""*/

#define DCT2_RADIX2__ALGORITHM(Cnum, real_type, \
                              Fast_Fourier_Transform_Core_name, \
                              Fast_Fourier_Transform_Malloc, Fast_Fourier_Transform_Weight, \
                              Fast_Fourier_Transform_Core, \
                              sqrt, sin, cos) \
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
    if (n > UINT64_MAX / (2 * sizeof(Cnum))){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n is too large, causing integer overflow.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return; \
    } \
 \
    uint64_t len_fft = 2 * n, log2_n_ceil = Log2_Ceil(len_fft), n_ceil = (uint64_t)1 << log2_n_ceil, i; \
    Cnum *w = (Cnum*)malloc(n_ceil * sizeof(Cnum)); \
    if (w == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for weight points.", __func__, n_ceil * sizeof(Cnum)); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return; \
    } \
    Cnum *arr_fft = (Cnum*)Fast_Fourier_Transform_Malloc(len_fft); \
    if (arr_fft == NULL){ \
        free(w); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for FFT array.", __func__, len_fft * sizeof(Cnum)); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return; \
    } \
    for (i=0; i<n; i++){ \
        arr_fft[i].real = arr[i]; \
        /*arr_fft[i].imag = 0;*/ \
        arr_fft[n + i].real = arr[n - 1 - i]; \
    } \
 \
    Fast_Fourier_Transform_Weight(n_ceil, w, MADD_FFT_FORWARD); \
    bool flag_fft = Fast_Fourier_Transform_Core(n_ceil, arr_fft, w); \
    if (!flag_fft){ \
        free(w); \
        free(arr_fft); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: see error info from %hs.", __func__, Fast_Fourier_Transform_Core_name); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return; \
    } \
 \
    const real_type scale = sqrt(2.0 / n); \
    for (i=0; i<n; i++){ \
        real_type theta = _CONSTANT_PI * i / (2.0 * n); \
        real_type cos_theta = cos(theta), sin_theta = sin(theta); \
        real_type real_part = arr_fft[i].real * cos_theta + arr_fft[i].imag * sin_theta; \
        real_type c_i = (i == 0) ? sqrt(0.5) : 1; \
        arr[i] = scale * c_i * real_part / 2.; \
    } \
 \
    free(w); \
    free(arr_fft); \
} \

void Discrete_Cosine_Transform_2_Radix2(uint64_t n, double *arr)
DCT2_RADIX2__ALGORITHM(Cnum, double,
                       "Fast_Fourier_Transform_Radix2_Core",
                       Fast_Fourier_Transform_Radix2_Malloc, Fast_Fourier_Transform_Weight,
                       Fast_Fourier_Transform_Radix2_Core,
                       sqrt, sin, cos)

void Discrete_Cosine_Transform_2_Radix2_f32(uint32_t n, float *arr)
DCT2_RADIX2__ALGORITHM(Cnum32, float,
                       "Fast_Fourier_Transform_Radix2_Core_f32",
                       Fast_Fourier_Transform_Radix2_Malloc_f32, Fast_Fourier_Transform_Weight_f32,
                       Fast_Fourier_Transform_Radix2_Core_f32,
                       sqrtf, sinf, cosf)

void Discrete_Cosine_Transform_2_Radix2_fl(uint64_t n, long double *arr)
DCT2_RADIX2__ALGORITHM(Cnuml, long double,
                       "Fast_Fourier_Transform_Radix2_Core_fl",
                       Fast_Fourier_Transform_Radix2_Malloc_fl, Fast_Fourier_Transform_Weight_fl,
                       Fast_Fourier_Transform_Radix2_Core_fl,
                       sqrtl, sinl, cosl)

#ifdef ENABLE_QUADPRECISION
void Discrete_Cosine_Transform_2_Radix2_f128(uint64_t n, __float128 *arr)
DCT2_RADIX2__ALGORITHM(Cnum128, __float128,
                       "Fast_Fourier_Transform_Radix2_Core_f128",
                       Fast_Fourier_Transform_Radix2_Malloc_f128, Fast_Fourier_Transform_Weight_f128,
                       Fast_Fourier_Transform_Radix2_Core_f128,
                       sqrtq, sinq, cosq)
#endif /* ENABLE_QUADPRECISION */