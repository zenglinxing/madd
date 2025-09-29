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

#define DCT2__ALGORITHM(Fast_Fourier_Transform, Fast_Fourier_Transform_name, \
                        Cnum, real_type, cos, sin, sqrt) \
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
 \
    uint64_t len_fft = (uint64_t)n << 1, i; \
    Cnum *arr_fft = (Cnum*)malloc(len_fft * sizeof(Cnum)); \
    if (arr_fft == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for FFT array.", __func__, len_fft * sizeof(Cnum)); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    for (i=0; i<n; i++){ \
        arr_fft[i].real = arr[i]; \
        arr_fft[i].imag = arr_fft[n + i].imag = 0; \
        arr_fft[n + i].real = arr[n - 1 - i]; \
    } \
 \
    bool flag_fft = Fast_Fourier_Transform(len_fft, arr_fft, MADD_FFT_FORWARD); \
    if (!flag_fft){ \
        free(arr_fft); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: see error info from %hs.", __func__, Fast_Fourier_Transform_name); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    const real_type scale = sqrt(2.0 / n), angle_base = _CONSTANT_PI / len_fft; \
    for (i=0; i<n; i++){ \
        real_type theta = angle_base * i; \
        real_type cos_theta = cos(theta), sin_theta = sin(theta); \
        real_type real_part = arr_fft[i].real * cos_theta + arr_fft[i].imag * sin_theta; \
        real_type c_i = (i == 0) ? sqrt(0.5) : 1; \
        arr[i] = scale * c_i * real_part / 2.; \
    } \
 \
    free(arr_fft); \
    return true; \
} \

bool Discrete_Cosine_Transform_2(uint64_t n, double *arr)
DCT2__ALGORITHM(Fast_Fourier_Transform, "Fast_Fourier_Transform",
                Cnum, double, cos, sin, sqrt)

bool Discrete_Cosine_Transform_2_f32(uint32_t n, float *arr)
DCT2__ALGORITHM(Fast_Fourier_Transform_f32, "Fast_Fourier_Transform_f32",
                Cnum32, float, cosf, sinf, sqrtf)

bool Discrete_Cosine_Transform_2_fl(uint64_t n, long double *arr)
DCT2__ALGORITHM(Fast_Fourier_Transform_fl, "Fast_Fourier_Transform_fl",
                Cnuml, long double, cosl, sinl, sqrtl)

#ifdef ENABLE_QUADPRECISION
bool Discrete_Cosine_Transform_2_f128(uint64_t n, __float128 *arr)
DCT2__ALGORITHM(Fast_Fourier_Transform_f128, "Fast_Fourier_Transform_f128",
                Cnum128, __float128, cosq, sinq, sqrtq)
#endif /* ENABLE_QUADPRECISION */