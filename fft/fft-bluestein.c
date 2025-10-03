/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fft/fft-bluestein.c
*/
#include<string.h>
#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>

#include"fft.h"
#include"../basic/basic.h"

#define FFT_BLUESTEIN__ALGORITHM(Cnum, \
                                 Fast_Fourier_Transform_Weight, \
                                 cos, sin, \
                                 Cnum_Mul, Cnum_Conj, Cnum_Div_Real, \
                                 Madd_Set0_c64, \
                                 Fast_Fourier_Transform_Radix2_Core) \
{ \
    if (n == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n is 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (arr == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: arr is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (fft_direction != MADD_FFT_FORWARD && fft_direction != MADD_FFT_INVERSE){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: fft_direction should be either MADD_FFT_FORWARD or MADD_FFT_INVERSE, but you set %d.", __func__, fft_direction); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    uint64_t m = (uint64_t)1 << Log2_Ceil(2*(uint64_t)n-1); \
    size_t size_abc = 3 * m * sizeof(Cnum), size_wm = m * sizeof(Cnum); \
 \
    Cnum *a = (Cnum*)malloc(size_abc), *b, *c; \
    if (a == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for a & b & c.", __func__, size_abc); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    b = a + m; \
    c = b + m; \
    /* wm & wn */ \
    Cnum *wm = (Cnum*)malloc(size_wm); \
    if (wm == NULL){ \
        free(a); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for wm (weight).", __func__, size_wm); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    Fast_Fourier_Transform_Weight(m, wm, MADD_FFT_FORWARD); \
 \
    uint64_t i; \
    for (i=0; i<n; i++){ \
        double angle = fft_direction * _CONSTANT_PI * i * i / n; \
        Cnum chirp = {.real = cos(angle), .imag = sin(angle)}; \
        a[i] = Cnum_Mul(arr[i], chirp); \
        b[i] = Cnum_Conj(chirp); \
        if (i) b[m - i] = b[i]; \
    } \
    /* Madd_Set0_c64(uint64_t N, Cnum *array): Set the length N of array to be 0; c64 means the real and imag of complex are all 64-bit */ \
    Madd_Set0_c64(m - n, a + n); \
    Madd_Set0_c64(m - 2 * n + 1, b + n); \
 \
    Fast_Fourier_Transform_Radix2_Core(m, a, wm); \
    Fast_Fourier_Transform_Radix2_Core(m, b, wm); \
 \
    for (i=0; i<m; i++){ \
        Cnum temp = Cnum_Mul(a[i], b[i]); \
        c[i] = Cnum_Conj(temp); \
    } \
    Fast_Fourier_Transform_Radix2_Core(m, c, wm); \
 \
    for (i=0; i<n; i++){ \
        double angle =  fft_direction * _CONSTANT_PI * i * i / n; \
        Cnum chirp = {.real = cos(angle), .imag = sin(angle)}; \
        Cnum c_temp = Cnum_Div_Real(Cnum_Conj(c[i]), m); \
        arr[i] = Cnum_Mul(c_temp, chirp); \
        if (fft_direction == MADD_FFT_INVERSE){ \
            arr[i] = Cnum_Div_Real(arr[i], n); \
        } \
    } \
 \
    free(a); \
    free(wm); \
    return true; \
} \

bool Fast_Fourier_Transform_Bluestein(uint64_t n, Cnum *arr, int fft_direction)
FFT_BLUESTEIN__ALGORITHM(Cnum,
                         Fast_Fourier_Transform_Weight,
                         cos, sin,
                         Cnum_Mul, Cnum_Conj, Cnum_Div_Real,
                         Madd_Set0_c64,
                         Fast_Fourier_Transform_Radix2_Core)

bool Fast_Fourier_Transform_Bluestein_c32(uint64_t n, Cnum32 *arr, int fft_direction)
FFT_BLUESTEIN__ALGORITHM(Cnum32,
                         Fast_Fourier_Transform_Weight_c32,
                         cosf, sinf,
                         Cnum_Mul_c32, Cnum_Conj_c32, Cnum_Div_Real_c32,
                         Madd_Set0_c32,
                         Fast_Fourier_Transform_Radix2_Core_c32)

bool Fast_Fourier_Transform_Bluestein_cl(uint64_t n, Cnuml *arr, int fft_direction)
FFT_BLUESTEIN__ALGORITHM(Cnuml,
                         Fast_Fourier_Transform_Weight_cl,
                         cosl, sinl,
                         Cnum_Mul_cl, Cnum_Conj_cl, Cnum_Div_Real_cl,
                         Madd_Set0_cl,
                         Fast_Fourier_Transform_Radix2_Core_cl)

#ifdef ENABLE_QUADPRECISION
bool Fast_Fourier_Transform_Bluestein_c128(uint64_t n, Cnum128 *arr, int fft_direction)
FFT_BLUESTEIN__ALGORITHM(Cnum128,
                         Fast_Fourier_Transform_Weight_c128,
                         cosq, sinq,
                         Cnum_Mul_c128, Cnum_Conj_c128, Cnum_Div_Real_c128,
                         Madd_Set0_c128,
                         Fast_Fourier_Transform_Radix2_Core_c128)
#endif /* ENABLE_QUADPRECISION */