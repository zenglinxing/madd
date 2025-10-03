/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fft/fft.c
*/
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>

#include"fft.h"
#include"../basic/basic.h"

#define FFT__ALGORITHM(Fast_Fourier_Transform_Radix2, Fast_Fourier_Transform_Radix2_func_name, \
                       Discrete_Fourier_Transform_Naive, Discrete_Fourier_Transform_Naive_func_name) \
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
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: fft_direction should be either MADD_FFT_FORWARD or MADD_FFT_INVERSE. You set %d.", __func__, fft_direction); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    uint64_t n_floor, n_ceil; \
    bool flag_fft; \
    Log2_Full(n, &n_floor, &n_ceil); \
    if (n_floor == n_ceil){ \
        flag_fft = Fast_Fourier_Transform_Radix2(n, arr, fft_direction); \
        if (!flag_fft){ \
            wchar_t error_info[MADD_ERROR_INFO_LEN]; \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: see info from %hs.", __func__, Fast_Fourier_Transform_Radix2_func_name); \
            return false; \
        } \
    }else{ \
        flag_fft = Discrete_Fourier_Transform_Naive(n, arr, fft_direction); \
        if (!flag_fft){ \
            wchar_t error_info[MADD_ERROR_INFO_LEN]; \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: see info from %hs.", __func__, Discrete_Fourier_Transform_Naive_func_name); \
            return false; \
        } \
    } \
    return true; \
} \

bool Fast_Fourier_Transform(uint64_t n, Cnum *arr, int fft_direction)
FFT__ALGORITHM(Fast_Fourier_Transform_Radix2, "Fast_Fourier_Transform_Radix2",
               Fast_Fourier_Transform_Bluestein, "Fast_Fourier_Transform_Bluestein")

bool Fast_Fourier_Transform_c32(uint32_t n, Cnum32 *arr, int fft_direction)
FFT__ALGORITHM(Fast_Fourier_Transform_Radix2_c32, "Fast_Fourier_Transform_Radix2_c32",
               Discrete_Fourier_Transform_Naive_c32, "Discrete_Fourier_Transform_Naive_c32")

bool Fast_Fourier_Transform_cl(uint64_t n, Cnuml *arr, int fft_direction)
FFT__ALGORITHM(Fast_Fourier_Transform_Radix2_cl, "Fast_Fourier_Transform_Radix2_cl",
               Discrete_Fourier_Transform_Naive_cl, "Discrete_Fourier_Transform_Naive_cl")

#ifdef ENABLE_QUADPRECISION
bool Fast_Fourier_Transform_c128(uint64_t n, Cnum128 *arr, int fft_direction)
FFT__ALGORITHM(Fast_Fourier_Transform_Radix2_c128, "Fast_Fourier_Transform_Radix2_c128",
               Discrete_Fourier_Transform_Naive_c128, "Discrete_Fourier_Transform_Naive_c128")
#endif /* ENABLE_QUADPRECISION */