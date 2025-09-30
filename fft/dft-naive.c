/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fft/dft-naive.c
*/
#include<string.h>
#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>

#include"fft.h"
#include"../basic/basic.h"

#define DFT_NAIVE__ALGORITHM(integer_type, Cnum, Cnum_Value, Cnum_Add, Cnum_Mul, Cnum_Div_Real, \
                             Fast_Fourier_Transform_Weight) \
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
    size_t size_w = (uint64_t)n*sizeof(Cnum); \
    Cnum *w = (Cnum*)malloc(size_w); \
    if (w == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for w.", __func__, size_w); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    Fast_Fourier_Transform_Weight(n, w, fft_direction); \
 \
    size_t size_narr = (uint64_t)n*sizeof(Cnum); \
    Cnum *narr = (Cnum*)malloc(size_narr); \
    if (narr == NULL){ \
        free(w); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for narr.", __func__, size_narr); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    memcpy(narr, arr, size_narr); \
 \
    uint64_t i, j; \
    for (i=0; i<n; i++){ \
        Cnum sum = Cnum_Value(0, 0); \
        uint64_t w_index = 0; \
        for (j=0; j<n; j++){ \
            sum = Cnum_Add(sum, Cnum_Mul(narr[j], w[w_index])); \
            w_index = (w_index + i) % n; \
        } \
        arr[i] = sum; \
    } \
 \
    if (fft_direction == MADD_FFT_INVERSE){ \
        for (i=0; i<n; i++){ \
            arr[i] = Cnum_Div_Real(arr[i], n); \
        } \
    } \
 \
    free(w); \
    free(narr); \
    return true; \
} \

bool Discrete_Fourier_Transform_Naive(uint64_t n, Cnum *arr, int fft_direction)
DFT_NAIVE__ALGORITHM(uint64_t, Cnum, Cnum_Value, Cnum_Add, Cnum_Mul, Cnum_Div_Real,
                     Fast_Fourier_Transform_Weight)

bool Discrete_Fourier_Transform_Naive_c32(uint32_t n, Cnum32 *arr, int fft_direction)
DFT_NAIVE__ALGORITHM(uint32_t, Cnum32, Cnum_Value_c32, Cnum_Add_c32, Cnum_Mul_c32, Cnum_Div_Real_c32,
                     Fast_Fourier_Transform_Weight_c32)

bool Discrete_Fourier_Transform_Naive_cl(uint64_t n, Cnuml *arr, int fft_direction)
DFT_NAIVE__ALGORITHM(uint64_t, Cnuml, Cnum_Value_cl, Cnum_Add_cl, Cnum_Mul_cl, Cnum_Div_Real_cl,
                     Fast_Fourier_Transform_Weight_cl)

#ifdef ENABLE_QUADPRECISION
bool Discrete_Fourier_Transform_Naive_c128(uint64_t n, Cnum128 *arr, int fft_direction)
DFT_NAIVE__ALGORITHM(uint64_t, Cnum128, Cnum_Value_c128, Cnum_Add_c128, Cnum_Mul_c128, Cnum_Div_Real_c128,
                     Fast_Fourier_Transform_Weight_c128)
#endif /* ENABLE_QUADPRECISION */