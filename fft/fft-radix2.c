/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fft/fft-radix2.c
*/
#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>

#include"fft.h"
#include"../basic/basic.h"
/*#include"cnum.h"*/

static inline void FFT_swap(void *a, void *b, size_t usize, void *temp)
{
    memcpy(temp, a, usize);
    memcpy(a, b, usize);
    memcpy(b, temp, usize);
}

#define FFT_RADIX2_MALLOC__ALGORITHM(Cnum) \
{ \
    if (n_element == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n_element = 0.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return NULL; \
    } \
    uint64_t log2_n_ceil = Log2_Ceil(n_element), n_ceil = (uint64_t)1 << log2_n_ceil, i; \
    size_t nsize = n_ceil * sizeof(Cnum); \
    Cnum *ptr = malloc(nsize); \
    if (ptr == NULL) return NULL; \
    for (i=0; i<n_ceil; i++){ \
        ptr[i].real = ptr[i].imag = 0; \
    } \
    return ptr; \
} \

#define FFT_RADIX2_CORE__ALGORITHM(Cnum, Cnum_Mul, Cnum_Add, Cnum_Sub) \
{ \
    if (n_ceil == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n_ceil = 0.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return false; \
    } \
    if (arr == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: array pointer (arr) is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (w == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: weight pointer (w) is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    uint64_t log2_n=Log2_Floor(n_ceil), id, id_reverse, mask=(uint64_t)BIN64 >> (64 - log2_n); \
    Cnum temp; \
    /* copy arr1 -> tarr1 */ \
    for (id=0; id<n_ceil; id++){ \
        union _union64 u64 = {.u=id}; \
        id_reverse = Bit_Reverse_64(u64).u >> (64-log2_n); \
        if (id < id_reverse){ \
            FFT_swap(arr+id, arr+id_reverse, sizeof(Cnum), &temp); \
        } \
    } \
 \
    for (uint64_t len=2; len<=n_ceil; len<<=1){ \
        uint64_t len2 = len >> 1; \
        uint64_t skip_w = n_ceil / len; \
        for (uint64_t i=0; i<n_ceil; i+=len){ \
            uint64_t i_w = 0; \
            for (uint64_t j=0; j<len2; j++, i_w=(i_w+skip_w)&mask){ \
                Cnum u, v; \
                u = arr[i + j]; \
                v = Cnum_Mul(arr[i+j+len2], w[i_w]); \
                arr[i+j] = Cnum_Add(u, v); \
                arr[i+j+len2] = Cnum_Sub(u, v); \
            } \
        } \
    } \
    return true; \
} \

#define FFT_RADIX2__ALGORITHM(Cnum, Fast_Fourier_Transform_w, Fast_Fourier_Transform_Core, Cnum_Div_Real) \
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
    if (fft_direction != MADD_FFT_FORWARD && fft_direction != MADD_FFT_INVERSE){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: fft_direction should be either MADD_FFT_FORWARD or MADD_FFT_INVERSE. You set %d.", __func__, fft_direction); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    uint64_t log2_n_ceil = Log2_Ceil(n), n_ceil = (uint64_t)1 << log2_n_ceil, i; \
 \
    Cnum *w = (Cnum*)malloc(n_ceil * sizeof(Cnum)); \
    if (w == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for weight points.", __func__, n_ceil * sizeof(Cnum)); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    Fast_Fourier_Transform_w(n_ceil, w, fft_direction); \
 \
    /*Cnum cnum_zero = {.real=0, .imag=0};*/ \
    Fast_Fourier_Transform_Core(n_ceil, arr, w); \
 \
    if (fft_direction == MADD_FFT_INVERSE){ \
        for (i=0; i<n_ceil; i++){ \
            arr[i] = Cnum_Div_Real(arr[i], n_ceil); \
        } \
    } \
 \
    free(w); \
    return true; \
} \

/* Cnum */
void *Fast_Fourier_Transform_Radix2_Malloc(uint64_t n_element)
FFT_RADIX2_MALLOC__ALGORITHM(Cnum)

bool Fast_Fourier_Transform_Radix2_Core(uint64_t n_ceil, Cnum *arr, const Cnum *w)
FFT_RADIX2_CORE__ALGORITHM(Cnum, Cnum_Mul, Cnum_Add, Cnum_Sub)

bool Fast_Fourier_Transform_Radix2(uint64_t n, Cnum *arr, int fft_direction)
FFT_RADIX2__ALGORITHM(Cnum, Fast_Fourier_Transform_Weight, Fast_Fourier_Transform_Radix2_Core, Cnum_Div_Real)

/* Cnum_f32 */
void *Fast_Fourier_Transform_Radix2_Malloc_f32(uint64_t n_element)
FFT_RADIX2_MALLOC__ALGORITHM(Cnum32)

bool Fast_Fourier_Transform_Radix2_Core_f32(uint64_t n_ceil, Cnum32 *arr, const Cnum32 *w)
FFT_RADIX2_CORE__ALGORITHM(Cnum32, Cnum_Mul_c32, Cnum_Add_c32, Cnum_Sub_c32)

bool Fast_Fourier_Transform_Radix2_f32(uint64_t n, Cnum32 *arr, int fft_direction)
FFT_RADIX2__ALGORITHM(Cnum32, Fast_Fourier_Transform_Weight_f32, Fast_Fourier_Transform_Radix2_Core_f32, Cnum_Div_Real_c32)

/* Cnum_fl */
void *Fast_Fourier_Transform_Radix2_Malloc_fl(uint64_t n_element)
FFT_RADIX2_MALLOC__ALGORITHM(Cnuml)

bool Fast_Fourier_Transform_Radix2_Core_fl(uint64_t n_ceil, Cnuml *arr, const Cnuml *w)
FFT_RADIX2_CORE__ALGORITHM(Cnuml, Cnum_Mul_cl, Cnum_Add_cl, Cnum_Sub_cl)

bool Fast_Fourier_Transform_Radix2_fl(uint64_t n, Cnuml *arr, int fft_direction)
FFT_RADIX2__ALGORITHM(Cnuml, Fast_Fourier_Transform_Weight_fl, Fast_Fourier_Transform_Radix2_Core_fl, Cnum_Div_Real_cl)

#ifdef ENABLE_QUADPRECISION
/* Cnum_f128 */
void *Fast_Fourier_Transform_Radix2_Malloc_f128(uint64_t n_element)
FFT_RADIX2_MALLOC__ALGORITHM(Cnum128)

bool Fast_Fourier_Transform_Radix2_Core_f128(uint64_t n_ceil, Cnum128 *arr, const Cnum128 *w)
FFT_RADIX2_CORE__ALGORITHM(Cnum128, Cnum_Mul_c128, Cnum_Add_c128, Cnum_Sub_c128)

bool Fast_Fourier_Transform_Radix2_f128(uint64_t n, Cnum128 *arr, int fft_direction)
FFT_RADIX2__ALGORITHM(Cnum128, Fast_Fourier_Transform_Weight_f128, Fast_Fourier_Transform_Radix2_Core_f128, Cnum_Div_Real_c128)
#endif /* ENABLE_QUADPRECISION */
