/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fft/fft.c
*/
#include<stdio.h>
#include<string.h>
#include<math.h>
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

/* This code is suitable for embedded devices */
#define FFT_W_QUAD_SYMMETRY__ALGORITHM(Cnum, float_type, cos, sin) \
{ \
    if (n == 0){ \
        Madd_Error_Add(MADD_ERROR, L"Fast_Fourier_Transform_w: n = 0."); \
        return; \
    } \
    w[0].real = 1; \
    w[0].imag = 0; \
    if (n == 1){ \
        return; \
    } \
    uint64_t n2 = n/2, n4 = n/4, n3=n2+n4 /* n*3/4 */, n8 = n/8, i; \
    float_type angle = sign * 2 * _CONSTANT_PI / n, real, imag, angle_i; \
    uint64_t n_mod_4 = n & 0b11; /* n % 4 */ \
    if (n_mod_4 == 0){ \
        for (i=0; i<n8+1; i++){ \
            angle_i = angle * i; \
            real = cos(angle_i); \
            imag = sin(angle_i); \
            w[i].real    = w[n4+i].imag = w[n4-i].imag = real; \
            w[n2+i].real = w[n2-i].real = w[n3+i].imag = w[n3-i].imag = -real; \
            w[i].imag    = w[n2-i].imag = w[n4-i].real = w[n3+i].real = imag; \
            w[n4+i].real = w[n2+i].imag = w[n3-i].real = -imag; \
            if (i!=0){ \
                w[n-i].real = real; \
                w[n-i].imag = -imag; \
            } \
        } \
    }else if (n_mod_4 == 0b10){ \
        for (i=0; i<n4+1; i++){ \
            angle_i = angle * i; \
            real = cos(angle_i); \
            imag = sin(angle_i); \
            w[i].real    = real; \
            w[n2+i].real = w[n2-i].real = -real; \
            w[i].imag    = w[n2-i].imag = imag; \
            w[n2+i].imag = -imag; \
            if (i!=0){ \
                w[n-i].real = real; \
                w[n-i].imag = -imag; \
            } \
        } \
    }else{ \
        for (i=0; i<n2; i++){ \
            angle_i = angle * i; \
            real = cos(angle_i); \
            imag = sin(angle_i); \
            w[i+1].real = w[n-1-i].real = real; \
            w[i+1].imag = imag; \
            w[n-1-i].imag = -imag; \
        } \
    } \
} \

#define FFT_MALLOC__ALGORITHM(Cnum) \
{ \
    if (n_element == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n_element = 0.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return NULL; \
    } \
    uint64_t log2_n_ceil = Log2_Ceil(n_element), n_ceil = 1 << log2_n_ceil, i; \
    size_t nsize = n_ceil * sizeof(Cnum); \
    Cnum *ptr = malloc(nsize); \
    if (ptr == NULL) return NULL; \
    for (i=0; i<n_ceil; i++){ \
        ptr[i].real = ptr[i].imag = 0; \
    } \
    return ptr; \
} \

#define FFT_W__ALGORITHM(Cnum, Cnum_Value, cos, sin, real_type) \
{ \
    if (n == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n = 0.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return; \
    } \
    if (w == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: weight pointer (w) is NULL."); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return; \
    } \
 \
    w[0] = Cnum_Value(1, 0); \
    real_type real, imag, angle_base = sign * 2 * _CONSTANT_PI / n, angle; \
    uint64_t n2 = n / 2, i; \
    if (n & 0b1){ /* n is odd */ \
        for (i=0; i<n2; i++){ \
            uint64_t id1 = n2 - i, id2 = n2 + i + 1; \
            angle = angle_base * id1; \
            real = cos(angle); \
            imag = sin(angle); \
            w[id1].real = w[id2].real = real; \
            w[id1].imag = imag; \
            w[id2].imag = -imag; \
        } \
    }else{ /* n is even */ \
        w[n2] = Cnum_Value(-1, 0); \
        for (i=1; i<n2; i++){ \
            uint64_t id1 = n2 - i, id2 = n2 + i; \
            angle = angle_base * id1; \
            real = cos(angle); \
            imag = sin(angle); \
            w[id1].real = w[id2].real = real; \
            w[id1].imag = imag; \
            w[id2].imag = -imag; \
        } \
        if (n & 0b11 == 0){ /* n % 4 == 0 */ \
            uint64_t n4 = n >> 2; \
            w[n4].imag = 1; \
            w[3*n4].imag = -1; \
            w[n4].real = w[3*n4].real = 0; \
        } \
    } \
} \

#define FFT_CORE__ALGORITHM(Cnum, Cnum_Mul, Cnum_Add, Cnum_Sub) \
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
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: weight pointer (arr) is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    uint64_t log2_n=Log2_Floor(n_ceil), id, id_reverse, mask=BIN64 >> (64 - log2_n); \
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

#define FFT__ALGORITHM(Cnum, Fast_Fourier_Transform_w, Fast_Fourier_Transform_Core, Cnum_Div_Real) \
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
    if (fft_direction != MADD_FFT_FORWARD && fft_direction != MADD_FFT_INVERSE){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: fft_direction should be either MADD_FFT_FORWARD or MADD_FFT_INVERSE. You set %d.", __func__, fft_direction); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return; \
    } \
 \
    uint64_t log2_n_ceil = Log2_Ceil(n), n_ceil = 1 << log2_n_ceil, i; \
 \
    Cnum *w = (Cnum*)malloc(n_ceil * sizeof(Cnum)); \
    if (w == NULL) return; \
 \
    Fast_Fourier_Transform_w(n_ceil, w, fft_direction); \
 \
    Cnum cnum_zero = {.real=0, .imag=0}; \
    Fast_Fourier_Transform_Core(n_ceil, arr, w); \
 \
    if (fft_direction == MADD_FFT_INVERSE){ \
        for (i=0; i<n_ceil; i++){ \
            arr[i] = Cnum_Div_Real(arr[i], n_ceil); \
        } \
    } \
 \
    free(w); \
} \

/* Cnum */
void *Fast_Fourier_Transform_Malloc(uint64_t n_element)
FFT_MALLOC__ALGORITHM(Cnum)

void Fast_Fourier_Transform_w(uint64_t n, Cnum *w, int sign)
FFT_W__ALGORITHM(Cnum, Cnum_Value, cos, sin, double)

bool Fast_Fourier_Transform_Core(uint64_t n_ceil, Cnum *arr, const Cnum *w)
FFT_CORE__ALGORITHM(Cnum, Cnum_Mul, Cnum_Add, Cnum_Sub)

void Fast_Fourier_Transform(uint64_t n, Cnum *arr, int fft_direction)
FFT__ALGORITHM(Cnum, Fast_Fourier_Transform_w, Fast_Fourier_Transform_Core, Cnum_Div_Real)

/* Cnum_f32 */
void *Fast_Fourier_Transform_Malloc_f32(uint64_t n_element)
FFT_MALLOC__ALGORITHM(Cnum_f32)

void Fast_Fourier_Transform_w_f32(uint64_t n, Cnum_f32 *w, int sign)
FFT_W__ALGORITHM(Cnum_f32, Cnum_Value_f32, cosf, sinf, float)

bool Fast_Fourier_Transform_Core_f32(uint64_t n_ceil, Cnum_f32 *arr, const Cnum_f32 *w)
FFT_CORE__ALGORITHM(Cnum_f32, Cnum_Mul_f32, Cnum_Add_f32, Cnum_Sub_f32)

void Fast_Fourier_Transform_f32(uint64_t n, Cnum_f32 *arr, int fft_direction)
FFT__ALGORITHM(Cnum_f32, Fast_Fourier_Transform_w_f32, Fast_Fourier_Transform_Core_f32, Cnum_Div_Real_f32)

/* Cnum_fl */
void *Fast_Fourier_Transform_Malloc_fl(uint64_t n_element)
FFT_MALLOC__ALGORITHM(Cnum_fl)

void Fast_Fourier_Transform_w_fl(uint64_t n, Cnum_fl *w, int sign)
FFT_W__ALGORITHM(Cnum_fl, Cnum_Value_fl, cosl, sinl, long double)

bool Fast_Fourier_Transform_Core_fl(uint64_t n_ceil, Cnum_fl *arr, const Cnum_fl *w)
FFT_CORE__ALGORITHM(Cnum_fl, Cnum_Mul_fl, Cnum_Add_fl, Cnum_Sub_fl)

void Fast_Fourier_Transform_fl(uint64_t n, Cnum_fl *arr, int fft_direction)
FFT__ALGORITHM(Cnum_fl, Fast_Fourier_Transform_w_fl, Fast_Fourier_Transform_Core_fl, Cnum_Div_Real_fl)

#ifdef ENABLE_QUADPRECISION
/* Cnum_f128 */
void *Fast_Fourier_Transform_Malloc_f128(uint64_t n_element)
FFT_MALLOC__ALGORITHM(Cnum_f128)

void Fast_Fourier_Transform_w_f128(uint64_t n, Cnum_f128 *w, int sign)
FFT_W__ALGORITHM(Cnum_f128, Cnum_Value_f128, cosq, sinq, __float128)

bool Fast_Fourier_Transform_Core_f128(uint64_t n_ceil, Cnum_f128 *arr, const Cnum_f128 *w)
FFT_CORE__ALGORITHM(Cnum_f128, Cnum_Mul_f128, Cnum_Add_f128, Cnum_Sub_f128)

void Fast_Fourier_Transform_f128(uint64_t n, Cnum_f128 *arr, int fft_direction)
FFT__ALGORITHM(Cnum_f128, Fast_Fourier_Transform_w_f128, Fast_Fourier_Transform_Core_f128, Cnum_Div_Real_f128)
#endif /* ENABLE_QUADPRECISION */