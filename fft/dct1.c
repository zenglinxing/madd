/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fft/dct1.c
Discrete Cosine Transform (DCT-I)
*/
#include<stdlib.h>
#include<math.h>
#include<stdint.h>
#include<stdbool.h>

#include"fft.h"
#include"../basic/basic.h"

#define DCT1__ALGORITHM(Cnum, real_type, Madd_Set0_c64, Fast_Fourier_Transform, sqrt) \
{ \
    if (n < 2){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: requires at least 2 elements, but got %llu.", __func__, n); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (arr == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: arr is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    uint64_t n1 = n - 1, N = 2 * n1; \
    Cnum *narr = (Cnum*)malloc(sizeof(Cnum) * N); \
    if (narr == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for narr.", __func__, N*sizeof(Cnum)); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    for (uint64_t i=0; i<n; i++){ \
        narr[i].real = arr[i]; \
        narr[i].imag = 0; \
    } \
    Madd_Set0_c64(N-n, narr+n); \
 \
    Fast_Fourier_Transform(N, narr, MADD_FFT_FORWARD); \
 \
    real_type scale = sqrt((real_type)2 / n1); \
    for (uint64_t k=0; k<n; k++){ \
        real_type real_part = narr[k].real; \
        if (k == 0 || k == n1){ \
            arr[k] = real_part * scale * 0.5; \
        }else{ \
            arr[k] = real_part * scale; \
        } \
    } \
 \
    free(narr); \
    return true; \
} \

bool Discrete_Cosine_Transform_1(uint64_t n, double *arr)
DCT1__ALGORITHM(Cnum, double, Madd_Set0_c64, Fast_Fourier_Transform, sqrt)

bool Discrete_Cosine_Transform_1_f32(uint64_t n, float *arr)
DCT1__ALGORITHM(Cnum32, float, Madd_Set0_c32, Fast_Fourier_Transform_c32, sqrtf)

bool Discrete_Cosine_Transform_1_fl(uint64_t n, long double *arr)
DCT1__ALGORITHM(Cnuml, long double, Madd_Set0_cl, Fast_Fourier_Transform_cl, sqrtl)

#ifdef ENABLE_QUADPRECISION
bool Discrete_Cosine_Transform_1_f128(uint64_t n, __float128 *arr)
DCT1__ALGORITHM(Cnum128, __float128, Madd_Set0_c128, Fast_Fourier_Transform_c128, sqrtq)
#endif /* ENABLE_QUADPRECISION */