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

void Inverse_Discrete_Cosine_Transform_2(uint64_t n, double *arr)
{
    if (n == 0){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n = 0.", __func__);
        Madd_Error_Add(MADD_WARNING, error_info);
        return;
    }
    if (arr == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: array pointer (arr) is NULL.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return;
    }
    if (n > UINT64_MAX / sizeof(Cnum)){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n is too large, causing integer overflow.", __func__);
        Madd_Error_Add(MADD_ERROR, error_info);
        return;
    }

    uint64_t i, len_fft = n << 1, log2_n_ceil = Log2_Ceil(len_fft), n_ceil = (uint64_t)1 << log2_n_ceil;
    Cnum *arr_fft = (Cnum*)Fast_Fourier_Transform_Malloc(len_fft);
    if (arr_fft == NULL){
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to allocate %llu bytes for fft.", __func__, n_ceil * sizeof(Cnum));
        Madd_Error_Add(MADD_ERROR, error_info);
        return;
    }
    Cnum *w = (Cnum*)malloc(n_ceil*sizeof(Cnum));
    if (w == NULL){
        free(arr_fft);
        wchar_t error_info[MADD_ERROR_INFO_LEN];
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to allocate %llu bytes for w.", __func__, n_ceil * sizeof(Cnum));
        Madd_Error_Add(MADD_ERROR, error_info);
        return;
    }

    double scale = sqrt(2.0 / n);
    for (i=0; i<n; i++){
        double theta = _CONSTANT_PI * i / (2.0 * n_ceil), c_i = (i) ? 1 : sqrt(0.5);
        Cnum cnum_theta = Cnum_Value(cos(theta), -sin(theta));
        arr_fft[i] = arr_fft[n - 1 - i] = Cnum_Mul_Real(cnum_theta, arr[i] * scale * c_i);
    }
    /*for (i=0; i<n; i++){
        arr_fft[i].real = arr_fft[len_fft - i - 1].real = arr[i] * ((i) ? 1 : sqrt(0.5));
    }*/

    Fast_Fourier_Transform_w(n_ceil, w, MADD_FFT_FORWARD);
    Fast_Fourier_Transform_Core(n_ceil, arr_fft, w);

    /*for (i=0; i<n; i++){
        double theta = _CONSTANT_PI * i / (2.0 * n_ceil);
        double real_part = arr_fft[i].real * cos(theta) + arr_fft[i].imag * sin(theta);
        arr[i] = real_part * scale;
    }*/
    for (i=0; i<n; i++){
        arr[i] = arr_fft[i].real;
    }

    free(arr_fft);
    free(w);
}