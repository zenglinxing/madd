/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fft/fft.h
*/
#ifndef MADD_FFT_H
#define MADD_FFT_H

#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>

#include"../basic/cnum.h"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

#define MADD_FFT_FORWARD -1
#define MADD_FFT_INVERSE 1

/* Cnum */
void *Fast_Fourier_Transform_Malloc(uint64_t n_element);
void Fast_Fourier_Transform_w(uint64_t n, Cnum *w, int sign);
bool Fast_Fourier_Transform_Core(uint64_t n_ceil, Cnum *arr, const Cnum *w);
void Fast_Fourier_Transform(uint64_t n, Cnum *arr, int fft_direction);

/* Cnum_f32 */
void *Fast_Fourier_Transform_Malloc_f32(uint64_t n_element);
void Fast_Fourier_Transform_w_f32(uint64_t n, Cnum32 *w, int sign);
bool Fast_Fourier_Transform_Core_f32(uint64_t n_ceil, Cnum32 *arr, const Cnum32 *w);
void Fast_Fourier_Transform_f32(uint64_t n, Cnum32 *arr, int fft_direction);

/* Cnum_fl */
void *Fast_Fourier_Transform_Malloc_fl(uint64_t n_element);
void Fast_Fourier_Transform_w_fl(uint64_t n, Cnuml *w, int sign);
bool Fast_Fourier_Transform_Core_fl(uint64_t n_ceil, Cnuml *arr, const Cnuml *w);
void Fast_Fourier_Transform_fl(uint64_t n, Cnuml *arr, int fft_direction);

#ifdef ENABLE_QUADPRECISION
/* Cnum_f128 */
void *Fast_Fourier_Transform_Malloc_f128(uint64_t n_element);
void Fast_Fourier_Transform_w_f128(uint64_t n, Cnum128 *w, int sign);
bool Fast_Fourier_Transform_Core_f128(uint64_t n_ceil, Cnum128 *arr, const Cnum128 *w);
void Fast_Fourier_Transform_f128(uint64_t n, Cnum128 *arr, int fft_direction);
#endif /* ENABLE_QUADPRECISION */

void Discrete_Cosine_Transform_2(uint64_t n, double *arr);
void Discrete_Cosine_Transform_2_f32(uint64_t n, float *arr);
void Discrete_Cosine_Transform_2_fl(uint64_t n, long double *arr);
#ifdef ENABLE_QUADPRECISION
void Discrete_Cosine_Transform_2_f128(uint64_t n, __float128 *arr);
#endif /* ENABLE_QUADPRECISION */

#endif /* MADD_FFT_H */