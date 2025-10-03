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

#include"fft.cuh"
#include"../basic/cnum.h"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

#define MADD_FFT_FORWARD -1
#define MADD_FFT_INVERSE 1

/*
===============================================================================
FFT weight
===============================================================================
*/
void Fast_Fourier_Transform_Weight(uint64_t n, Cnum *w, int sign);
void Fast_Fourier_Transform_Weight_c32(uint64_t n, Cnum32 *w, int sign);
void Fast_Fourier_Transform_Weight_cl(uint64_t n, Cnuml *w, int sign);
#ifdef ENABLE_QUADPRECISION
void Fast_Fourier_Transform_Weight_c128(uint64_t n, Cnum128 *w, int sign);
#endif /* ENABLE_QUADPRECISION */

/*
===============================================================================
Fast Fourier Transform
===============================================================================
*/
bool Fast_Fourier_Transform(uint64_t n, Cnum *arr, int fft_direction);
bool Fast_Fourier_Transform_c32(uint32_t n, Cnum32 *arr, int fft_direction);
bool Fast_Fourier_Transform_cl(uint64_t n, Cnuml *arr, int fft_direction);
#ifdef ENABLE_QUADPRECISION
bool Fast_Fourier_Transform_c128(uint64_t n, Cnum128 *arr, int fft_direction);
#endif /* ENABLE_QUADPRECISION */

/*
FFT Radix-2
*/
/* Cnum */
void *Fast_Fourier_Transform_Radix2_Malloc(uint64_t n_element);
bool Fast_Fourier_Transform_Radix2_Core(uint64_t n_ceil, Cnum *arr, const Cnum *w);
bool Fast_Fourier_Transform_Radix2(uint64_t n, Cnum *arr, int fft_direction);

/* Cnum32 */
void *Fast_Fourier_Transform_Radix2_Malloc_c32(uint64_t n_element);
bool Fast_Fourier_Transform_Radix2_Core_c32(uint64_t n_ceil, Cnum32 *arr, const Cnum32 *w);
bool Fast_Fourier_Transform_Radix2_c32(uint64_t n, Cnum32 *arr, int fft_direction);

/* Cnuml */
void *Fast_Fourier_Transform_Radix2_Malloc_cl(uint64_t n_element);
bool Fast_Fourier_Transform_Radix2_Core_cl(uint64_t n_ceil, Cnuml *arr, const Cnuml *w);
bool Fast_Fourier_Transform_Radix2_cl(uint64_t n, Cnuml *arr, int fft_direction);

#ifdef ENABLE_QUADPRECISION
/* Cnum128 */
void *Fast_Fourier_Transform_Radix2_Malloc_c128(uint64_t n_element);
bool Fast_Fourier_Transform_Radix2_Core_c128(uint64_t n_ceil, Cnum128 *arr, const Cnum128 *w);
bool Fast_Fourier_Transform_Radix2_c128(uint64_t n, Cnum128 *arr, int fft_direction);
#endif /* ENABLE_QUADPRECISION */

bool Fast_Fourier_Transform_Bluestein(uint64_t n, Cnum *arr, int fft_direction);
bool Fast_Fourier_Transform_Bluestein_c32(uint64_t n, Cnum32 *arr, int fft_direction);
bool Fast_Fourier_Transform_Bluestein_cl(uint64_t n, Cnuml *arr, int fft_direction);
#ifdef ENABLE_QUADPRECISION
bool Fast_Fourier_Transform_Bluestein_c128(uint64_t n, Cnum128 *arr, int fft_direction);
#endif /* ENABLE_QUADPRECISION */

bool Discrete_Fourier_Transform_Naive(uint64_t n, Cnum *arr, int fft_direction);
bool Discrete_Fourier_Transform_Naive_c32(uint32_t n, Cnum32 *arr, int fft_direction);
bool Discrete_Fourier_Transform_Naive_cl(uint64_t n, Cnuml *arr, int fft_direction);
#ifdef ENABLE_QUADPRECISION
bool Discrete_Fourier_Transform_Naive_c128(uint64_t n, Cnum128 *arr, int fft_direction);
#endif /* ENABLE_QUADPRECISION */

/*
===============================================================================
Discrete Cosine Transform
===============================================================================
*/
bool Discrete_Cosine_Transform_Weight(uint64_t n, double *w);
bool Discrete_Cosine_Transform_Weight_f32(uint32_t n, float *w);
bool Discrete_Cosine_Transform_Weight_fl(uint64_t n, long double *w);
#ifdef ENABLE_QUADPRECISION
bool Discrete_Cosine_Transform_Weight_f128(uint64_t n, __float128 *w);
#endif /* ENABLE_QUADPRECISION */

/*
===============================================================================
DCT-2
===============================================================================
*/
bool Discrete_Cosine_Transform_2(uint64_t n, double *arr);
bool Discrete_Cosine_Transform_2_f32(uint32_t n, float *arr);
bool Discrete_Cosine_Transform_2_fl(uint64_t n, long double *arr);
#ifdef ENABLE_QUADPRECISION
bool Discrete_Cosine_Transform_2_f128(uint64_t n, __float128 *arr);
#endif /* ENABLE_QUADPRECISION */

void Discrete_Cosine_Transform_2_Radix2(uint64_t n, double *arr);
void Discrete_Cosine_Transform_2_Radix2_f32(uint32_t n, float *arr);
void Discrete_Cosine_Transform_2_Radix2_fl(uint64_t n, long double *arr);
#ifdef ENABLE_QUADPRECISION
void Discrete_Cosine_Transform_2_Radix2_f128(uint64_t n, __float128 *arr);
#endif /* ENABLE_QUADPRECISION */

bool Discrete_Cosine_Transform_2_Naive(uint64_t n, double *arr);
bool Discrete_Cosine_Transform_2_Naive_f32(uint32_t n, float *arr);
bool Discrete_Cosine_Transform_2_Naive_fl(uint64_t n, long double *arr);
#ifdef ENABLE_QUADPRECISION
bool Discrete_Cosine_Transform_2_Naive_f128(uint64_t n, __float128 *arr);
#endif /* ENABLE_QUADPRECISION */

/*
===============================================================================
IDCT-2
===============================================================================
*/
bool Inverse_Discrete_Cosine_Transform_2(uint64_t n, double *arr);
bool Inverse_Discrete_Cosine_Transform_2_f32(uint32_t n, float *arr);
bool Inverse_Discrete_Cosine_Transform_2_fl(uint64_t n, long double *arr);
#ifdef ENABLE_QUADPRECISION
bool Inverse_Discrete_Cosine_Transform_2_f128(uint64_t n, __float128 *arr);
#endif /* ENABLE_QUADPRECISION */

bool Inverse_Discrete_Cosine_Transform_2_Naive(uint64_t n, double *arr);
bool Inverse_Discrete_Cosine_Transform_2_Naive_f32(uint32_t n, float *arr);
bool Inverse_Discrete_Cosine_Transform_2_Naive_fl(uint64_t n, long double *arr);
#ifdef ENABLE_QUADPRECISION
bool Inverse_Discrete_Cosine_Transform_2_Naive_f128(uint64_t n, __float128 *arr);
#endif /* ENABLE_QUADPRECISION */

#endif /* MADD_FFT_H */