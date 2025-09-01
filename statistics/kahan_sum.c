/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./statistics/kahan_sum.c
*/
#include<stdint.h>
#include"statistics.h"
/*#include"../basic/cnum.h"*/ /* already in statistics.h */

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif

#define KAHAN_SUMMATION__ALGORITHM(num_type, integer_type, Kahan_Summation_Step) \
{ \
    num_type sum = 0, compensate = 0; \
    for (integer_type i=0; i<n; i++){ \
        sum = Kahan_Summation_Step(sum, arr[i], &compensate); \
    } \
    return sum; \
} \

#define KAHAN_SUMMATION_CNUM__ALGORITHM(num_type, integer_type, Kahan_Summation_Step) \
{ \
    num_type sum = {.real=0, .imag=0}, compensate = {.real=0, .imag=0}; \
    for (integer_type i=0; i<n; i++){ \
        sum = Kahan_Summation_Step(sum, arr[i], &compensate); \
    } \
    return sum; \
} \

double Kahan_Summation(uint64_t n, double *arr)
KAHAN_SUMMATION__ALGORITHM(double, uint64_t, Kahan_Summation_Step)

float Kahan_Summation_f32(uint32_t n, float *arr)
KAHAN_SUMMATION__ALGORITHM(float, uint32_t, Kahan_Summation_Step_f32)

long double Kahan_Summation_fl(uint64_t n, long double *arr)
KAHAN_SUMMATION__ALGORITHM(long double, uint64_t, Kahan_Summation_Step_fl)

Cnum Kahan_Summation_c(uint64_t n, Cnum *arr)
KAHAN_SUMMATION_CNUM__ALGORITHM(Cnum, uint64_t, Kahan_Summation_Step_c)

Cnum_f32 Kahan_Summation_c32(uint64_t n, Cnum_f32 *arr)
KAHAN_SUMMATION_CNUM__ALGORITHM(Cnum_f32, uint64_t, Kahan_Summation_Step_c32)

Cnum_fl Kahan_Summation_cl(uint64_t n, Cnum_fl *arr)
KAHAN_SUMMATION_CNUM__ALGORITHM(Cnum_fl, uint64_t, Kahan_Summation_Step_cl)

#ifdef ENABLE_QUADPRECISION
__float128 Kahan_Summation_f128(uint64_t n, __float128 *arr)
KAHAN_SUMMATION__ALGORITHM(__float128, uint64_t, Kahan_Summation_Step_f128)

Cnum_f128 Kahan_Summation_c128(uint64_t n, Cnum_f128 *arr)
KAHAN_SUMMATION_CNUM__ALGORITHM(Cnum_f128, uint64_t, Kahan_Summation_Step_c128)
#endif