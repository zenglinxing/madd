/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./basic/norm.c
*/
#include<stdint.h>
#include<math.h>
#include"cnum.h"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

#define NORM1__ALGORITHM(float_type, fabs) \
{ \
    float_type sum = 0, last_sum, loss=0, xx; \
    uint64_t i; \
    for (i=0; i<n; i++){ \
        last_sum = sum; \
        xx = fabs(x[i]); \
        sum += xx + loss; \
        loss = last_sum - sum + xx; \
    } \
    return sum; \
} \

double Norm1(uint64_t n, double *x)
NORM1__ALGORITHM(double, fabs)

float Norm1_f32(uint64_t n, float *x)
NORM1__ALGORITHM(float, fabsf)

long double Norm1_fl(uint64_t n, long double *x)
NORM1__ALGORITHM(long double, fabsl)

double Norm1_c64(uint64_t n, Cnum *x)
NORM1__ALGORITHM(double, Cnum_Radius)

float Norm1_c32(uint64_t n, Cnum32 *x)
NORM1__ALGORITHM(float, Cnum_Radius_c32)

long double Norm1_cl(uint64_t n, Cnuml *x)
NORM1__ALGORITHM(long double, Cnum_Radius_cl)

#ifdef ENABLE_QUADPRECISION
__float128 Norm1_f128(uint64_t n, __float128 *x)
NORM1__ALGORITHM(__float128, fabsq)

__float128 Norm1_c128(uint64_t n, Cnum128 *x)
NORM1__ALGORITHM(__float128, Cnum_Radius_c128)
#endif /* ENABLE_QUADPRECISION */

#define NORM2__ALGORITHM(float_type, sqrt) \
{ \
    float_type sum = 0, last_sum, loss=0, xx; \
    uint64_t i; \
    for (i=0; i<n; i++){ \
        xx = x[i] * x[i]; \
        last_sum = sum; \
        sum += xx; \
        loss = last_sum - sum + xx; \
    } \
    return sqrt(sum); \
} \

#define NORM2_CNUM__ALGORITHM(float_type, sqrt) \
{ \
    float_type sum = 0, last_sum, loss=0, xx; \
    uint64_t i; \
    for (i=0; i<n; i++){ \
        xx = x[i].real*x[i].real + x[i].imag*x[i].imag; \
        last_sum = sum; \
        sum += xx; \
        loss = last_sum - sum + xx; \
    } \
    return sqrt(sum); \
} \

double Norm2(uint64_t n, double *x)
NORM2__ALGORITHM(double, sqrt)

float Norm2_f32(uint64_t n, float *x)
NORM2__ALGORITHM(float, sqrtf)

long double Norm2_fl(uint64_t n, long double *x)
NORM2__ALGORITHM(long double, sqrtl)

double Norm2_c64(uint64_t n, Cnum *x)
NORM2_CNUM__ALGORITHM(double, sqrt)

float Norm2_c32(uint64_t n, Cnum32 *x)
NORM2_CNUM__ALGORITHM(float, sqrtf)

long double Norm2_cl(uint64_t n, Cnuml *x)
NORM2_CNUM__ALGORITHM(long double, sqrtl)

#ifdef ENABLE_QUADPRECISION
__float128 Norm2_f128(uint64_t n, __float128 *x)
NORM2__ALGORITHM(__float128, sqrtq)

__float128 Norm2_c128(uint64_t n, Cnum128 *x)
NORM2_CNUM__ALGORITHM(__float128, sqrtq)
#endif /* ENABLE_QUADPRECISION */

#define NORM_INFINITY__ALGORITHM(float_type, fabs) \
{ \
    float_type max = 0, xx; \
    uint64_t i; \
    for (i=0; i<n; i++){ \
        xx = fabs(x[i]); \
        max = (xx > max) ? xx : max; \
    } \
    return max; \
} \

double Norm_Infinity(uint64_t n, double *x)
NORM_INFINITY__ALGORITHM(double, fabs)

float Norm_Infinity_f32(uint64_t n, float *x)
NORM_INFINITY__ALGORITHM(float, fabsf)

long double Norm_Infinity_fl(uint64_t n, long double *x)
NORM_INFINITY__ALGORITHM(long double, fabsl)

double Norm_Infinity_c64(uint64_t n, Cnum *x)
NORM_INFINITY__ALGORITHM(double, Cnum_Radius)

float Norm_Infinity_c32(uint64_t n, Cnum32 *x)
NORM_INFINITY__ALGORITHM(float, Cnum_Radius_c32)

long double Norm_Infinity_cl(uint64_t n, Cnuml *x)
NORM_INFINITY__ALGORITHM(long double, Cnum_Radius_cl)

#ifdef ENABLE_QUADPRECISION
__float128 Norm_Infinity_f128(uint64_t n, __float128 *x)
NORM_INFINITY__ALGORITHM(__float128, fabsq)

__float128 Norm_Infinity_c128(uint64_t n, Cnum128 *x)
NORM_INFINITY__ALGORITHM(__float128, Cnum_Radius_c128)
#endif /* ENABLE_QUADPRECISION */