/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./statistics/statistics.h
*/
#ifndef MADD_STATISTICS_H
#define MADD_STATISTICS_H

#include<stdint.h>

#include"../basic/cnum.h"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

#define KAHAN_SUMMATION_STEP__ALGORITHm(num_type) \
{ \
    num_type y = add - *compensate, t = sum + y; \
    *compensate = t - sum - y; \
    return t; \
} \

#define KAHAN_SUMMATION_STEP_CNUM__ALGORITHm(num_type, Cnum_Sub, Cnum_Add) \
{ \
    num_type y = Cnum_Sub(add, *compensate), t = Cnum_Add(sum, y); \
    *compensate = Cnum_Sub(Cnum_Sub(t, sum), y); \
    return t; \
} \

inline double Kahan_Summation_Step(double sum, double add, double *compensate)
KAHAN_SUMMATION_STEP__ALGORITHm(double)

inline float Kahan_Summation_Step_f32(float sum, float add, float *compensate)
KAHAN_SUMMATION_STEP__ALGORITHm(float)

inline long double Kahan_Summation_Step_fl(long double sum, long double add, long double *compensate)
KAHAN_SUMMATION_STEP__ALGORITHm(long double)

inline Cnum Kahan_Summation_Step_c(Cnum sum, Cnum add, Cnum *compensate)
KAHAN_SUMMATION_STEP_CNUM__ALGORITHm(Cnum, Cnum_Sub, Cnum_Add)

inline Cnum_f32 Kahan_Summation_Step_c32(Cnum_f32 sum, Cnum_f32 add, Cnum_f32 *compensate)
KAHAN_SUMMATION_STEP_CNUM__ALGORITHm(Cnum_f32, Cnum_Sub_f32, Cnum_Add_f32)

inline Cnum_fl Kahan_Summation_Step_cl(Cnum_fl sum, Cnum_fl add, Cnum_fl *compensate)
KAHAN_SUMMATION_STEP_CNUM__ALGORITHm(Cnum_fl, Cnum_Sub_fl, Cnum_Add_fl)

#ifdef ENABLE_QUADPRECISION
inline __float128 Kahan_Summation_Step_f128(__float128 sum, __float128 add, __float128 *compensate)
KAHAN_SUMMATION_STEP__ALGORITHm(__float128)

inline Cnum_f128 Kahan_Summation_Step_c128(Cnum_f128 sum, Cnum_f128 add, Cnum_f128 *compensate)
KAHAN_SUMMATION_STEP_CNUM__ALGORITHm(Cnum_f128, Cnum_Sub_f128, Cnum_Add_f128)
#endif /* ENABLE_QUADPRECISION */

double      Kahan_Summation(uint64_t n, double *arr);
float       Kahan_Summation_f32(uint32_t n, float *arr);
long double Kahan_Summation_fl(uint64_t n, long double *arr);
Cnum        Kahan_Summation_c(uint64_t n, Cnum *arr);
Cnum_f32    Kahan_Summation_c32(uint64_t n, Cnum_f32 *arr);
Cnum_fl     Kahan_Summation_cl(uint64_t n, Cnum_fl *arr);

#endif /* MADD_STATISTICS_H */