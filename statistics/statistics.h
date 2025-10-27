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

#ifdef __CUDACC__
__host__ __device__
#endif /* __CUDACC__ */
inline double Kahan_Summation_Step(double sum, double add, double *compensate)
KAHAN_SUMMATION_STEP__ALGORITHm(double)

#ifdef __CUDACC__
__host__ __device__
#endif /* __CUDACC__ */
inline float Kahan_Summation_Step_f32(float sum, float add, float *compensate)
KAHAN_SUMMATION_STEP__ALGORITHm(float)

#ifdef __CUDACC__
__host__ __device__
#endif /* __CUDACC__ */
inline long double Kahan_Summation_Step_fl(long double sum, long double add, long double *compensate)
KAHAN_SUMMATION_STEP__ALGORITHm(long double)

#ifdef __CUDACC__
__host__ __device__
#endif /* __CUDACC__ */
inline Cnum Kahan_Summation_Step_c64(Cnum sum, Cnum add, Cnum *compensate)
KAHAN_SUMMATION_STEP_CNUM__ALGORITHm(Cnum, Cnum_Sub, Cnum_Add)

#ifdef __CUDACC__
__host__ __device__
#endif /* __CUDACC__ */
inline Cnum32 Kahan_Summation_Step_c32(Cnum32 sum, Cnum32 add, Cnum32 *compensate)
KAHAN_SUMMATION_STEP_CNUM__ALGORITHm(Cnum32, Cnum_Sub_c32, Cnum_Add_c32)

#ifdef __CUDACC__
__host__ __device__
#endif /* __CUDACC__ */
inline Cnuml Kahan_Summation_Step_cl(Cnuml sum, Cnuml add, Cnuml *compensate)
KAHAN_SUMMATION_STEP_CNUM__ALGORITHm(Cnuml, Cnum_Sub_cl, Cnum_Add_cl)

#ifdef ENABLE_QUADPRECISION
inline __float128 Kahan_Summation_Step_f128(__float128 sum, __float128 add, __float128 *compensate)
KAHAN_SUMMATION_STEP__ALGORITHm(__float128)

inline Cnum128 Kahan_Summation_Step_c128(Cnum128 sum, Cnum128 add, Cnum128 *compensate)
KAHAN_SUMMATION_STEP_CNUM__ALGORITHm(Cnum128, Cnum_Sub_c128, Cnum_Add_c128)
#endif /* ENABLE_QUADPRECISION */

double      Kahan_Summation(uint64_t n, double *arr);
float       Kahan_Summation_f32(uint32_t n, float *arr);
long double Kahan_Summation_fl(uint64_t n, long double *arr);
Cnum        Kahan_Summation_c64(uint64_t n, Cnum *arr);
Cnum32      Kahan_Summation_c32(uint64_t n, Cnum32 *arr);
Cnuml       Kahan_Summation_cl(uint64_t n, Cnuml *arr);

#ifdef ENABLE_QUADPRECISION
__float128 Kahan_Summation_f128(uint64_t n, __float128 *arr);
Cnum128 Kahan_Summation_c128(uint64_t n, Cnum128 *arr);
#endif

bool Linspace(double start, double end, uint64_t n, double *arr, bool include_end);

#endif /* MADD_STATISTICS_H */