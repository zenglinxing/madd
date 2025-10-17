/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./interpolation/interpolation.h
*/
#ifndef MADD_INTERPOLATION_H
#define MADD_INTERPOLATION_H

#include<stdint.h>
#include<stdbool.h>

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

/*
===============================================================================
Lagrange Interpolation
===============================================================================
*/
typedef struct{
    uint64_t n;
    double *x,*y,*denominator;
} Interpolation_Lagrange_Param;

bool Interpolation_Lagrange_Init(uint64_t n, double *x, double *y, Interpolation_Lagrange_Param *ilp);
void Interpolation_Lagrange_Free(Interpolation_Lagrange_Param *ilp);
double Interpolation_Lagrange_Value_Internal(double x, const Interpolation_Lagrange_Param *ilp, double *xx);
double Interpolation_Lagrange_Value(double x, const Interpolation_Lagrange_Param *ilp);

typedef struct{
    uint64_t n;
    float *x,*y,*denominator;
} Interpolation_Lagrange_Param_f32;

bool Interpolation_Lagrange_Init_f32(uint64_t n, float *x, float *y, Interpolation_Lagrange_Param_f32 *ilp);
void Interpolation_Lagrange_Free_f32(Interpolation_Lagrange_Param_f32 *ilp);
float Interpolation_Lagrange_Value_Internal_f32(float x, const Interpolation_Lagrange_Param_f32 *ilp, float *xx);
float Interpolation_Lagrange_Value_f32(float x, const Interpolation_Lagrange_Param_f32 *ilp);

typedef struct{
    uint64_t n;
    long double *x,*y,*denominator;
} Interpolation_Lagrange_Param_fl;

bool Interpolation_Lagrange_Init_fl(uint64_t n, long double *x, long double *y, Interpolation_Lagrange_Param_fl *ilp);
void Interpolation_Lagrange_Free_fl(Interpolation_Lagrange_Param_fl *ilp);
long double Interpolation_Lagrange_Value_Internal_fl(long double x, const Interpolation_Lagrange_Param_fl *ilp, long double *xx);
long double Interpolation_Lagrange_Value_fl(long double x, const Interpolation_Lagrange_Param_fl *ilp);

#ifdef ENABLE_QUADPRECISION
typedef struct{
    uint64_t n;
    __float128 *x,*y,*denominator;
} Interpolation_Lagrange_Param_f128;

bool Interpolation_Lagrange_Init_f128(uint64_t n, __float128 *x, __float128 *y, Interpolation_Lagrange_Param_f128 *ilp);
void Interpolation_Lagrange_Free_f128(Interpolation_Lagrange_Param_f128 *ilp);
__float128 Interpolation_Lagrange_Value_Internal_f128(__float128 x, const Interpolation_Lagrange_Param_f128 *ilp, __float128 *xx);
__float128 Interpolation_Lagrange_Value_f128(__float128 x, const Interpolation_Lagrange_Param_f128 *ilp);
#endif /* ENABLE_QUADPRECISION */

/*
===============================================================================
Cubic Spline Interpolation
===============================================================================
*/
typedef struct{
    uint64_t n;
    double *x, *a, *b, *c, *d;
} Interpolation_Cubic_Spline_Param;

bool Interpolation_Cubic_Spline_Init(uint64_t n, const double *x, const double *y, Interpolation_Cubic_Spline_Param *icsp);

#endif /* MADD_INTERPOLATION_H */