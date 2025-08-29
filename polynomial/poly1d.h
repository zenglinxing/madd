/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./polynomial/poly1d.h
*/
#ifndef MADD_POLY1D_H
#define MADD_POLY1D_H

#include<stdlib.h>
#include<string.h>
#include<stdint.h>

#include"../basic/cnum.h"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

typedef struct{
    uint64_t n, _n;
    double *a, *mem;
} Poly1d;

typedef struct{
    uint32_t n, _n;
    float *a, *mem;
} Poly1d_f32;

typedef struct{
    uint64_t n, _n;
    long double *a, *mem;
} Poly1d_fl;

Poly1d     Poly1d_Create(uint64_t n, uint64_t _n);
Poly1d_f32 Poly1d_Create_f32(uint32_t n, uint32_t _n);
Poly1d_fl  Poly1d_Create_fl(uint64_t n, uint64_t _n);

Poly1d     Poly1d_Init(uint64_t n, uint64_t _n, double *a);
Poly1d_f32 Poly1d_Init_f32(uint32_t n, uint32_t _n, float *a);
Poly1d_fl  Poly1d_Init_fl(uint64_t n, uint64_t _n, long double *a);

void Poly1d_Free(Poly1d *poly);
void Poly1d_Free_f32(Poly1d_f32 *poly);
void Poly1d_Free_fl(Poly1d_fl *poly);

double      Poly1d_Value(double x,Poly1d *poly);
float       Poly1d_Value_f32(float x,Poly1d_f32 *poly);
long double Poly1d_Value_fl(long double x,Poly1d_fl *poly);

void Poly1d_Derivative(Poly1d *poly, Poly1d *dpoly);
void Poly1d_Derivative_f32(Poly1d_f32 *poly, Poly1d_f32 *dpoly);
void Poly1d_Derivative_fl(Poly1d_fl *poly, Poly1d_fl *dpoly);

void Poly1d_Derivative_N_order_Allocated(const Poly1d *poly, uint64_t n_order, Poly1d *dpoly);

Poly1d Poly1d_Derivative_N_order(const Poly1d *poly, uint64_t n_order);

double      Poly1d_Integrate(Poly1d *poly,Poly1d *ipoly);
float       Poly1d_Integrate_f32(Poly1d_f32 *poly,Poly1d_f32 *ipoly);
long double Poly1d_Integrate_fl(Poly1d_fl *poly,Poly1d_fl *ipoly);

double      Poly1d_NIntegrate(Poly1d *poly, double xmin, double xmax);
float       Poly1d_NIntegrate_f32(Poly1d_f32 *poly, float xmin, float xmax);
long double Poly1d_NIntegrate_fl(Poly1d_fl *poly, long double xmin, long double xmax);

#ifdef ENABLE_QUADPRECISION
/* uint64_t & __float128 */
typedef struct{
    uint64_t n, _n;
    __float128 *a, *mem;
} Poly1d_f128;

Poly1d_f128 Poly1d_Create_f128(uint64_t n, uint64_t _n);

Poly1d_f128 Poly1d_Init_f128(uint64_t n, uint64_t _n, __float128 *a);

void Poly1d_Free_f128(Poly1d_f128 *poly);

__float128 Poly1d_Value_f128(__float128 x,Poly1d_f128 *poly);

void Poly1d_Derivative_f128(Poly1d_f128 *poly,Poly1d_f128 *dpoly);

__float128 Poly1d_Integrate_f128(Poly1d_f128 *poly, Poly1d_f128 *ipoly);

__float128 Poly1d_NIntegrate_f128(Poly1d_f128 *poly, __float128 xmin, __float128 xmax);
#endif /* ENABLE_QUADPRECISION */

/* uint64_t & cnum */
typedef struct{
    uint64_t n, _n;
    Cnum *a, *mem;
} Poly1d_c;

typedef struct{
    uint64_t n, _n;
    Cnum_f32 *a, *mem;
} Poly1d_c32;

typedef struct{
    uint64_t n, _n;
    Cnum_fl *a, *mem;
} Poly1d_cl;

Poly1d_c   Poly1d_Create_c(uint64_t n, uint64_t _n);
Poly1d_c32 Poly1d_Create_c32(uint64_t n, uint64_t _n);
Poly1d_cl  Poly1d_Create_cl(uint64_t n, uint64_t _n);

Poly1d_c    Poly1d_Init_c(uint64_t n, uint64_t _n, Cnum *a);
Poly1d_c32  Poly1d_Init_c32(uint64_t n, uint64_t _n,Cnum_f32 *a);
Poly1d_cl   Poly1d_Init_cl(uint64_t n, uint64_t _n,Cnum_fl *a);

void Poly1d_Free_c(Poly1d_c *poly);
void Poly1d_Free_c32(Poly1d_c32 *poly);
void Poly1d_Free_cl(Poly1d_cl *poly);

Cnum     Poly1d_Value_c(Cnum x, Poly1d_c *poly);
Cnum_f32 Poly1d_Value_c32(Cnum_f32 x, Poly1d_c32 *poly);
Cnum_fl  Poly1d_Value_cl(Cnum_fl x, Poly1d_cl *poly);

void Poly1d_Derivative_c(Poly1d_c *poly, Poly1d_c *dpoly);
void Poly1d_Derivative_c32(Poly1d_c32 *poly, Poly1d_c32 *dpoly);
void Poly1d_Derivative_cl(Poly1d_cl *poly, Poly1d_cl *dpoly);

#ifdef ENABLE_QUADPRECISION
/* uint64_t & __complex128 */
typedef struct{
    uint64_t n, _n;
    Cnum_f128 *a, *mem;
} Poly1d_c128;

Poly1d_c128 Poly1d_Create_c128(uint64_t n,uint64_t _n);

Poly1d_c128 Poly1d_Init_c128(uint64_t n,uint64_t _n, Cnum_f128 *a);

void Poly1d_Free_c128(Poly1d_c128 *poly);

Cnum_f128 Poly1d_Value_c128(Cnum_f128 x, Poly1d_c128 *poly);

void Poly1d_Derivative_c128(Poly1d_c128 *poly, Poly1d_c128 *dpoly);
#endif /* ENABLE_QUADPRECISION */

#endif /* MADD_POLYN1D_H */
