/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./special_func/special_func.h
*/
#ifndef MADD_SPECIAL_FUNC_H
#define MADD_SPECIAL_FUNC_H

#include<stdint.h>
#include"../polynomial/poly1d.h"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

/*
polynomial array returned:
if the polynomial is
    a0 x^0 + a1 x^1 + a2 x^2 + ... + an x^n
returned array will be
[ a0, a1, a2, ... , an, ]
*/

/*
=======================================================================================
Legendire polynomial
*/
void Special_Func_Legendre(Poly1d *poly);
void Special_Func_Legendre_f32(Poly1d_f32 *poly);
void Special_Func_Legendre_fl(Poly1d_fl *poly);

#ifdef ENABLE_QUADPRECISION
void Special_Func_Legendre_f128(Poly1d_f128 *poly);
#endif /* ENABLE_QUADPRECISION */

/* uint64_t & double */
double Special_Func_Legendre_Coefficient_First(uint64_t n);
double Special_Func_Legendre_Coefficient_Iter(uint64_t n, uint64_t i, double previous_coefficient);

/* uint32_t & float */
float Special_Func_Legendre_Coefficient_First_f32(uint32_t n);
float Special_Func_Legendre_Coefficient_Iter_f32(uint32_t n, uint32_t i, float previous_coefficient);

/* uint64_t & long double */
long double Special_Func_Legendre_Coefficient_First_fl(uint64_t n);
long double Special_Func_Legendre_Coefficient_Iter_fl(uint64_t n, uint64_t i, long double previous_coefficient);

#ifdef ENABLE_QUADPRECISION
/* uint64_t & __float128 */
__float128 Special_Func_Legendre_Coefficient_First_f128(uint64_t n);
__float128 Special_Func_Legendre_Coefficient_Iter_f128(uint64_t n, uint64_t i, __float128 previous_coefficient);
#endif /* ENABLE_QUADPRECISION */

double Special_Func_Legendre_Value(uint64_t n, double x);
float Special_Func_Legendre_Value_f32(uint32_t n, float x);
long double Special_Func_Legendre_Value_fl(uint64_t n, long double x);

#ifdef ENABLE_QUADPRECISION
/* uint64_t & __float128 */
__float128 Special_Func_Legendre_Value_f128(uint64_t n, __float128 x);
#endif /* ENABLE_QUADPRECISION */

#endif /* MADD_SPECIAL_FUNC_H */
