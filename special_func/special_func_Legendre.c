/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./special_func/special_func_Legendre.c
*/
#include<string.h>
#include<stdint.h>

#include"special_func.h"
#include"../polynomial/poly1d.h"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

#define SPECIAL_FUNC_LEGENDRE__ALGORITHM(num_type,size_num_type) \
{ \
    memset(poly->mem, 0, ((uint64_t)poly->n+poly->_n+1)*sizeof(num_type)); \
    register size_num_type i; \
    register uint64_t i2; \
    register num_type rate_global=1.,rate_n=1.; \
    /* calculate the coefficient of x^n */ \
    for (i=0,i2=1; i<poly->n; i++,i2+=2){ \
        rate_global *= i2; \
        rate_n *= i+1; \
    } \
    rate_global /= rate_n; \
    const size_num_type n2=poly->n>>1, n=poly->n; \
    register size_num_type n_2i; \
    register uint64_t n_i_2; \
    register num_type *a=poly->a+poly->n, rate; \
    *a = rate_global; \
    if (poly->n > 0) a[-1] = 0.; \
    a-=2; \
    n_2i=n; \
    n_i_2=((uint64_t)n<<1)-1; \
    for (i=1; i<=n2; i++,a-=2,n_2i-=2,n_i_2-=2){ \
        if (poly->n&0b1 || i!=n2) a[-1]=0.; \
        /*rate = - 0.5 * (n-2*i+1.)*(n-2*i+2.) / ( i* (2*n-2*i+1.) );*/ \
        rate = - 0.5 * (n_2i-1.)*n_2i / ( i * (num_type)n_i_2 ); \
        rate_global *= rate; \
        *a = rate_global; \
    } \
} \

void Special_Func_Legendre(Poly1d *poly)
SPECIAL_FUNC_LEGENDRE__ALGORITHM(double, uint64_t)

void Special_Func_Legendre_f32(Poly1d_f32 *poly)
SPECIAL_FUNC_LEGENDRE__ALGORITHM(float, uint32_t)

void Special_Func_Legendre_fl(Poly1d_fl *poly)
SPECIAL_FUNC_LEGENDRE__ALGORITHM(long double, uint64_t)

#ifdef ENABLE_QUADPRECISION
void Special_Func_Legendre_f128(Poly1d_f128 *poly)
SPECIAL_FUNC_LEGENDRE__ALGORITHM(__float128, uint64_t)
#endif /* ENABLE_QUADPRECISION */

#define SPECIAL_FUNC_LEGENDRE_COEFFICIENT_FIRST__ALGORITHM(num_type,size_num_type) \
{ \
    register size_num_type i; \
    register uint64_t i2; \
    register num_type rate_global=1.,rate_n=1.; \
    /* calculate the coefficient of x^n */ \
    for (i=0,i2=1; i<n; i++,i2+=2){ \
        rate_global *= i2; \
        rate_n *= i+1; \
    } \
    rate_global /= rate_n; \
    return rate_global; \
} \

#define SPECIAL_FUNC_LEGENDRE_COEFFICIENT_ITER__ALGORITHM(num_type,size_num_type) \
{ \
    double rate = - 0.5 * (n-2*i+1.)*(n-2*i+2.) / ( i* (2*n-2*i+1.) ); \
    return rate * previous_coefficient; \
} \

/* uint64_t & double */
double Special_Func_Legendre_Coefficient_First(uint64_t n)
SPECIAL_FUNC_LEGENDRE_COEFFICIENT_FIRST__ALGORITHM(double, uint64_t)

double Special_Func_Legendre_Coefficient_Iter(uint64_t n, uint64_t i, double previous_coefficient)
SPECIAL_FUNC_LEGENDRE_COEFFICIENT_ITER__ALGORITHM(double, uint64_t)

/* uint32_t & float */
float Special_Func_Legendre_Coefficient_First_f32(uint32_t n)
SPECIAL_FUNC_LEGENDRE_COEFFICIENT_FIRST__ALGORITHM(float, uint32_t)

float Special_Func_Legendre_Coefficient_Iter_f32(uint32_t n, uint32_t i, float previous_coefficient)
SPECIAL_FUNC_LEGENDRE_COEFFICIENT_ITER__ALGORITHM(float, uint32_t)

/* uint64_t & long double */
long double Special_Func_Legendre_Coefficient_First_fl(uint64_t n)
SPECIAL_FUNC_LEGENDRE_COEFFICIENT_FIRST__ALGORITHM(long double, uint64_t)

long double Special_Func_Legendre_Coefficient_Iter_fl(uint64_t n, uint64_t i, long double previous_coefficient)
SPECIAL_FUNC_LEGENDRE_COEFFICIENT_ITER__ALGORITHM(long double, uint64_t)

#ifdef ENABLE_QUADPRECISION
/* uint64_t & __float128 */
__float128 Special_Func_Legendre_Coefficient_First_f128(uint64_t n)
SPECIAL_FUNC_LEGENDRE_COEFFICIENT_FIRST__ALGORITHM(__float128, uint64_t)

__float128 Special_Func_Legendre_Coefficient_Iter_f128(uint64_t n, uint64_t i, __float128 previous_coefficient)
SPECIAL_FUNC_LEGENDRE_COEFFICIENT_ITER__ALGORITHM(__float128, uint64_t)
#endif /* ENABLE_QUADPRECISION */

#define SPECIAL_FUNC_LEGENDRE_VALUE__ALGORITHM(size_num_type, num_type) \
{ \
    size_num_type i; \
    num_type p0=1, p1=x, p2; \
    if (n==0) return p0; \
    else if (n==1) return p1; \
    for (i=1; i<n; i++){ \
        p2 = ((2.*i+1)*x*p1 - i*p0)/(i+1); \
        p0 = p1; \
        p1 = p2; \
    } \
    return p1; \
} \

double Special_Func_Legendre_Value(uint64_t n, double x)
SPECIAL_FUNC_LEGENDRE_VALUE__ALGORITHM(uint64_t, double)

float Special_Func_Legendre_Value_f32(uint32_t n, float x)
SPECIAL_FUNC_LEGENDRE_VALUE__ALGORITHM(uint32_t, float)

long double Special_Func_Legendre_Value_fl(uint64_t n, long double x)
SPECIAL_FUNC_LEGENDRE_VALUE__ALGORITHM(uint64_t, long double)

#ifdef ENABLE_QUADPRECISION
/* uint64_t & __float128 */
__float128 Special_Func_Legendre_Value_f128(uint64_t n, __float128 x)
SPECIAL_FUNC_LEGENDRE_VALUE__ALGORITHM(uint64_t, __float128)
#endif /* ENABLE_QUADPRECISION */