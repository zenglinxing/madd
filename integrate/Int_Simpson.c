/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./integrate/int_Simpson.c
*/
#include<math.h>
#include<stdint.h>
#include"integrate.h"

/*#include<stdint.h>*/
/*#include<math.h>*/
#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

#ifndef INTEGRATE_LOG2_VALUE
#define INTEGRATE_LOG2_VALUE 0.6931471805599453
#endif

#define INTEGRATE_SIMPSON__ALGORITHM(num_type, size_num_type, log_func, ceil) \
{ \
    register size_num_type i; \
    register num_type nd=(num_type)n, sum=0, gap=(xmax-xmin)/nd, gap2=gap/2, a=func(xmin,other_param), b, temp,x; \
    num_type xd=xmax-xmin; \
    const size_num_type log2_len=ceil( log_func((num_type)n)/INTEGRATE_LOG2_VALUE ) + 1; \
    num_type s = 0; \
    for (i=0;i<n;i++){ \
        x=xmin+xd*(i+1)/nd; \
        b=func(x, other_param); \
        s += (a+b+4*func(x-gap2, other_param)) * gap/6; \
        temp=a; /* swap */ \
        a=b; \
        b=temp; \
    } \
    return s; \
} \

double Integrate_Simpson(double func(double,void*), double xmin, double xmax,
                         uint64_t n,void *other_param)
INTEGRATE_SIMPSON__ALGORITHM(double, uint64_t, log, ceil)

float Integrate_Simpson_f32(float func(float,void*), float xmin, float xmax,
                            uint32_t n, void *other_param)
INTEGRATE_SIMPSON__ALGORITHM(float, uint32_t, logf, ceilf)

long double Integrate_Simpson_fl(long double func(long double,void*), long double xmin, long double xmax,
                                 uint64_t n, void *other_param)
INTEGRATE_SIMPSON__ALGORITHM(long double, uint64_t, logl, ceill)

#ifdef ENABLE_QUADPRECISION
__float128 Integrate_Simpson_f128(__float128 func(__float128,void*), __float128 xmin, __float128 xmax,
                                  uint64_t n, void *other_param)
INTEGRATE_SIMPSON__ALGORITHM(__float128, uint64_t, logq, ceilq)
#endif /* ENABLE_QUADPRECISION */