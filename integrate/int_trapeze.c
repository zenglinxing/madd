/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./integrate/int_trapeze.c
*/
#include<stdint.h>

#include"../basic/basic.h"

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

#define INTEGRATE_TRAPEZE__ALGORITHM(num_type, integer_type) \
{ \
    if (n == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given n_int is 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return 0; \
    } \
    register integer_type i; \
    register num_type nd=(num_type)n, sum=0, gap=(xmax-xmin)/nd, gap2=gap/2, a=func(xmin, other_param), b; \
    /*num_type xd=xmax-xmin;*/ \
    for (i=0; i<n; i++){ \
        b=func(xmin+gap*(i+1), other_param); \
        sum += (a + b) * gap2; \
        a = b; \
    } \
    return sum; \
} \

double Integrate_Trapeze(double func(double, void*), double xmin, double xmax,
                         uint64_t n, void *other_param)
INTEGRATE_TRAPEZE__ALGORITHM(double, uint64_t)

float Integrate_Trapeze_f32(float func(float, void*), float xmin, float xmax,
                            uint32_t n, void *other_param)
INTEGRATE_TRAPEZE__ALGORITHM(float, uint32_t)

long double Integrate_Trapeze_fl(long double func(long double, void*),long double xmin, long double xmax,
                                 uint64_t n, void *other_param)
INTEGRATE_TRAPEZE__ALGORITHM(long double,uint64_t)

#ifdef ENABLE_QUADPRECISION
__float128 Integrate_Trapeze_f128(__float128 func(__float128, void*), __float128 xmin, __float128 xmax,
                                  uint64_t n, void *other_param)
INTEGRATE_TRAPEZE__ALGORITHM(__float128, uint64_t)
#endif /* ENABLE_QUADPRECISION */