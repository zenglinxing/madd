/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./integrate/integrate.h
*/
#ifndef MADD_INTEGRATE_H
#define MADD_INTEGRATE_H

#include<stdint.h>

/* Simpson integrate */
double Integrate_Simpson(double func(double,void*),
                         double xmin, double xmax,
                         uint64_t n,void *other_param);
float Integrate_Simpson_f32(float func(float,void*),
                            float xmin, float xmax,
                            uint32_t n, void *other_param);
long double Integrate_Simpson_fl(long double func(long double,void*),
                                 long double xmin, long double xmax,
                                 uint64_t n, void *other_param);
#ifdef ENABLE_QUADPRECISION
__float128 Int_Simpson_f128(__float128 func(__float128,void*),
                            __float128 xmin, __float128 xmax,
                            uint64_t n, void *other_param);
#endif /* ENABLE_QUADPRECISION */

#endif /* MADD_INTEGRATE_H */