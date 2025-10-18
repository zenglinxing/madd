/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./integrate/integrate.h
*/
#ifndef MADD_INTEGRATE_H
#define MADD_INTEGRATE_H

#include<stdint.h>
#include<stdbool.h>

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

/*
===============================================================================
trapezoidal integrate
===============================================================================
*/
double Integrate_Trapeze(double func(double, void*), double xmin, double xmax,
                         uint64_t n_int, void *other_param);
float Integrate_Trapeze_f32(float func(float, void*), float xmin, float xmax,
                            uint32_t n_int, void *other_param);
long double Integrate_Trapeze_fl(long double func(long double, void*),long double xmin, long double xmax,
                                 uint64_t n_int, void *other_param);
#ifdef ENABLE_QUADPRECISION
__float128 Integrate_Trapeze_f128(__float128 func(__float128, void*), __float128 xmin, __float128 xmax,
                                  uint64_t n_int, void *other_param);
#endif /* ENABLE_QUADPRECISION */

/*
===============================================================================
Simpson integrate
===============================================================================
*/
double Integrate_Simpson(double func(double,void*),
                         double xmin, double xmax,
                         uint64_t n_int,void *other_param);
float Integrate_Simpson_f32(float func(float,void*),
                            float xmin, float xmax,
                            uint32_t n_int, void *other_param);
long double Integrate_Simpson_fl(long double func(long double,void*),
                                 long double xmin, long double xmax,
                                 uint64_t n_int, void *other_param);
#ifdef ENABLE_QUADPRECISION
__float128 Integrate_Simpson_f128(__float128 func(__float128,void*),
                                  __float128 xmin, __float128 xmax,
                                  uint64_t n_int, void *other_param);
#endif /* ENABLE_QUADPRECISION */

/*
===============================================================================
Gauss-Legendre integrate
===============================================================================
*/
bool Integrate_Gauss_Legendre_x(int32_t n_int_, double *x_int);
bool Integrate_Gauss_Legendre_w(int32_t n_int, double *x_int, double *w_int);
double Integrate_Gauss_Legendre_via_xw(double func(double, void *), double xmin, double xmax,
                                       int32_t n_int, void *other_param,
                                       double *x_int, double *w_int);
double Integrate_Gauss_Legendre(double func(double, void *), double xmin, double xmax,
                                int32_t n_int, void *other_param);

/* float */
bool Integrate_Gauss_Legendre_x_f32(int32_t n_int_, float *x_int);
bool Integrate_Gauss_Legendre_w_f32(int32_t n_int, float *x_int, float *w_int);
float Integrate_Gauss_Legendre_via_xw_f32(float func(float, void *), float x1, float x2,
                                          int32_t n_int, void *other_param,
                                          float *x_int, float *w_int);
float Integrate_Gauss_Legendre_f32(float func(float, void *), float x1, float x2,
                                   int32_t n_int, void *other_param);

/*
===============================================================================
Gauss-Laguerre integrate
===============================================================================
*/
/*
suppose h(x) = func(x) * exp(-exp_index * x)
return \int_{xmin}^{$\infty$} h(x) dx
*/
/* uint64_t & double */
bool Integrate_Gauss_Laguerre_xw(uint64_t n_int, double *x_int, double *w_int);
double Integrate_Gauss_Laguerre_via_xw(double func(double, void*), double xmin, double exp_index,
                                       uint64_t n_int, void *other_param,
                                       double *x_int, double *w_int);
double Integrate_Gauss_Laguerre(double func(double, void*), double xmin, double exp_index,
                                uint64_t n_int, void *other_param);
/* uint32_t & float */
bool Integrate_Gauss_Laguerre_xw_f32(uint32_t n_int, float *x_int, float *w_int);
float Integrate_Gauss_Laguerre_via_xw_f32(float func(float, void*), float xmin, float exp_index,
                                          uint32_t n_int, void *other_param,
                                          float *x_int, float *w_int);
float Integrate_Gauss_Laguerre_f32(float func(float, void*), float xmin, float exp_index,
                                   uint32_t n_int, void *other_param);

/*
===============================================================================
Clenshaw-Curtis integrate
===============================================================================
*/
bool Integrate_Clenshaw_Curtis_x(uint64_t n_int, double *x_int);
bool Integrate_Clenshaw_Curtis_w(uint64_t n_int, double *w_int);
double Integrate_Clenshaw_Curtis_via_xw(double func(double, void *), double xmin, double xmax,
                                        uint64_t n_int, void *other_param,
                                        double *x_int, double *w_int);
double Integrate_Clenshaw_Curtis(double func(double, void *), double xmin, double xmax,
                                 uint64_t n_int, void *other_param);

/* uint32_t & float */
bool Integrate_Clenshaw_Curtis_x_f32(uint32_t n_int, float *x_int);
bool Integrate_Clenshaw_Curtis_w_f32(uint32_t n_int, float *w_int);
float Integrate_Clenshaw_Curtis_via_xw_f32(float func(float, void *), float xmin, float xmax,
                                           uint32_t n_int, void *other_param,
                                           float *x_int, float *w_int);
float Integrate_Clenshaw_Curtis_f32(float func(float, void *), float xmin, float xmax,
                                    uint32_t n_int, void *other_param);

/* uint64_t & long double */
bool Integrate_Clenshaw_Curtis_x_fl(uint64_t n_int, long double *x_int);
bool Integrate_Clenshaw_Curtis_w_fl(uint64_t n_int, long double *w_int);
long double Integrate_Clenshaw_Curtis_via_xw_fl(long double func(long double, void *), long double xmin, long double xmax,
                                                uint64_t n_int, void *other_param,
                                                long double *x_int, long double *w_int);
long double Integrate_Clenshaw_Curtis_fl(long double func(long double, void *), long double xmin, long double xmax,
                                         uint64_t n_int, void *other_param);

#ifdef ENABLE_QUADPRECISION
/* uint64_t & __float128 */
bool Integrate_Clenshaw_Curtis_x_f128(uint64_t n_int, __float128 *x_int);
bool Integrate_Clenshaw_Curtis_w_f128(uint64_t n_int, __float128 *w_int);
__float128 Integrate_Clenshaw_Curtis_via_xw_f128(__float128 func(__float128, void *), __float128 xmin, __float128 xmax,
                                                 uint64_t n_int, void *other_param,
                                                 __float128 *x_int, __float128 *w_int);
__float128 Integrate_Clenshaw_Curtis_f128(__float128 func(__float128, void *), __float128 xmin, __float128 xmax,
                                          uint64_t n_int, void *other_param);
#endif /* ENABLE_QUADPRECISION */

#endif /* MADD_INTEGRATE_H */