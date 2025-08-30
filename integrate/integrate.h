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

/* trapezodial integrate */
double Integrate_Trapeze(double func(double, void*), double xmin, double xmax,
                         uint64_t n, void *other_param);
float Integrate_Trapeze_f32(float func(float, void*), float xmin, float xmax,
                            uint32_t n, void *other_param);
long double Integrate_Trapeze_fl(long double func(long double, void*),long double xmin, long double xmax,
                                 uint64_t n, void *other_param);
#ifdef ENABLE_QUADPRECISION
__float128 Integrate_Trapeze_f128(__float128 func(__float128, void*), __float128 xmin, __float128 xmax,
                                  uint64_t n, void *other_param);
#endif /* ENABLE_QUADPRECISION */

/* Simpson integrate */
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

/* Gauss-Legendre integrate */
bool Integrate_Gauss_Legendre_x(uint64_t n_int_, double *x_int);
bool Integrate_Gauss_Legendre_w(uint64_t n_int, double *x_int, double *w_int);
double Integrate_Gauss_Legendre_via_xw(double func(double, void *), double xmin, double xmax,
                                       uint64_t n_int, void *other_param,
                                       double *x_int, double *w_int);
double Integrate_Gauss_Legendre(double func(double, void *), double xmin, double xmax,
                                uint64_t n_int, void *other_param);

/* uint32_t & float */
bool Integrate_Gauss_Legendre_x_f32(uint32_t n_int_, float *x_int);
bool Integrate_Gauss_Legendre_w_f32(uint32_t n_int, float *x_int, float *w_int);
float Integrate_Gauss_Legendre_via_xw_f32(float func(float, void *), float x1, float x2,
                                          uint32_t n_int, void *other_param,
                                          float *x_int, float *w_int);
float Integrate_Gauss_Legendre_f32(float func(float, void *), float x1, float x2,
                                   uint32_t n_int, void *other_param);

/* Gauss-Laguerre integrate */
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
#endif /* MADD_INTEGRATE_H */