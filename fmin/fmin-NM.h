/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fmin/fmin-NM.h
Nelder-Mead Search
*/
#ifndef _FMIN_NM_H
#define _FMIN_NM_H

#include<stdlib.h>
#include<stdint.h>

#ifdef ENABLE_QUADPRECISION
#include<quadmath.h>
#endif /* ENABLE_QUADPRECISION */

#define FMIN_NM_FAIL 100
#define FMIN_NM_COMPARE_FAIL 3

/*
n_param is the length of x, nx in the function is n_param+1 (how many x in **start)
*/
/* uint64_t & double */
int Fmin_NM(uint64_t n_param, double **start,
            double func(double *params, void *other_param), void *other_param,
            uint64_t n_step, uint64_t print_start, uint64_t print_step);

/* uint64_t & float */
int Fmin_NM_f32(uint64_t n_param, float **start,
                float func(float *params, void *other_param), void *other_param,
                uint64_t n_step, uint64_t print_start, uint64_t print_step);

/* uint64_t & long double */
int Fmin_NM_fl(uint64_t n_param, long double **start,
               long double func(long double *params, void *other_param), void *other_param,
               uint64_t n_step, uint64_t print_start, uint64_t print_step);

#ifdef ENABLE_QUADPRECISION
/* uint64_t & __float128 */
int Fmin_NM_f128(uint64_t n_param, __float128 **start,
                 __float128 func(__float128 *params, void *other_param), void *other_param,
                 uint64_t n_step, uint64_t print_start, uint64_t print_step);
#endif /* ENABLE_QUADPRECISION */

#endif /* _FMIN_NM_H */
