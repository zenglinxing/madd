/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fmin/fmin.h
*/
#ifndef MADD_FMIN_H
#define MADD_FMIN_H

#include<stdint.h>
#include<stdbool.h>
#include"../rng/rng.h"

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

/* Particle Swarm Optimization */
/* uint64_t & double */
void Fmin_PSO(uint64_t n_param, uint64_t n_bird, double **start,
              double func(double *param,void *other_param), void *other_param,
              uint64_t n_step,
              double weight_ini, double weight_end, double c1, double c2,
              double **velocity, double dt, RNG_Param *rng);

/* uint64_t & long double */
void Fmin_PSO_fl(uint64_t n_param, uint64_t n_bird, long double **start,
                 long double func(long double *param,void *other_param), void *other_param,
                 uint64_t n_step,
                 long double weight_ini, long double weight_end, long double c1, long double c2,
                 long double **velocity, long double dt,RNG_Param *rng);

/* uint32_t & float */
void Fmin_PSO_F(uint32_t n_param,uint32_t n_bird, float **start,
                float func(float *param,void *other_param), void *other_param,
                uint32_t n_step,
                float weight_ini, float weight_end, float c1, float c2,
                float **velocity, float dt, RNG_Param *rng);

#ifdef ENABLE_QUADPRECISION
/* uint64_t & __float128 */
void Fmin_PSO_f128(uint64_t n_param, uint64_t n_bird, __float128 **start,
                   __float128 func(__float128 *param,void *other_param), void *other_param,
                   uint64_t n_step,
                   __float128 weight_ini, __float128 weight_end, __float128 c1, __float128 c2,
                   __float128 **velocity, __float128 dt, RNG_Param *rng);
#endif /* ENABLE_QUADPRECISION */

/* Stimulated Annealing */
bool Fmin_SA(uint64_t n_param, double *params,
             double func(double *params, void *other_param), void *other_param,
             double *perturbation_step,
             uint64_t n_step, double temp_start, double temp_end,
             RNG_Param *rng,
             uint64_t print_start, uint64_t print_step);
bool Fmin_SA_f32(uint64_t n_param, float *params,
                 float func(float *params, void *other_param), void *other_param,
                 float *perturbation_step,
                 uint64_t n_step, float temp_start, float temp_end,
                 RNG_Param *rng,
                 uint64_t print_start, uint64_t print_step);
bool Fmin_SA_fl(uint64_t n_param, long double *params,
                long double func(long double *params, void *other_param), void *other_param,
                long double *perturbation_step,
                uint64_t n_step, long double temp_start, long double temp_end,
                RNG_Param *rng,
                uint64_t print_start, uint64_t print_step);
#ifdef ENABLE_QUADPRECISION
bool Fmin_SA_f128(uint64_t n_param, __float128 *params,
                  __float128 func(__float128 *params, void *other_param), void *other_param,
                  __float128 *perturbation_step,
                  uint64_t n_step, __float128 temp_start, __float128 temp_end,
                  RNG_Param *rng,
                  uint64_t print_start, uint64_t print_step);
#endif /* ENABLE_QUADPRECISION */

#endif /* MADD_FMIN_H */
