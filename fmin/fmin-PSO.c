/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fmin/fmin-PSO.c
Particle Swarm Optimization
*/
#include<stdlib.h>
#include<string.h>
#include<stdint.h>
#include<wchar.h>
#include"fmin.h"
#include"../basic/basic.h"
/*#include"../rng/rng.h"*/ /* already included in fmin.h */

#define FMIN_PSO_BIRD(size_num_type,num_type) \
{ \
    num_type y,*x,pbest,*pbest_x,*gbest,*velocity; \
    size_num_type id; \
}; \

#define FMIN_PSO__ALGORITHM(size_num_type, num_type, Fmin_PSO_Bird, Rand) \
{ \
    size_t size_cpy=(uint64_t)n_param*sizeof(num_type); \
    register uint64_t i_bird,i_param; \
    num_type gbest=func(start[0],other_param),*gbest_x=(num_type*)malloc(size_cpy); \
    if (gbest_x == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes.", __func__, size_cpy); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return; \
    } \
    /* allocate the birds */ \
    num_type *x_list=(num_type*)malloc(n_bird*size_cpy); \
    if (x_list == NULL){ \
        free(gbest_x); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes.", __func__, n_bird*size_cpy); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return; \
    } \
    struct Fmin_PSO_Bird *birds=(struct Fmin_PSO_Bird*)malloc(n_bird*sizeof(struct Fmin_PSO_Bird)); \
    if (birds == NULL){ \
        free(gbest_x); \
        free(x_list); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes.", __func__, n_bird*sizeof(struct Fmin_PSO_Bird)); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return; \
    } \
    register struct Fmin_PSO_Bird *bird; \
    for (i_bird=0,bird=birds; i_bird<n_bird; i_bird++,bird++){ \
        bird->id = i_bird; \
        bird->pbest_x = start[i_bird]; \
        bird->x = x_list+i_bird*n_param; \
        memcpy(bird->x, start[i_bird], size_cpy); \
        bird->pbest = bird->y = func(start[i_bird],other_param); \
        bird->gbest = &gbest; \
        if (bird->y < gbest){ \
            gbest=bird->y; \
            memcpy(gbest_x, start[i_bird], size_cpy); \
        } \
        bird->velocity = velocity[i_bird]; \
    } \
    /* start searching */ \
    register size_num_type i_step; \
    register num_type *bird_x,*bird_velocity,weight,weight_end_ini__n_step=(weight_ini-weight_end)/n_step; \
    for (i_step=0; i_step<n_step; i_step++){ \
        weight = weight_end_ini__n_step/(n_step-i_step)+weight_end; \
        for (i_bird=0; i_bird<n_bird; i_bird++){ \
            bird=birds+i_bird; \
            /* update displacement & velocity */ \
            bird_x=bird->x; \
            bird_velocity=bird->velocity; \
            for (i_param=0; i_param<n_param; i_param++){ \
                bird_x[i_param] += bird_velocity[i_param]*dt; \
                bird_velocity[i_param] = weight*bird_velocity[i_param] /* self direction */ \
                                         + c1*Rand(rng)*(bird->pbest_x[i_param]-bird_x[i_param]) /* toward self historical minimum */ \
                                         + c2*Rand(rng)*(gbest_x[i_param]-bird_x[i_param]); /* toward global historical minimum */ \
            } \
            /* update y */ \
            bird->y = func(bird_x,other_param); \
            if (bird->y < bird->pbest){ \
                bird->pbest = bird->y; \
                memcpy(bird->pbest_x, bird_x, size_cpy); \
            } \
            if (bird->y < gbest){ \
                gbest = bird->y; \
                memcpy(gbest_x, bird_x, size_cpy); \
            } \
        } \
    } \
    free(x_list); \
    free(birds); \
    free(gbest_x); \
} \

/* uint64_t & double */
struct Fmin_PSO_Bird
FMIN_PSO_BIRD(uint64_t, double)

void Fmin_PSO(uint64_t n_param, uint64_t n_bird, double **start,
              double func(double *param,void *other_param), void *other_param,
              uint64_t n_step,
              double weight_ini, double weight_end, double c1, double c2,
              double **velocity, double dt, RNG_Param *rng)
FMIN_PSO__ALGORITHM(uint32_t, double, Fmin_PSO_Bird, Rand)

/* uint64_t & long double */
struct Fmin_PSO_Bird_fl
FMIN_PSO_BIRD(uint64_t,long double)

void Fmin_PSO_fl(uint64_t n_param, uint64_t n_bird, long double **start,
                 long double func(long double *param,void *other_param), void *other_param,
                 uint64_t n_step,
                 long double weight_ini, long double weight_end, long double c1, long double c2,
                 long double **velocity, long double dt,RNG_Param *rng)
FMIN_PSO__ALGORITHM(uint64_t, long double, Fmin_PSO_Bird_fl, Rand_fl)

/* uint32_t & float */
struct Fmin_PSO_Bird_f32
FMIN_PSO_BIRD(uint32_t,float)

void Fmin_PSO_F(uint32_t n_param,uint32_t n_bird, float **start,
                float func(float *param,void *other_param), void *other_param,
                uint32_t n_step,
                float weight_ini, float weight_end, float c1, float c2,
                float **velocity, float dt, RNG_Param *rng)
FMIN_PSO__ALGORITHM(uint32_t, float, Fmin_PSO_Bird_f32,Rand_f32)

#ifdef ENABLE_QUADPRECISION
/* uint64_t & __float128 */
struct Fmin_PSO_Bird_f128
FMIN_PSO_BIRD(uint64_t,__float128)

void Fmin_PSO_f128(uint64_t n_param, uint64_t n_bird, __float128 **start,
                   __float128 func(__float128 *param,void *other_param), void *other_param,
                   uint64_t n_step,
                   __float128 weight_ini, __float128 weight_end, __float128 c1, __float128 c2,
                   __float128 **velocity, __float128 dt, RNG_Param *rng)
FMIN_PSO__ALGORITHM(uint64_t, __float128, Fmin_PSO_Bird_f128, Rand_f128)
#endif /* ENABLE_QUADPRECISION */