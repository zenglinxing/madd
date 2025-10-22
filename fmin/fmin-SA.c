/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fmin/fmin-SA.h
Stimulated Annealing Minimization.
*/
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<string.h>
#include<stdint.h>
#include<stdbool.h>
#include"fmin.h"
#include"../basic/basic.h"
#include"../rng/rng.h"

#define FMIN_SA__ALGORITHM(num_type, fabs, exp, Rand, temp_check) \
{ \
    if (n_param == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n_param = 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (params == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: params is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (func == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: func is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (perturbation_step == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: perturbation_step is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (n_step < 2){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n_step should be greater than 1.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (temp_start <= 0 || temp_end < 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: temp_start <= 0 or temp_end < 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    num_type d_temp = temp_end - temp_start; \
    if (fabs(d_temp)/fabs(temp_start) < temp_check){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: temp_start and temp_end are too close.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    size_t size_cpy = n_param * sizeof(num_type); \
    num_type y_prev = func(params, other_param); \
    num_type *param_tent = (num_type *)malloc(size_cpy); \
    if (param_tent == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for param_tent.", __func__, size_cpy); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    for (uint64_t i_step = 0; i_step < n_step; i_step++){ \
        for (uint64_t i_param = 0; i_param < n_param; i_param++){ \
            num_type perturbation = (Rand(rng) * 2 - 1) * perturbation_step[i_param]; \
            param_tent[i_param] = params[i_param] + perturbation; \
        } \
        num_type y_tent = func(param_tent, other_param); \
        num_type dy = y_tent - y_prev; \
        if (dy < 0){ \
            memcpy(params, param_tent, size_cpy); \
            y_prev = y_tent; \
        }else{ \
            num_type temp = temp_start + i_step * d_temp / (n_step - 1); \
            num_type prob_accept = exp(-dy / temp); \
            num_type r = Rand(rng); \
            if (r < prob_accept){ \
                memcpy(params, param_tent, size_cpy); \
                y_prev = y_tent; \
            } \
        } \
        if (print_step && i_step >= print_start && (i_step - print_start) % print_step == 0){ \
            printf("Step %llu\tTemp %.6e\tFunc Value %.6e\n", i_step, (double)(temp_start + i_step * d_temp / (n_step - 1)), (double)y_prev); \
            for (uint64_t i_param = 0; i_param < n_param; i_param++){ \
                printf("\tParam[%llu]: %.6e\n", i_param, (double)params[i_param]); \
            } \
        } \
    } \
 \
    free(param_tent); \
    return true; \
} \

bool Fmin_SA(uint64_t n_param, double *params,
             double func(double *params, void *other_param), void *other_param,
             double *perturbation_step,
             uint64_t n_step, double temp_start, double temp_end,
             RNG_Param *rng,
             uint64_t print_start, uint64_t print_step)
FMIN_SA__ALGORITHM(double, fabs, exp, Rand, 1e-12)

bool Fmin_SA_f32(uint64_t n_param, float *params,
                 float func(float *params, void *other_param), void *other_param,
                 float *perturbation_step,
                 uint64_t n_step, float temp_start, float temp_end,
                 RNG_Param *rng,
                 uint64_t print_start, uint64_t print_step)
FMIN_SA__ALGORITHM(float, fabsf, expf, Rand_f32, 1e-6)

bool Fmin_SA_fl(uint64_t n_param, long double *params,
                long double func(long double *params, void *other_param), void *other_param,
                long double *perturbation_step,
                uint64_t n_step, long double temp_start, long double temp_end,
                RNG_Param *rng,
                uint64_t print_start, uint64_t print_step)
FMIN_SA__ALGORITHM(long double, fabsl, expl, Rand_fl, 1e-12)

#ifdef ENABLE_QUADPRECISION
bool Fmin_SA_f128(uint64_t n_param, __float128 *params,
                  __float128 func(__float128 *params, void *other_param), void *other_param,
                  __float128 *perturbation_step,
                  uint64_t n_step, __float128 temp_start, __float128 temp_end,
                  RNG_Param *rng,
                  uint64_t print_start, uint64_t print_step)
FMIN_SA__ALGORITHM(__float128, fabsq, expq, Rand_f128, 1e-30)
#endif /* ENABLE_QUADPRECISION */