/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fmin/Newton-Iteration.c
*/
#include<stdlib.h>
#include<stdint.h>
#include<stdbool.h>
#include"fmin.h"
#include"../basic/basic.h"
#include"../linalg/linalg.h"

#define FMIN_NEWTON_ITERATION__ALGORITHM(num_type, \
                                         Matrix_Inverse, Matrix_Inverse_func_name, \
                                         Matrix_Multiply) \
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
    if (param_steps == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: param_steps is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (n_step == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n_step = 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    size_t size_malloc = (n_param + (uint64_t)n_param*n_param + n_param + n_param) * sizeof(num_type); \
    num_type *param_tent = (num_type*)malloc(size_malloc), *Hessian = param_tent + n_param, *grad = Hessian + (uint64_t)n_param*n_param, *delta_param = grad + n_param; \
    if (param_tent == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for param_tent & Hessian & grad & delta_param.", __func__, size_malloc); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    size_t size_param = (uint64_t)n_param * sizeof(num_type); \
    uint64_t n_add_1 = (uint64_t)n_param + 1; \
    for (uint64_t i_step=0; i_step < n_step; i_step++){ \
        num_type *pH = Hessian; \
        memcpy(param_tent, params, size_param); \
        /* construct Hessian matrix */ \
        for (uint64_t i_param=0; i_param < n_param; i_param++, pH += n_param){ \
            num_type old_x = params[i_param], dx = param_steps[i_param]; \
            for (uint64_t j_param=0; j_param < n_param; j_param++){ \
                num_type old_y = params[j_param], dy = param_steps[j_param]; \
                if (i_param == j_param){ \
                    param_tent[i_param] = old_x + dx; \
                    num_type f1 = func(param_tent, other_param); \
                    param_tent[i_param] = old_x - dx; \
                    num_type f2 = func(param_tent, other_param); \
                    num_type f0 = func(params, other_param); \
                    pH[j_param] = (f1 - 2*f0 + f2) / (dx * dx); \
                }else{ \
                    param_tent[i_param] = old_x + dx; \
                    param_tent[j_param] = old_y + dy; \
                    num_type f1 = func(param_tent, other_param); \
                    param_tent[j_param] = old_y - dy; \
                    num_type f2 = func(param_tent, other_param); \
                    param_tent[i_param] = old_x - dx; \
                    num_type f3 = func(param_tent, other_param); \
                    param_tent[j_param] = old_y + dy; \
                    num_type f4 = func(param_tent, other_param); \
                    pH[j_param] = (f1 - f2 - f4 + f3) / (4 * dx * dy); \
                } \
                param_tent[i_param] = old_x; \
                param_tent[j_param] = old_y; \
            } \
            /* grad */ \
            param_tent[i_param] = old_x + dx; \
            num_type f1 = func(param_tent, other_param); \
            param_tent[i_param] = old_x - dx; \
            num_type f2 = func(param_tent, other_param); \
            grad[i_param] = (f1 - f2) / (2 * dx); \
            param_tent[i_param] = old_x; \
        } \
        /* Hessian + mu * I */ \
        if (norm_term != 0){ \
            num_type *pH = Hessian; \
            for (uint64_t i_param=0; i_param < n_param; i_param++, pH += n_add_1){ \
                *pH += norm_term; \
            } \
        } \
        /* Hessian^-1 */ \
        bool flag_inverse = Matrix_Inverse(n_param, Hessian); \
        if (!flag_inverse){ \
            free(param_tent); \
            wchar_t error_info[MADD_ERROR_INFO_LEN]; \
            swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: see info from %hs.", __func__, Matrix_Inverse_func_name); \
            Madd_Error_Add(MADD_ERROR, error_info); \
            return false; \
        } \
        Matrix_Multiply(n_param, 1, n_param, Hessian, grad, delta_param); \
        /* update params */ \
        for (uint64_t i_param=0; i_param<n_param; i_param++){ \
            params[i_param] -= delta_param[i_param]; \
        } \
    } \
 \
    free(param_tent); \
    return true; \
} \

bool Fmin_Newton_Iteration(int32_t n_param, double *params,
                           double func(double *params, void *other_param), void *other_param,
                           double *param_steps, double norm_term, uint64_t n_step)
FMIN_NEWTON_ITERATION__ALGORITHM(double, Matrix_Inverse, "Matrix_Inverse", Matrix_Multiply)

bool Fmin_Newton_Iteration_f32(int32_t n_param, float *params,
                               float func(float *params, void *other_param), void *other_param,
                               float *param_steps, float norm_term, uint64_t n_step)
FMIN_NEWTON_ITERATION__ALGORITHM(float, Matrix_Inverse_f32, "Matrix_Inverse_f32", Matrix_Multiply_f32)