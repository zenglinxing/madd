/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./interpolation/Lagrange.c
*/
#include<stdlib.h>

#include"interpolation.h"
#include"../basic/basic.h"

#define INTERPOLATION_LAGRANGE_INIT__ALGORITHM(num_type) \
{ \
    if (n == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n = 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (x == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: x is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (y == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: y is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (ilp == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: ilp is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    size_t size_cpy = (uint64_t)n * sizeof(num_type); \
    ilp->n = n; \
    ilp->x = (num_type*)malloc(3 * size_cpy); \
    if (ilp->x == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for ilp's x & y & denominator.", __func__, 3 * size_cpy); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    ilp->y = ilp->x + n; \
    ilp->denominator = ilp->y + n; \
 \
    memcpy(ilp->x, x, size_cpy); \
    memcpy(ilp->y, y, size_cpy); \
    for (uint64_t i=0; i<n; i++){ \
        num_type denominator = 1; \
        for (uint64_t j=0; j<n; j++){ \
            if (i==j) continue; \
            num_type diff = x[i] - x[j]; \
            if (fabs(diff) < 1e-15){ \
                wchar_t error_info[MADD_ERROR_INFO_LEN]; \
                swprintf(error_info, MADD_ERROR_INFO_LEN, \
                         L"%hs: Nodes too close: x[%llu] and x[%llu]", __func__, i, j); \
                Madd_Error_Add(MADD_WARNING, error_info); \
            } \
            denominator *= diff; \
        } \
        ilp->denominator[i] = denominator; \
    } \
    return true; \
} \

#define INTERPOLATION_LAGRANGE_FREE__ALGORITHM \
{ \
    if (ilp == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: ilp is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return; \
    } \
    if (ilp->x == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: ilp->x is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return; \
    } \
    free(ilp->x); \
} \

#define INTERPOLATION_LAGRANGE_VALUE_INTERNAL__ALGORITHM(num_type) \
{ \
    if (ilp == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: ilp is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return 0; \
    } \
    if (xx == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: xx is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return 0; \
    } \
 \
    num_type res, k, *numerator=xx+ilp->n; \
    uint64_t i; \
    for (i=0,k=1; i<ilp->n; i++){ \
        numerator[i] = k; \
        xx[i] = x - ilp->x[i]; \
        k *= xx[i]; \
    } \
    for (i=ilp->n-1,k=1,res=0; i>=0; i--){ \
        numerator[i] *= k; \
        k *= xx[i]; \
        res += numerator[i] * ilp->y[i] / ilp->denominator[i]; \
        if (i==0) break; \
    } \
    return res; \
} \

#define INTERPOLATION_LAGRANGE_VALUE__ALGORITHM(num_type, Interpolation_Lagrange_Value_Internal) \
{ \
    if (ilp == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: ilp is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return 0; \
    } \
 \
    size_t size_float = (uint64_t)ilp->n*sizeof(num_type); \
    num_type *xx = (num_type*)malloc(size_float), res; \
    if (xx == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for xx.", __func__, size_float); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return 0; \
    } \
    res = Interpolation_Lagrange_Value_Internal(x, ilp, xx); \
    free(xx); \
    return res; \
} \

/* double */
bool Interpolation_Lagrange_Init(uint64_t n, double *x, double *y, Interpolation_Lagrange_Param *ilp)
INTERPOLATION_LAGRANGE_INIT__ALGORITHM(double)

void Interpolation_Lagrange_Free(Interpolation_Lagrange_Param *ilp)
INTERPOLATION_LAGRANGE_FREE__ALGORITHM

double Interpolation_Lagrange_Value_Internal(double x, const Interpolation_Lagrange_Param *ilp, double *xx)
INTERPOLATION_LAGRANGE_VALUE_INTERNAL__ALGORITHM(double)

double Interpolation_Lagrange_Value(double x, const Interpolation_Lagrange_Param *ilp)
INTERPOLATION_LAGRANGE_VALUE__ALGORITHM(double, Interpolation_Lagrange_Value_Internal)

/* float */
bool Interpolation_Lagrange_Init_f32(uint64_t n, float *x, float *y, Interpolation_Lagrange_Param_f32 *ilp)
INTERPOLATION_LAGRANGE_INIT__ALGORITHM(float)

void Interpolation_Lagrange_Free_f32(Interpolation_Lagrange_Param_f32 *ilp)
INTERPOLATION_LAGRANGE_FREE__ALGORITHM

float Interpolation_Lagrange_Value_Internal_f32(float x, const Interpolation_Lagrange_Param_f32 *ilp, float *xx)
INTERPOLATION_LAGRANGE_VALUE_INTERNAL__ALGORITHM(float)

float Interpolation_Lagrange_Value_f32(float x, const Interpolation_Lagrange_Param_f32 *ilp)
INTERPOLATION_LAGRANGE_VALUE__ALGORITHM(float, Interpolation_Lagrange_Value_Internal_f32)

/* long double */
bool Interpolation_Lagrange_Init_fl(uint64_t n, long double *x, long double *y, Interpolation_Lagrange_Param_fl *ilp)
INTERPOLATION_LAGRANGE_INIT__ALGORITHM(long double)

void Interpolation_Lagrange_Free_fl(Interpolation_Lagrange_Param_fl *ilp)
INTERPOLATION_LAGRANGE_FREE__ALGORITHM

long double Interpolation_Lagrange_Value_Internal_fl(long double x, const Interpolation_Lagrange_Param_fl *ilp, long double *xx)
INTERPOLATION_LAGRANGE_VALUE_INTERNAL__ALGORITHM(long double)

long double Interpolation_Lagrange_Value_fl(long double x, const Interpolation_Lagrange_Param_fl *ilp)
INTERPOLATION_LAGRANGE_VALUE__ALGORITHM(long double, Interpolation_Lagrange_Value_Internal_fl)

#ifdef ENABLE_QUADPRECISION
/* __float128 */
bool Interpolation_Lagrange_Init_f128(uint64_t n, __float128 *x, __float128 *y, Interpolation_Lagrange_Param_f128 *ilp)
INTERPOLATION_LAGRANGE_INIT__ALGORITHM(__float128)

void Interpolation_Lagrange_Free_f128(Interpolation_Lagrange_Param_f128 *ilp)
INTERPOLATION_LAGRANGE_FREE__ALGORITHM

__float128 Interpolation_Lagrange_Value_Internal_f128(__float128 x, const Interpolation_Lagrange_Param_f128 *ilp, __float128 *xx)
INTERPOLATION_LAGRANGE_VALUE_INTERNAL__ALGORITHM(__float128)

__float128 Interpolation_Lagrange_Value_f128(__float128 x, const Interpolation_Lagrange_Param_f128 *ilp)
INTERPOLATION_LAGRANGE_VALUE__ALGORITHM(__float128, Interpolation_Lagrange_Value_Internal_f128)
#endif /* ENABLE_QUADPRECISION */