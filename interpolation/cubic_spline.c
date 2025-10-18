/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./interpolation/Lagrange.c
*/
#include<stdlib.h>
#include<string.h>

#include"interpolation.h"
#include"../basic/basic.h"
#include"../linalg/linalg.h"
#include"../sort/sort.h"

#define INTERPOLATION_CUBIC_SPLINE_PAIR_COMPARE_ASCENDING__ALGORITHM(Interpolation_Cubic_Spline_Pair) \
{ \
    Interpolation_Cubic_Spline_Pair *p1 = (Interpolation_Cubic_Spline_Pair*)a; \
    Interpolation_Cubic_Spline_Pair *p2 = (Interpolation_Cubic_Spline_Pair*)b; \
    if (p1->x < p2->x) return MADD_LESS; \
    else if (p1->x > p2->x) return MADD_GREATER; \
    else return MADD_SAME; \
} \

#define INTERPOLATION_CUBIC_SPLINE_INIT__ALGORITHM(num_type, Interpolation_Cubic_Spline_Pair, \
                                                   Interpolation_Cubic_Spline_Pair_Compare_Ascending, \
                                                   Linear_Equations_Tridiagonal, \
                                                   Linear_Equations_Tridiagonal_func_name) \
{ \
    if (n < 2){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n < 2.", __func__); \
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
    if (icsp == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: icsp is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    uint64_t n1 = n - 1, i; \
    icsp->n = n; \
    icsp->x = (num_type*)malloc(((uint64_t)n1 * 4 + n) * sizeof(num_type)); \
    if (icsp->x == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for icsp.", __func__, ((uint64_t)n1 * 4 + n) * sizeof(num_type)); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    icsp->a = icsp->x + n; \
    icsp->b = icsp->a + n1; \
    icsp->c = icsp->b + n1; \
    icsp->d = icsp->c + n1; \
    Interpolation_Cubic_Spline_Pair *pairs = (Interpolation_Cubic_Spline_Pair*)malloc((uint64_t)n * sizeof(Interpolation_Cubic_Spline_Pair)); \
    if (pairs == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for pairs.", __func__, (uint64_t)n * sizeof(Interpolation_Cubic_Spline_Pair)); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        free(icsp->x); \
        return false; \
    } \
    for (i=0; i<n; i++){ \
        pairs[i].x = x[i]; \
        pairs[i].y = y[i]; \
    } \
    Sort(n, sizeof(num_type), pairs, Interpolation_Cubic_Spline_Pair_Compare_Ascending, NULL); \
    for (i=0; i<n; i++){ \
        icsp->x[i] = pairs[i].x; \
    } \
 \
    num_type *delta = (num_type*)malloc((n1*4 + n*3) * sizeof(num_type)), *Delta = delta + n1, *vec = Delta + n1, *lower = vec + n, *diag = lower + n1, *upper = diag + n; \
    if (delta == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for delta & Delta & vec & lower & diag & upper.", __func__, (n1*4 + n*3) * sizeof(num_type)); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        free(icsp->x); \
        free(pairs); \
        return false; \
    } \
    /* get *a, *delta, *Delta */ \
    for (i=0; i<n1; i++){ \
        icsp->a[i] = pairs[i].y; \
        delta[i] = icsp->x[i+1] - icsp->x[i]; \
        Delta[i] = pairs[i+1].y - pairs[i].y; \
    } \
    /* get tridiagnal matrix param and *vec */ \
    vec[0] = vec[n1] = upper[0] = lower[n-2] = 0; \
    diag[0] = diag[n1] = 1; \
    for (i=1; i<n1; i++){ \
        lower[i-1] = delta[i-1]; \
        diag[i] = 2*(delta[i-1] + delta[i]); \
        upper[i] = delta[i]; \
        vec[i] = 3*(Delta[i]/delta[i] - Delta[i-1]/delta[i-1]); \
    } \
    /* solve *c */ \
    bool ret_let = Linear_Equations_Tridiagonal(n, lower, diag, upper, 1, vec); \
    if (!ret_let){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: see info from %hs.", __func__, Linear_Equations_Tridiagonal_func_name); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        free(icsp->x); \
        free(pairs); \
        free(delta); \
        return false; \
    } \
    memcpy(icsp->c, vec, ((uint64_t)n1*sizeof(num_type))); \
    /* *d and *b */ \
    for (i=0; i<n1; i++){ \
        icsp->d[i] = (vec[i+1]-vec[i])/(3*delta[i]); \
        icsp->b[i] = Delta[i]/delta[i] - delta[i]*(2*vec[i]+vec[i+1])/3; \
    } \
 \
    free(pairs); \
    free(delta); \
    return true; \
} \

#define INTERPOLATION_CUBIC_SPLINE_FREE__ALGORITHM \
{ \
    if (icsp == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: icsp is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return; \
    } \
    if (icsp->x != NULL) free(icsp->x); \
    icsp->n = 0; \
    icsp->x = icsp->a = icsp->b = icsp->c = icsp->d = NULL; \
} \

#define INTERPOLATION_CUBIC_SPLINE_VALUE__ALGORITHM(num_type, Sort_Compare_Ascending_f64) \
{ \
    if (icsp == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: icsp is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return 0.0; \
    } \
    uint64_t n = icsp->n; \
    if (n < 2){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: icsp->n < 2.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return 0.0; \
    } \
 \
    uint64_t i; \
    if (x <= icsp->x[0]){ \
        i = 0; \
    } \
    else if (x >= icsp->x[n-1]){ \
        i = n - 2; \
    } \
    else{ \
        /* binary search */ \
        i = Binary_Search_Insert(icsp->n, sizeof(num_type), icsp->x, &x, Sort_Compare_Ascending_f64, NULL); \
        i --; \
    } \
 \
    num_type dx = x - icsp->x[i], sum = icsp->a[i], dxx = dx; \
    sum += icsp->b[i]*dxx; \
    dxx *= dx; \
    sum += icsp->c[i]*dxx; \
    dxx *= dx; \
    sum += icsp->d[i]*dxx; \
    return sum; \
} \

/* double */
typedef struct{
    double x, y;
} Interpolation_Cubic_Spline_Pair;

static char Interpolation_Cubic_Spline_Pair_Compare_Ascending(void *a, void *b, void *other_param)
INTERPOLATION_CUBIC_SPLINE_PAIR_COMPARE_ASCENDING__ALGORITHM(Interpolation_Cubic_Spline_Pair)

bool Interpolation_Cubic_Spline_Init(int32_t n, const double *x, const double *y, Interpolation_Cubic_Spline_Param *icsp)
INTERPOLATION_CUBIC_SPLINE_INIT__ALGORITHM(double, Interpolation_Cubic_Spline_Pair,
                                           Interpolation_Cubic_Spline_Pair_Compare_Ascending,
                                           Linear_Equations_Tridiagonal,
                                           "Linear_Equations_Tridiagonal")

void Interpolation_Cubic_Spline_Free(Interpolation_Cubic_Spline_Param *icsp)
INTERPOLATION_CUBIC_SPLINE_FREE__ALGORITHM

double Interpolation_Cubic_Spline_Value(double x, const Interpolation_Cubic_Spline_Param *icsp)
INTERPOLATION_CUBIC_SPLINE_VALUE__ALGORITHM(double, Sort_Compare_Ascending_f64)

/* float */
typedef struct{
    float x, y;
} Interpolation_Cubic_Spline_Pair_f32;

static char Interpolation_Cubic_Spline_Pair_Compare_Ascending_f32(void *a, void *b, void *other_param)
INTERPOLATION_CUBIC_SPLINE_PAIR_COMPARE_ASCENDING__ALGORITHM(Interpolation_Cubic_Spline_Pair_f32)

bool Interpolation_Cubic_Spline_Init_f32(int32_t n, const float *x, const float *y, Interpolation_Cubic_Spline_Param_f32 *icsp)
INTERPOLATION_CUBIC_SPLINE_INIT__ALGORITHM(float, Interpolation_Cubic_Spline_Pair_f32,
                                           Interpolation_Cubic_Spline_Pair_Compare_Ascending_f32,
                                           Linear_Equations_Tridiagonal_f32,
                                           "Linear_Equations_Tridiagonal_f32")

void Interpolation_Cubic_Spline_Free_f32(Interpolation_Cubic_Spline_Param_f32 *icsp)
INTERPOLATION_CUBIC_SPLINE_FREE__ALGORITHM

float Interpolation_Cubic_Spline_Value_f32(float x, const Interpolation_Cubic_Spline_Param_f32 *icsp)
INTERPOLATION_CUBIC_SPLINE_VALUE__ALGORITHM(float, Sort_Compare_Ascending_f32)