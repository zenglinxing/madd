/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./integrate/int_Clenshaw_Curtis.c
*/
#include<stdlib.h>
#include<math.h>
#include<stdint.h>
#include<stdbool.h>

#include"integrate.h"
#include"../basic/basic.h"
#include"../fft/fft.h"

#define INTEGRATE_CLENSHAW_CURTIS_X__ALGORITHM(integer_type, real_type, cos) \
{ \
    if (n_int == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given n_int is 0.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return false; \
    } \
    if (x_int == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given x_int is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (n_int == 1){ \
        x_int[0] = 0; \
        return true; \
    } \
 \
    const real_type rate = _CONSTANT_PI / (2 * (real_type)n_int); \
    for (integer_type i = 0; i < n_int; i++){ \
        x_int[i] = -cos( ((i<<1)+1) * rate ); \
    } \
    return true; \
} \

#define INTEGRATE_CLENSHAW_CURTIS_W__ALGORITHM(integer_type, real_type, \
                                               Inverse_Discrete_Cosine_Transform_2, sqrt) \
{ \
    if (n_int == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given n_int is 0.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return false; \
    } \
    if (w_int == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given w_int is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    w_int[0] = sqrt(2); \
    if (n_int == 1){ \
        return true; \
    } \
    w_int[1] = 0; \
 \
    integer_type i; \
    for (i=2; i<n_int; i++){ \
        w_int[i] = (1 + ((i &0b1) ? -1 : 1)) / (1 - (real_type)i*i); \
    } \
    Inverse_Discrete_Cosine_Transform_2(n_int, w_int); \
    real_type scale = sqrt(2/(real_type)n_int); \
    for (i=0; i<n_int; i++){ \
        w_int[i] *= scale; \
    } \
 \
    return true; \
} \

#define INTEGRATE_CLENSHAW_CURTIS_VIA_XW__ALGORITHM(integer_type, real_type) \
{ \
    if (n_int == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given n_int is 0.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return 0; \
    } \
    if (x_int == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given x_int is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return 0; \
    } \
    if (w_int == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given w_int is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return 0; \
    } \
 \
    real_type x_mod = (xmax - xmin) / 2; \
    real_type x_mid = (xmax + xmin) / 2; \
 \
    real_type s = 0; \
    for (integer_type i = 0; i < n_int; i++){ \
        s += w_int[i] * func(x_mod * x_int[i] + x_mid, other_param); \
    } \
    s *= x_mod; \
 \
    return s; \
} \

#define INTEGRATE_CLENSHAW_CURTIS__ALGORITHM(real_type, \
                                             Integrate_Clenshaw_Curtis_x, \
                                             Integrate_Clenshaw_Curtis_w, \
                                             Integrate_Clenshaw_Curtis_via_xw) \
{ \
    if (n_int == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: given n_int is 0.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return 0; \
    } \
 \
    real_type *x_int = (real_type*)malloc(n_int * sizeof(real_type)); \
    if (x_int == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to allocate %llu bytes for x_int.", __func__, n_int * sizeof(real_type)); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return 0; \
    } \
    Integrate_Clenshaw_Curtis_x(n_int, x_int); \
 \
    real_type *w_int = (real_type*)malloc(n_int * sizeof(real_type)); \
    if (w_int == NULL){ \
        free(x_int); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to allocate %llu bytes for w_int.", __func__, n_int * sizeof(real_type)); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return 0; \
    } \
    bool flag_get_w = Integrate_Clenshaw_Curtis_w(n_int, w_int); \
    if (!flag_get_w){ \
        free(x_int); \
        free(w_int); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: see error info from %hs.", __func__, "Integrate_Clenshaw_Curtis_w"); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return 0; \
    } \
 \
    real_type res = Integrate_Clenshaw_Curtis_via_xw(func, xmin, xmax, n_int, other_param, x_int, w_int); \
 \
    free(x_int); \
    free(w_int); \
    return res; \
} \

/* uint64_t & double */
bool Integrate_Clenshaw_Curtis_x(uint64_t n_int, double *x_int)
INTEGRATE_CLENSHAW_CURTIS_X__ALGORITHM(uint64_t, double, cos)

bool Integrate_Clenshaw_Curtis_w(uint64_t n_int, double *w_int)
INTEGRATE_CLENSHAW_CURTIS_W__ALGORITHM(uint64_t, double, Inverse_Discrete_Cosine_Transform_2, sqrt)

double Integrate_Clenshaw_Curtis_via_xw(double func(double, void *), double xmin, double xmax,
                                        uint64_t n_int, void *other_param,
                                        double *x_int, double *w_int)
INTEGRATE_CLENSHAW_CURTIS_VIA_XW__ALGORITHM(uint64_t, double)

double Integrate_Clenshaw_Curtis(double func(double, void *), double xmin, double xmax,
                                 uint64_t n_int, void *other_param)
INTEGRATE_CLENSHAW_CURTIS__ALGORITHM(double,
                                     Integrate_Clenshaw_Curtis_x,
                                     Integrate_Clenshaw_Curtis_w,
                                     Integrate_Clenshaw_Curtis_via_xw)

/* uint32_t & float */
bool Integrate_Clenshaw_Curtis_x_f32(uint32_t n_int, float *x_int)
INTEGRATE_CLENSHAW_CURTIS_X__ALGORITHM(uint32_t, float, cosf)

bool Integrate_Clenshaw_Curtis_w_f32(uint32_t n_int, float *w_int)
INTEGRATE_CLENSHAW_CURTIS_W__ALGORITHM(uint32_t, float, Inverse_Discrete_Cosine_Transform_2_f32, sqrtf)

float Integrate_Clenshaw_Curtis_via_xw_f32(float func(float, void *), float xmin, float xmax,
                                           uint32_t n_int, void *other_param,
                                           float *x_int, float *w_int)
INTEGRATE_CLENSHAW_CURTIS_VIA_XW__ALGORITHM(uint32_t, float)

float Integrate_Clenshaw_Curtis_f32(float func(float, void *), float xmin, float xmax,
                                    uint32_t n_int, void *other_param)
INTEGRATE_CLENSHAW_CURTIS__ALGORITHM(float,
                                     Integrate_Clenshaw_Curtis_x_f32,
                                     Integrate_Clenshaw_Curtis_w_f32,
                                     Integrate_Clenshaw_Curtis_via_xw_f32)

/* uint64_t & long double */
bool Integrate_Clenshaw_Curtis_x_fl(uint64_t n_int, long double *x_int)
INTEGRATE_CLENSHAW_CURTIS_X__ALGORITHM(uint64_t, long double, cosl)

bool Integrate_Clenshaw_Curtis_w_fl(uint64_t n_int, long double *w_int)
INTEGRATE_CLENSHAW_CURTIS_W__ALGORITHM(uint64_t, long double, Inverse_Discrete_Cosine_Transform_2_fl, sqrtl)

long double Integrate_Clenshaw_Curtis_via_xw_fl(long double func(long double, void *), long double xmin, long double xmax,
                                                uint64_t n_int, void *other_param,
                                                long double *x_int, long double *w_int)
INTEGRATE_CLENSHAW_CURTIS_VIA_XW__ALGORITHM(uint64_t, long double)

long double Integrate_Clenshaw_Curtis_fl(long double func(long double, void *), long double xmin, long double xmax,
                                         uint64_t n_int, void *other_param)
INTEGRATE_CLENSHAW_CURTIS__ALGORITHM(long double,
                                     Integrate_Clenshaw_Curtis_x_fl,
                                     Integrate_Clenshaw_Curtis_w_fl,
                                     Integrate_Clenshaw_Curtis_via_xw_fl)

#ifdef ENABLE_QUADPRECISION
/* uint64_t & __float128 */
bool Integrate_Clenshaw_Curtis_x_f128(uint64_t n_int, __float128 *x_int)
INTEGRATE_CLENSHAW_CURTIS_X__ALGORITHM(uint64_t, __float128, cosq)

bool Integrate_Clenshaw_Curtis_w_f128(uint64_t n_int, __float128 *w_int)
INTEGRATE_CLENSHAW_CURTIS_W__ALGORITHM(uint64_t, __float128, Inverse_Discrete_Cosine_Transform_2_f128, sqrtq)

__float128 Integrate_Clenshaw_Curtis_via_xw_f128(__float128 func(__float128, void *), __float128 xmin, __float128 xmax,
                                                 uint64_t n_int, void *other_param,
                                                 __float128 *x_int, __float128 *w_int)
INTEGRATE_CLENSHAW_CURTIS_VIA_XW__ALGORITHM(uint64_t, __float128)

__float128 Integrate_Clenshaw_Curtis_f128(__float128 func(__float128, void *), __float128 xmin, __float128 xmax,
                                          uint64_t n_int, void *other_param)
INTEGRATE_CLENSHAW_CURTIS__ALGORITHM(__float128,
                                     Integrate_Clenshaw_Curtis_x_f128,
                                     Integrate_Clenshaw_Curtis_w_f128,
                                     Integrate_Clenshaw_Curtis_via_xw_f128)
#endif /* ENABLE_QUADPRECISION */