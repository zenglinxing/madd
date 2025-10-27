/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./statistics/linspace.c
*/
#include<stdlib.h>
#include<float.h>
#include<math.h>
#include<stdint.h>
#include<stdbool.h>

#include"statistics.h"
#include"../basic/basic.h"

#define LINSPACE__ALGORITHM(num_type, fabs, float_min, float_max) \
{ \
    if (n == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n is 0.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (arr == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: arr is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    if (n == 1){ \
        arr[0] = start; \
        return true; \
    } \
    if (start == end){ \
        for (uint64_t i=0; i<n; i++){ \
            arr[i] = start; \
        } \
        return true; \
    } \
	uint64_t n_gap = (include_end) ? n - 1 : n; \
    double diff = end - start, diff_abs = fabs(diff); \
    double n_gap_lower = diff_abs / float_min; \
    if (n >= n_gap_lower){ \
        double gap = diff / n_gap; \
        for (uint64_t i=0; i<n; i++){ \
            arr[i] = start + i * gap; \
        } \
    }else{ \
        double n_gap_upper = float_max / diff_abs, f_gap = n_gap; \
        for (uint64_t i=0; i<n; i++){ \
            double increment; \
            if (i <= n_gap_upper){ \
                increment = (i * diff) / n_gap; \
            }else{ \
                double rate = i / f_gap; \
                increment = diff * rate; \
            } \
            arr[i] = start + increment; \
        } \
    } \
    if (include_end) arr[n-1] = end; \
    return true; \
} \

bool Linspace(double start, double end, uint64_t n, double *arr, bool include_end)
LINSPACE__ALGORITHM(double, fabs, DBL_MIN, DBL_MAX)

bool Linspace_f32(float start, float end, uint64_t n, float *arr, bool include_end)
LINSPACE__ALGORITHM(float, fabsf, FLT_MIN, FLT_MAX)

bool Linspace_fl(long double start, long double end, uint64_t n, long double *arr, bool include_end)
LINSPACE__ALGORITHM(long double, fabsl, LDBL_MIN, LDBL_MAX)

#ifdef ENABLE_QUADPRECISION
bool Linspace_f128(__float128 start, __float128 end, uint64_t n, __float128 *arr, bool include_end)
LINSPACE__ALGORITHM(__float128, fabsq, FLT128_MIN, FLT128_MAX)
#endif /* ENABLE_QUADPRECISION */