/* coding: utf-8 */
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<stdint.h>
#include<stdbool.h>
#include"fft.h"
#include"../basic/basic.h"

#define IDCT2_NAIVE__ALGORITHM(real_type, \
                        Discrete_Cosine_Transform_Weight, \
                        sqrt) \
{ \
    if (n == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n = 0.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return false; \
    } \
    if (arr == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: array pointer (arr) is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    real_type *w = (real_type*)malloc(4 * (uint64_t)n * sizeof(real_type)); \
    if (w == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for weight points.", __func__, 4 * (uint64_t)n * sizeof(real_type)); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    real_type *origin = (real_type*)malloc((uint64_t)n * sizeof(real_type)); \
    if (origin == NULL){ \
        free(w); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for original array.", __func__, (uint64_t)n * sizeof(real_type)); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    memcpy(origin, arr, (uint64_t)n * sizeof(real_type)); \
    origin[0] *= sqrt(0.5); \
    uint64_t i, j, /*n2 = (uint64_t)n << 1,*/ n4 = (uint64_t)n << 2; \
    for (i=0; i<n; i++){ \
        arr[0] = 0; \
    } \
    Discrete_Cosine_Transform_Weight(n, w); \
 \
    real_type scale = sqrt(2/(real_type)n); \
    for (i=0; i<n; i++){ \
        real_type sum = 0; \
        uint64_t angle_index = 0, angle_gap = (i << 1) + 1; \
        for (j=0; j<n; j++){ \
            sum += origin[j] * w[angle_index]; \
            angle_index = (angle_index + angle_gap) % n4; \
        } \
        arr[i] = sum * scale; \
    } \
 \
    free(w); \
    free(origin); \
    return true; \
} \

bool Inverse_Discrete_Cosine_Transform_2_Naive(uint64_t n, double *arr)
IDCT2_NAIVE__ALGORITHM(double, Discrete_Cosine_Transform_Weight, sqrt)

bool Inverse_Discrete_Cosine_Transform_2_Naive_f32(uint32_t n, float *arr)
IDCT2_NAIVE__ALGORITHM(float, Discrete_Cosine_Transform_Weight_f32, sqrtf)

bool Inverse_Discrete_Cosine_Transform_2_Naive_fl(uint64_t n, long double *arr)
IDCT2_NAIVE__ALGORITHM(long double, Discrete_Cosine_Transform_Weight_fl, sqrtl)

#ifdef ENABLE_QUADPRECISION
bool Inverse_Discrete_Cosine_Transform_2_Naive_f128(uint64_t n, __float128 *arr)
IDCT2_NAIVE__ALGORITHM(__float128, Discrete_Cosine_Transform_Weight_f128, sqrtq)
#endif /* ENABLE_QUADPRECISION */