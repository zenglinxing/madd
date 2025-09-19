/* coding: utf-8 */
//#include<stdio.h>
#include<stdlib.h>
#include<string.h>
#include<math.h>
#include<stdint.h>
#include<stdbool.h>
#include"fft.h"
#include"../basic/basic.h"

#define DCT2_NAIVE__ALGORITHM(integer_type, real_type, \
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
    real_type *w = (real_type*)malloc(4 * n *sizeof(real_type)); \
    if (w == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for weight points.", __func__, 4 * n * sizeof(real_type)); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
    real_type *original = (real_type*)malloc(n * sizeof(real_type)); \
    if (original == NULL){ \
        free(w); \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: unable to malloc %llu bytes for original array.", __func__, n * sizeof(real_type)); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    integer_type i, j; \
    memcpy(original, arr, n * sizeof(real_type)); \
    for (i=0; i<n; i++){ \
        arr[i] = 0; \
    } \
 \
    Discrete_Cosine_Transform_Weight(n, w); \
 \
    uint64_t angle_index, angle_gap, n2 = n << 1, n4 = n << 2; \
    real_type factor = sqrt(2 / (real_type)n); \
    for (i=0; i<n; i++){ \
        angle_index = i; /* angle_index = i * (2*j + 1) */ \
        angle_gap = i << 1; \
        real_type sum = 0; \
        for (j=0; j<n; j++){ \
            sum += original[j] * w[angle_index]; \
            /*printf("%llu %llu %llu %f %f\n", i, j, angle_index, original[j], w[angle_index]);*/ \
            angle_index = (angle_index + angle_gap) % n4; \
        } \
        arr[i] = sum * factor; \
    } \
    arr[0] *= sqrt(0.5); \
 \
    free(w); \
    free(original); \
    return true; \
} \

bool Discrete_Cosine_Transform_2_Naive(uint64_t n, double *arr)
DCT2_NAIVE__ALGORITHM(uint64_t, double, Discrete_Cosine_Transform_Weight, sqrt)

bool Discrete_Cosine_Transform_2_Naive_f32(uint32_t n, float *arr)
DCT2_NAIVE__ALGORITHM(uint32_t, float, Discrete_Cosine_Transform_Weight_f32, sqrtf)

bool Discrete_Cosine_Transform_2_Naive_fl(uint64_t n, long double *arr)
DCT2_NAIVE__ALGORITHM(uint64_t, long double, Discrete_Cosine_Transform_Weight_fl, sqrtl)

#ifdef ENABLE_QUADPRECISION
bool Discrete_Cosine_Transform_2_Naive_f128(uint64_t n, __float128 *arr)
DCT2_NAIVE__ALGORITHM(uint64_t, __float128, Discrete_Cosine_Transform_Weight_f128, sqrtq)
#endif /* ENABLE_QUADPRECISION */