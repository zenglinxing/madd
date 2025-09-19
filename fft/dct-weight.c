/* coding: utf-8 */
//#include<stdio.h>
#include<math.h>
#include<stdint.h>
#include<stdbool.h>
#include"fft.h"
#include"../basic/basic.h"

#define DCT_WEIGHT__ALGORITHM(integer_type, real_type, sqrt, cos, _CONSTANT_PI) \
{ \
    if (n == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n = 0.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return false; \
    } \
    if (w == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: weight array pointer (w) is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return false; \
    } \
 \
    integer_type i, n2 = (uint64_t)n << 1, n4 = (uint64_t)n << 2; \
    real_type rate = _CONSTANT_PI / n2; \
    for (i=0; i<n4; i++){ \
        w[i] = cos(rate * i); \
        /*printf("%llu %f\n", i, w[i]);*/ \
    } \
    return true; \
} \

bool Discrete_Cosine_Transform_Weight(uint64_t n, double *w)
DCT_WEIGHT__ALGORITHM(uint64_t, double, sqrt, cos, _CONSTANT_PI)

bool Discrete_Cosine_Transform_Weight_f32(uint32_t n, float *w)
DCT_WEIGHT__ALGORITHM(uint32_t, float, sqrtf, cosf, _CONSTANT_PI)

bool Discrete_Cosine_Transform_Weight_fl(uint64_t n, long double *w)
DCT_WEIGHT__ALGORITHM(uint64_t, long double, sqrtl, cosl, _CONSTANT_PI)

#ifdef ENABLE_QUADPRECISION
bool Discrete_Cosine_Transform_Weight_f128(uint64_t n, __float128 *w)
DCT_WEIGHT__ALGORITHM(uint64_t, __float128, sqrtq, cosq, _CONSTANT_PI)
#endif /* ENABLE_QUADPRECISION */