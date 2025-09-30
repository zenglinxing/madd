/* coding: utf-8 */
/*
Author: Lin-Xing Zeng
Email:  jasonphysics@outlook.com | jasonphysics19@gmail.com

This file is part of Math Addition, in ./fft/fft-weight.c
*/
#include<string.h>
#include<stdlib.h>
#include<stdint.h>

#include"fft.h"
#include"../basic/basic.h"

/* This code is suitable for embedded devices */
#define FFT_W_QUAD_SYMMETRY__ALGORITHM(Cnum, float_type, cos, sin) \
{ \
    if (n == 0){ \
        Madd_Error_Add(MADD_ERROR, L"Fast_Fourier_Transform_w: n = 0."); \
        return; \
    } \
    w[0].real = 1; \
    w[0].imag = 0; \
    if (n == 1){ \
        return; \
    } \
    uint64_t n2 = n/2, n4 = n/4, n3=n2+n4 /* n*3/4 */, n8 = n/8, i; \
    float_type angle = sign * 2 * _CONSTANT_PI / n, real, imag, angle_i; \
    uint64_t n_mod_4 = n & 0b11; /* n % 4 */ \
    if (n_mod_4 == 0){ \
        for (i=0; i<n8+1; i++){ \
            angle_i = angle * i; \
            real = cos(angle_i); \
            imag = sin(angle_i); \
            w[i].real    = w[n4+i].imag = w[n4-i].imag = real; \
            w[n2+i].real = w[n2-i].real = w[n3+i].imag = w[n3-i].imag = -real; \
            w[i].imag    = w[n2-i].imag = w[n4-i].real = w[n3+i].real = imag; \
            w[n4+i].real = w[n2+i].imag = w[n3-i].real = -imag; \
            if (i!=0){ \
                w[n-i].real = real; \
                w[n-i].imag = -imag; \
            } \
        } \
    }else if (n_mod_4 == 0b10){ \
        for (i=0; i<n4+1; i++){ \
            angle_i = angle * i; \
            real = cos(angle_i); \
            imag = sin(angle_i); \
            w[i].real    = real; \
            w[n2+i].real = w[n2-i].real = -real; \
            w[i].imag    = w[n2-i].imag = imag; \
            w[n2+i].imag = -imag; \
            if (i!=0){ \
                w[n-i].real = real; \
                w[n-i].imag = -imag; \
            } \
        } \
    }else{ \
        for (i=0; i<n2; i++){ \
            angle_i = angle * i; \
            real = cos(angle_i); \
            imag = sin(angle_i); \
            w[i+1].real = w[n-1-i].real = real; \
            w[i+1].imag = imag; \
            w[n-1-i].imag = -imag; \
        } \
    } \
} \

#define FFT_W__ALGORITHM(Cnum, Cnum_Value, cos, sin, real_type) \
{ \
    if (n == 0){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: n = 0.", __func__); \
        Madd_Error_Add(MADD_WARNING, error_info); \
        return; \
    } \
    if (w == NULL){ \
        wchar_t error_info[MADD_ERROR_INFO_LEN]; \
        swprintf(error_info, MADD_ERROR_INFO_LEN, L"%hs: weight pointer (w) is NULL.", __func__); \
        Madd_Error_Add(MADD_ERROR, error_info); \
        return; \
    } \
 \
    w[0] = Cnum_Value(1, 0); \
    real_type real, imag, angle_base = sign * 2 * _CONSTANT_PI / n, angle; \
    uint64_t n2 = n / 2, i; \
    if (n & 0b1){ /* n is odd */ \
        for (i=0; i<n2; i++){ \
            uint64_t id1 = i + 1, id2 = n - id1; \
            angle = angle_base * id1; \
            real = cos(angle); \
            imag = sin(angle); \
            w[id1].real = w[id2].real = real; \
            w[id1].imag = imag; \
            w[id2].imag = -imag; \
        } \
    }else{ /* n is even */ \
        w[n2] = Cnum_Value(-1, 0); \
        for (i=1; i<n2; i++){ \
            uint64_t id1 = i, id2 = n - id1; \
            angle = angle_base * id1; \
            real = cos(angle); \
            imag = sin(angle); \
            w[id1].real = w[id2].real = real; \
            w[id1].imag = imag; \
            w[id2].imag = -imag; \
        } \
        if ((n & 0b11) == 0){ /* (n % 4) == 0 */ \
            uint64_t n4 = n >> 2; \
            w[n4].imag = sign; \
            w[3*n4].imag = -sign; \
            w[n4].real = w[3*n4].real = 0; \
        } \
    } \
} \

void Fast_Fourier_Transform_Weight(uint64_t n, Cnum *w, int sign)
FFT_W__ALGORITHM(Cnum, Cnum_Value, cos, sin, double)

void Fast_Fourier_Transform_Weight_c32(uint64_t n, Cnum32 *w, int sign)
FFT_W__ALGORITHM(Cnum32, Cnum_Value_c32, cosf, sinf, float)

void Fast_Fourier_Transform_Weight_cl(uint64_t n, Cnuml *w, int sign)
FFT_W__ALGORITHM(Cnuml, Cnum_Value_cl, cosl, sinl, long double)

#ifdef ENABLE_QUADPRECISION
void Fast_Fourier_Transform_Weight_c128(uint64_t n, Cnum128 *w, int sign)
FFT_W__ALGORITHM(Cnum128, Cnum_Value_c128, cosq, sinq, __float128)
#endif /* ENABLE_QUADPRECISION */